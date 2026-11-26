"""Entanglement-profile-constrained VQE for the 12-qubit XXZ chain."""

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _build_xxz_hamiltonian(
    n_qubits: int, zz_anisotropy: float, staggered_field: float
) -> Any:
    """Build the open-boundary XXZ Hamiltonian with staggered Z field as COO sparse."""
    pauli_strings: List[List[int]] = []
    weights: List[float] = []
    for i in range(n_qubits - 1):
        for pauli_idx, weight in ((1, 1.0), (2, 1.0), (3, float(zz_anisotropy))):
            pstr = [0] * n_qubits
            pstr[i] = pauli_idx
            pstr[i + 1] = pauli_idx
            pauli_strings.append(pstr)
            weights.append(weight)
    for i in range(n_qubits):
        pstr = [0] * n_qubits
        pstr[i] = 3
        pauli_strings.append(pstr)
        weights.append(float(staggered_field) * ((-1) ** i))
    return tc.quantum.PauliStringSum2COO(pauli_strings, weights, numpy=False)


def _apply_sublayer(
    c: tc.Circuit,
    n_qubits: int,
    sub_params: Any,
    bonds: List[Tuple[int, int]],
    pauli_mats: Tuple[Any, Any, Any],
) -> None:
    """Apply RY-RZ single-qubit rotations and exp[-i(txx XX + tyy YY + tzz ZZ)] gates."""
    xx_mat, yy_mat, zz_mat = pauli_mats
    for q in range(n_qubits):
        c.ry(q, theta=sub_params[q])
        c.rz(q, theta=sub_params[n_qubits + q])
    offset = 2 * n_qubits
    for b_idx, (q0, q1) in enumerate(bonds):
        base = offset + 3 * b_idx
        c.exp1(q0, q1, unitary=xx_mat, theta=sub_params[base])
        c.exp1(q0, q1, unitary=yy_mat, theta=sub_params[base + 1])
        c.exp1(q0, q1, unitary=zz_mat, theta=sub_params[base + 2])


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Optimize the entanglement-profile-constrained VQE ansatz with Adam."""
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    zz_anisotropy = float(config["zz_anisotropy"])
    staggered_field = float(config["staggered_field"])
    n_layers = int(config["n_layers"])
    subsystem_size = int(config["subsystem_size"])
    target_entropies = jnp.asarray(config["target_entropies"], dtype=jnp.float64)
    entropy_weight = float(config["entropy_weight"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    if n_layers % 2 != 0:
        raise ValueError(f"n_layers must be even for even+odd blocks, got {n_layers}")
    n_blocks = n_layers // 2
    if target_entropies.shape != (n_blocks,):
        raise ValueError(
            f"target_entropies shape {target_entropies.shape} != ({n_blocks},)"
        )

    hamiltonian = _build_xxz_hamiltonian(n_qubits, zz_anisotropy, staggered_field)

    c_init = tc.Circuit(n_qubits)
    for i in range(1, n_qubits, 2):
        c_init.x(i)
    psi0 = c_init.state()

    pauli_mats = (
        tc.array_to_tensor(tc.gates._xx_matrix),
        tc.array_to_tensor(tc.gates._yy_matrix),
        tc.array_to_tensor(tc.gates._zz_matrix),
    )
    even_bonds = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    even_size = 2 * n_qubits + 3 * len(even_bonds)
    block_size = even_size + 2 * n_qubits + 3 * len(odd_bonds)
    traceout = list(range(subsystem_size, n_qubits))

    def apply_block(psi: Any, block_params: Any) -> Tuple[Any, Any]:
        c = tc.Circuit(n_qubits, inputs=psi)
        _apply_sublayer(c, n_qubits, block_params[:even_size], even_bonds, pauli_mats)
        _apply_sublayer(c, n_qubits, block_params[even_size:], odd_bonds, pauli_mats)
        psi_next = c.state()
        rho_a = tc.quantum.reduced_density_matrix(
            psi_next, subsystems_to_trace_out=traceout
        )
        s2 = jnp.real(tc.quantum.renyi_entropy(rho_a, k=2))
        return psi_next, s2

    def loss_fn(params: Any) -> Tuple[Any, Tuple[Any, Any, Any]]:
        psi_final, entropies = jax.lax.scan(apply_block, psi0, params)
        c_fin = tc.Circuit(n_qubits, inputs=psi_final)
        energy = tc.templates.measurements.sparse_expectation(c_fin, hamiltonian)
        energy_density = jnp.real(energy) / n_qubits
        entropy_mse = jnp.mean((entropies - target_entropies) ** 2)
        total_loss = energy_density + entropy_weight * entropy_mse
        return total_loss, (energy_density, entropy_mse, entropies)

    optimizer = optax.adam(learning_rate=learning_rate)
    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def step_fn(carry: Tuple[Any, Any], _: Any) -> Tuple[Tuple[Any, Any], Tuple[Any, ...]]:
        params, opt_state = carry
        (loss, (e_dens, ent_mse, entropies)), grads = value_and_grad_fn(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), (e_dens, loss, ent_mse, entropies)

    @jax.jit
    def run_optimization(p0: Any, os0: Any) -> Tuple[Any, ...]:
        _, history = jax.lax.scan(step_fn, (p0, os0), None, length=max_steps)
        return history

    rng = np.random.default_rng(int(config.get("seed", 42)))
    params0 = jnp.asarray(
        rng.normal(0.0, 0.02, size=(n_blocks, block_size)), dtype=jnp.float64
    )
    opt_state0 = optimizer.init(params0)

    e_hist, l_hist, mse_hist, s2_hist = run_optimization(params0, opt_state0)
    return {
        "energy_density_history": np.asarray(e_hist, dtype=np.float64),
        "loss_history": np.asarray(l_hist, dtype=np.float64),
        "entropy_mse_history": np.asarray(mse_hist, dtype=np.float64),
        "entropy_history": np.asarray(s2_hist, dtype=np.float64),
    }
