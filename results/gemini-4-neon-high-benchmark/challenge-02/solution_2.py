"""Entanglement-profile-constrained VQE using TensorCircuit-NG (JAX backend)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _apply_sublayer(
    circuit: tc.Circuit,
    params: jnp.ndarray,
    offset: int,
    n_qubits: int,
    bonds: list[tuple[int, int]],
) -> int:
    """Apply single-qubit RY/RZ rotations and commuting XX+YY+ZZ bond gates."""
    for q in range(n_qubits):
        circuit.ry(q, theta=params[offset + q])
        circuit.rz(q, theta=params[offset + n_qubits + q])
    offset += 2 * n_qubits

    for idx, (i, j) in enumerate(bonds):
        base = offset + 3 * idx
        circuit.rxx(i, j, theta=2.0 * params[base])
        circuit.ryy(i, j, theta=2.0 * params[base + 1])
        circuit.rzz(i, j, theta=2.0 * params[base + 2])
    return offset + 3 * len(bonds)


def run_solution(config: dict) -> dict[str, np.ndarray]:
    """Run entanglement-profile-constrained VQE for the 1D staggered XXZ chain."""
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    tc.set_contractor("greedy")

    n_qubits = int(config["n_qubits"])
    zz_anisotropy = float(config["zz_anisotropy"])
    staggered_field = float(config["staggered_field"])
    n_layers = int(config["n_layers"])
    subsystem_size = int(config["subsystem_size"])
    target_entropies = jnp.asarray(
        np.asarray(config["target_entropies"], dtype=np.float64), dtype=jnp.float64
    )
    entropy_weight = float(config["entropy_weight"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    if n_layers % 2 != 0:
        raise ValueError(f"n_layers must be even, got {n_layers}")
    n_blocks = n_layers // 2
    if target_entropies.shape != (n_blocks,):
        raise ValueError(
            f"target_entropies shape {target_entropies.shape} != ({n_blocks},)"
        )

    even_bonds = [(i, i + 1) for i in range(0, n_qubits - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n_qubits - 1, 2)]
    params_per_block = 4 * n_qubits + 3 * (len(even_bonds) + len(odd_bonds))
    subsystem_keep = list(range(subsystem_size))
    chain_graph = tc.templates.graphs.Line1D(n_qubits, pbc=False)

    init_circuit = tc.Circuit(n_qubits)
    for q in range(1, n_qubits, 2):
        init_circuit.x(q)
    psi0 = init_circuit.state()

    def apply_block(
        psi: jnp.ndarray, block_params: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        circuit = tc.Circuit(n_qubits, inputs=psi)
        offset = _apply_sublayer(circuit, block_params, 0, n_qubits, even_bonds)
        _apply_sublayer(circuit, block_params, offset, n_qubits, odd_bonds)
        psi_next = circuit.state()
        rho_a = tc.quantum.reduced_density_matrix(
            psi_next, subsystem_to_keep=subsystem_keep
        )
        s2 = tc.quantum.renyi_entropy(rho_a, k=2)
        return psi_next, s2

    def loss_and_metrics(
        params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        psi_final, entropies = jax.lax.scan(apply_block, psi0, params)
        final_circuit = tc.Circuit(n_qubits, inputs=psi_final)
        energy = tc.templates.measurements.heisenberg_measurements(
            final_circuit,
            chain_graph,
            hzz=zz_anisotropy,
            hxx=1.0,
            hyy=1.0,
            reuse=True,
        )
        if staggered_field != 0.0:
            for q in range(n_qubits):
                sign = 1.0 if q % 2 == 0 else -1.0
                energy = energy + (staggered_field * sign) * jnp.real(
                    final_circuit.expectation((tc.gates.z(), [q]), reuse=True)
                )
        energy_density = energy / float(n_qubits)
        entropy_mse = jnp.mean((entropies - target_entropies) ** 2)
        loss = energy_density + entropy_weight * entropy_mse
        return loss, (energy_density, entropy_mse, entropies)

    value_and_grad_fn = jax.value_and_grad(loss_and_metrics, has_aux=True)
    optimizer = optax.adam(learning_rate=learning_rate)

    @jax.jit
    def optimize(
        init_params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        init_opt_state = optimizer.init(init_params)

        def step(carry, _):
            params, opt_state = carry
            (loss, (energy_density, entropy_mse, entropies)), grads = (
                value_and_grad_fn(params)
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, energy_density, entropy_mse, entropies)

        _, history = jax.lax.scan(
            step, (init_params, init_opt_state), None, length=max_steps
        )
        return history

    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    init_params = jnp.asarray(
        rng.normal(0.0, 0.02, size=(n_blocks, params_per_block)), dtype=jnp.float64
    )

    loss_h, energy_density_h, entropy_mse_h, entropy_h = optimize(init_params)
    jax.block_until_ready(loss_h)

    return {
        "energy_density_history": np.asarray(energy_density_h, dtype=np.float64),
        "loss_history": np.asarray(loss_h, dtype=np.float64),
        "entropy_mse_history": np.asarray(entropy_mse_h, dtype=np.float64),
        "entropy_history": np.asarray(entropy_h, dtype=np.float64),
    }
