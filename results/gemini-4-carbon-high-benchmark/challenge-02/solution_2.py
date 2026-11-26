from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")


def _build_xxz_pauli_terms(n_qubits: int, zz_anisotropy: float, staggered_field: float):
    structures = []
    weights = []
    for i in range(n_qubits - 1):
        for pauli_idx, coeff in ((1, 1.0), (2, 1.0), (3, zz_anisotropy)):
            row = [0] * n_qubits
            row[i] = pauli_idx
            row[i + 1] = pauli_idx
            structures.append(row)
            weights.append(coeff)
    for i in range(n_qubits):
        row = [0] * n_qubits
        row[i] = 3
        structures.append(row)
        weights.append(staggered_field * ((-1.0) ** i))
    return structures, weights


@lru_cache(maxsize=8)
def _compile_vqe_runner(n_qubits: int, n_layers: int, subsystem_size: int, max_steps: int):
    if n_layers % 2 != 0:
        raise ValueError("n_layers must be even for even+odd brickwork blocks.")
    n_blocks = n_layers // 2
    even_bonds = tuple((i, i + 1) for i in range(0, n_qubits - 1, 2))
    odd_bonds = tuple((i, i + 1) for i in range(1, n_qubits - 1, 2))
    keep_sites = tuple(range(subsystem_size))
    params_per_block = 4 * n_qubits + 3 * (len(even_bonds) + len(odd_bonds))

    init_circuit = tc.Circuit(n_qubits)
    for q in range(1, n_qubits, 2):
        init_circuit.x(q)
    psi0 = init_circuit.state()

    def apply_block(psi, block_params):
        c = tc.Circuit(n_qubits, inputs=psi)
        idx = 0
        for q in range(n_qubits):
            c.ry(q, theta=block_params[idx])
            c.rz(q, theta=block_params[idx + 1])
            idx += 2
        for q0, q1 in even_bonds:
            c.rxx(q0, q1, theta=2.0 * block_params[idx])
            c.ryy(q0, q1, theta=2.0 * block_params[idx + 1])
            c.rzz(q0, q1, theta=2.0 * block_params[idx + 2])
            idx += 3
        for q in range(n_qubits):
            c.ry(q, theta=block_params[idx])
            c.rz(q, theta=block_params[idx + 1])
            idx += 2
        for q0, q1 in odd_bonds:
            c.rxx(q0, q1, theta=2.0 * block_params[idx])
            c.ryy(q0, q1, theta=2.0 * block_params[idx + 1])
            c.rzz(q0, q1, theta=2.0 * block_params[idx + 2])
            idx += 3
        psi_next = c.state()
        rho_a = tc.quantum.reduced_density_matrix(
            psi_next, subsystem_to_keep=keep_sites
        )
        s2 = jnp.real(tc.quantum.renyi_entropy(rho_a, k=2))
        return psi_next, s2

    @jax.jit
    def run_optimization(init_params, h_coo, target_entropies, entropy_weight, lr):
        def loss_fn(params):
            psi_final, entropies = jax.lax.scan(apply_block, psi0, params)
            c_final = tc.Circuit(n_qubits, inputs=psi_final)
            energy = tc.templates.measurements.operator_expectation(c_final, h_coo)
            energy_density = jnp.real(energy) / n_qubits
            entropy_mse = jnp.mean((entropies - target_entropies) ** 2)
            loss = energy_density + entropy_weight * entropy_mse
            return loss, (energy_density, entropy_mse, entropies)

        opt = optax.adam(learning_rate=lr)
        opt_state0 = opt.init(init_params)
        vg_fn = tc.backend.value_and_grad(loss_fn, has_aux=True)

        def step_fn(carry, _):
            params, opt_state = carry
            (loss, (energy_density, entropy_mse, entropies)), grads = vg_fn(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, energy_density, entropy_mse, entropies)

        _, histories = jax.lax.scan(
            step_fn, (init_params, opt_state0), None, length=max_steps
        )
        return histories

    return run_optimization, n_blocks, params_per_block


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    zz_anisotropy = float(config["zz_anisotropy"])
    staggered_field = float(config["staggered_field"])
    n_layers = int(config["n_layers"])
    subsystem_size = int(config["subsystem_size"])
    target_entropies = jnp.asarray(config["target_entropies"], dtype=jnp.float64)
    entropy_weight = jnp.asarray(config["entropy_weight"], dtype=jnp.float64)
    max_steps = int(config["max_steps"])
    learning_rate = jnp.asarray(config["learning_rate"], dtype=jnp.float64)

    runner, n_blocks, params_per_block = _compile_vqe_runner(
        n_qubits, n_layers, subsystem_size, max_steps
    )
    structures, weights = _build_xxz_pauli_terms(
        n_qubits, zz_anisotropy, staggered_field
    )
    h_coo = tc.quantum.PauliStringSum2COO(structures, weights)

    rng = np.random.default_rng(config.get("seed", 42))
    init_params = jnp.asarray(
        rng.normal(loc=0.0, scale=0.02, size=(n_blocks, params_per_block)),
        dtype=jnp.float64,
    )

    loss_hist, ed_hist, emse_hist, ent_hist = runner(
        init_params, h_coo, target_entropies, entropy_weight, learning_rate
    )

    return {
        "energy_density_history": np.asarray(ed_hist, dtype=np.float64),
        "loss_history": np.asarray(loss_hist, dtype=np.float64),
        "entropy_mse_history": np.asarray(emse_hist, dtype=np.float64),
        "entropy_history": np.asarray(ent_hist, dtype=np.float64),
    }
