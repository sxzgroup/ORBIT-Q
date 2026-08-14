import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _pauli_hamiltonian(n, zz_anisotropy, staggered_field):
    strings, weights = [], []
    for i in range(n - 1):
        xx = [0] * n
        xx[i] = 1
        xx[i + 1] = 1
        strings.append(xx)
        weights.append(1.0)
        yy = [0] * n
        yy[i] = 2
        yy[i + 1] = 2
        strings.append(yy)
        weights.append(1.0)
        zz = [0] * n
        zz[i] = 3
        zz[i + 1] = 3
        strings.append(zz)
        weights.append(float(zz_anisotropy))
    for i in range(n):
        z = [0] * n
        z[i] = 3
        strings.append(z)
        weights.append(float(staggered_field) * ((-1.0) ** i))
    return tc.quantum.PauliStringSum2COO(strings, weight=weights, numpy=False)


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("plain")
    backend = tc.backend

    n = int(config["n_qubits"])
    n_blocks = int(config["n_layers"]) // 2
    sub = int(config["subsystem_size"])
    n_even = n // 2
    n_odd = n_even - 1
    params_per_block = 4 * n + 3 * (n_even + n_odd)
    max_steps = int(config["max_steps"])
    entropy_weight = float(config["entropy_weight"])
    targets = jnp.asarray(config["target_entropies"], dtype=jnp.float32)
    ham = _pauli_hamiltonian(n, config["zz_anisotropy"], config["staggered_field"])
    xx_mat = tc.gates._xx_matrix
    yy_mat = tc.gates._yy_matrix
    zz_mat = tc.gates._zz_matrix

    def initial_state():
        circuit = tc.Circuit(n)
        for i in range(1, n, 2):
            circuit.x(i)
        return circuit.state()

    def apply_rotations(circuit, thetas, offset):
        for i in range(n):
            circuit.ry(i, theta=thetas[offset + i])
        offset += n
        for i in range(n):
            circuit.rz(i, theta=thetas[offset + i])
        return offset + n

    def apply_interactions(circuit, thetas, offset, start, count):
        for k in range(count):
            left = start + 2 * k
            circuit.exp1(left, left + 1, theta=thetas[offset], unitary=xx_mat)
            circuit.exp1(left, left + 1, theta=thetas[offset + 1], unitary=yy_mat)
            circuit.exp1(left, left + 1, theta=thetas[offset + 2], unitary=zz_mat)
            offset += 3
        return offset

    def apply_block(psi, thetas):
        circuit = tc.Circuit(n, inputs=psi)
        offset = apply_rotations(circuit, thetas, 0)
        offset = apply_interactions(circuit, thetas, offset, 0, n_even)
        offset = apply_rotations(circuit, thetas, offset)
        apply_interactions(circuit, thetas, offset, 1, n_odd)
        return circuit.state()

    def half_chain_s2(psi):
        rho = tc.quantum.reduced_density_matrix(
            psi, subsystem_to_keep=list(range(sub))
        )
        return backend.real(tc.quantum.renyi_entropy(rho, k=2))

    def energy(psi):
        ket = backend.reshape(psi, [-1, 1])
        hpsi = backend.sparse_dense_matmul(ham, ket)
        return backend.real(backend.adjoint(ket) @ hpsi)[0, 0]

    def block_step(psi, thetas):
        nxt = apply_block(psi, thetas)
        return nxt, half_chain_s2(nxt)

    def observables(params):
        blocks = params.reshape((n_blocks, params_per_block))
        psi, entropies = jax.lax.scan(block_step, initial_state(), blocks)
        dens = energy(psi) / n
        mse = jnp.mean((entropies - targets) ** 2)
        return dens + entropy_weight * mse, (dens, mse, entropies)

    value_and_grad = jax.value_and_grad(observables, has_aux=True)
    optimizer = optax.adam(float(config["learning_rate"]))
    rng = np.random.default_rng(0)
    params = jnp.asarray(
        rng.normal(0.0, 0.02, size=(n_blocks * params_per_block,)).astype(np.float32)
    )
    opt_state = optimizer.init(params)

    def train(params, opt_state):
        def step(carry, _):
            params, opt_state = carry
            (loss, (dens, mse, entropies)), grads = value_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (dens, loss, mse, entropies)

        return jax.lax.scan(step, (params, opt_state), None, length=max_steps)

    _, (dens_h, loss_h, mse_h, ent_h) = jax.jit(train)(params, opt_state)
    return {
        "energy_density_history": np.asarray(dens_h),
        "loss_history": np.asarray(loss_h),
        "entropy_mse_history": np.asarray(mse_h),
        "entropy_history": np.asarray(ent_h),
    }
