import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax


def run_solution(config):
    tc.set_backend('jax')
    n = int(config['n_qubits'])
    blocks = int(config['n_layers']) // 2
    na = int(config['subsystem_size'])
    target = jnp.asarray(config['target_entropies'], dtype=jnp.float32)
    weight = float(config['entropy_weight'])
    sizes = [2 * n + 3 * len(range(p, n - 1, 2)) for p in (0, 1)]
    per_block = sum(sizes)
    params = jnp.asarray(np.random.default_rng(42).normal(0, 0.02, (blocks, per_block)), dtype=jnp.float32)
    strings, weights = [], []
    for i in range(n - 1):
        for pauli, coeff in [(1, 1.0), (2, 1.0), (3, config['zz_anisotropy'])]:
            term = [0] * n
            term[i] = term[i + 1] = pauli
            strings.append(term)
            weights.append(coeff)
    for i in range(n):
        term = [0] * n
        term[i] = 3
        strings.append(term)
        weights.append(config['staggered_field'] * (-1) ** i)
    hamiltonian = tc.quantum.PauliStringSum2MVP(strings, weights)
    initial_circuit = tc.Circuit(n)
    for i in range(1, n, 2):
        initial_circuit.x(i)
    initial_state = initial_circuit.state()

    def block(state, parameters):
        circuit = tc.Circuit(n, inputs=state)
        offset = 0
        for parity in (0, 1):
            for q in range(n):
                circuit.ry(q, theta=parameters[offset + 2 * q])
                circuit.rz(q, theta=parameters[offset + 2 * q + 1])
            offset += 2 * n
            for q in range(parity, n - 1, 2):
                circuit.rxx(q, q + 1, theta=2 * parameters[offset])
                circuit.ryy(q, q + 1, theta=2 * parameters[offset + 1])
                circuit.rzz(q, q + 1, theta=2 * parameters[offset + 2])
                offset += 3
        state = circuit.state()
        rho = tc.quantum.reduced_density_matrix(state, subsystem_to_keep=tuple(range(na)))
        entropy = tc.quantum.renyi_entropy(rho, k=2)
        return state, entropy

    def objective(parameters):
        state, entropies = jax.lax.scan(block, initial_state, parameters)
        energy = jnp.real(jnp.vdot(state, hamiltonian(state))) / n
        mse = jnp.mean((entropies - target) ** 2)
        return energy + weight * mse, (energy, mse, entropies)

    optimizer = optax.adam(float(config['learning_rate']))
    optimizer_state = optimizer.init(params)
    value_grad = jax.value_and_grad(objective, has_aux=True)

    @jax.jit
    def train(parameters, opt_state):
        def step(carry, _):
            parameters, opt_state = carry
            (loss, (energy, mse, entropies)), gradient = value_grad(parameters)
            updates, opt_state = optimizer.update(gradient, opt_state, parameters)
            parameters = optax.apply_updates(parameters, updates)
            return (parameters, opt_state), (energy, loss, mse, entropies)
        return jax.lax.scan(step, (parameters, opt_state), None, length=int(config['max_steps']))

    _, histories = train(params, optimizer_state)
    energy, loss, mse, entropies = jax.tree.map(np.asarray, histories)
    return {'energy_density_history': energy, 'loss_history': loss,
            'entropy_mse_history': mse, 'entropy_history': entropies}
