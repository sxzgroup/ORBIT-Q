import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
import optax


def run_solution(config):
    tc.set_backend('jax')
    n = int(config['n_qubits'])
    blocks = int(config['n_layers']) // 2
    keep = int(config['subsystem_size'])
    even = list(range(0, n - 1, 2))
    odd = list(range(1, n - 1, 2))
    per_block = 4 * n + 3 * (len(even) + len(odd))
    strings, weights = [], []
    for i in range(n - 1):
        for p, w in ((1, 1.0), (2, 1.0), (3, config['zz_anisotropy'])):
            s = [0] * n
            s[i] = s[i + 1] = p
            strings.append(s)
            weights.append(w)
    for i in range(n):
        s = [0] * n
        s[i] = 3
        strings.append(s)
        weights.append(config['staggered_field'] * (-1) ** i)
    hamiltonian = tc.quantum.PauliStringSum2MVP(strings, weights)
    initial = tc.Circuit(n)
    for i in range(1, n, 2):
        initial.x(i)
    initial_state = initial.state()
    targets = jnp.asarray(config['target_entropies'])

    def circuit_block(state, params):
        c = tc.Circuit(n, inputs=state)
        offset = 0
        for bonds in (even, odd):
            for q in range(n):
                c.ry(q, theta=params[offset + 2 * q])
                c.rz(q, theta=params[offset + 2 * q + 1])
            offset += 2 * n
            for i in bonds:
                c.rxx(i, i + 1, theta=2 * params[offset])
                c.ryy(i, i + 1, theta=2 * params[offset + 1])
                c.rzz(i, i + 1, theta=2 * params[offset + 2])
                offset += 3
        state = c.state()
        rho = tc.quantum.reduced_density_matrix(state, subsystem_to_keep=list(range(keep)))
        purity = jnp.real(jnp.sum(rho * jnp.conj(rho)))
        return state, -jnp.log(purity)

    def objective(params):
        state, entropy = jax.lax.scan(circuit_block, initial_state, params)
        energy = jnp.real(jnp.vdot(state, hamiltonian(state))) / n
        mse = jnp.mean((entropy - targets) ** 2)
        return energy + config['entropy_weight'] * mse, (energy, mse, entropy)

    optimizer = optax.adam(config['learning_rate'])
    params = jax.random.normal(jax.random.PRNGKey(42), (blocks, per_block)) * 0.02
    opt_state = optimizer.init(params)
    value_grad = jax.value_and_grad(objective, has_aux=True)

    def step(carry, unused):
        params, opt_state = carry
        (loss, (energy, mse, entropy)), grad = value_grad(params)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (energy, loss, mse, entropy)

    @jax.jit
    def train(params, opt_state):
        return jax.lax.scan(step, (params, opt_state), None, length=int(config['max_steps']))[1]

    energy, loss, mse, entropy = train(params, opt_state)
    return {'energy_density_history': np.asarray(energy), 'loss_history': np.asarray(loss),
            'entropy_mse_history': np.asarray(mse), 'entropy_history': np.asarray(entropy)}
