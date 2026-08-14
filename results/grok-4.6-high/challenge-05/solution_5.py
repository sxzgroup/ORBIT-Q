"""Variational non-unitary filter cooling for the open TFIM."""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
from tensorcircuit.quantum import PauliStringSum2COO


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    tc.set_contractor("plain")

    n = int(config["n_qubits"])
    h_field = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    init_s = float(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    n_blocks = n_layers // 2
    backend = tc.backend
    x_mat = tc.gates._x_matrix
    zz_mat = tc.gates._zz_matrix

    paulis, weights = [], []
    for i in range(n - 1):
        term = [0] * n
        term[i] = 3
        term[i + 1] = 3
        paulis.append(term)
        weights.append(-1.0)
    for i in range(n):
        term = [0] * n
        term[i] = 1
        paulis.append(term)
        weights.append(-h_field)
    ham = PauliStringSum2COO(paulis, weights)

    init_c = tc.Circuit(n)
    for i in range(n):
        init_c.h(i)
    psi0 = init_c.state()

    def pair_filter(a, b):
        c = tc.Circuit(2)
        c.exp1(0, unitary=x_mat, theta=1j * a)
        c.exp1(1, unitary=x_mat, theta=1j * a)
        c.exp1(0, 1, unitary=zz_mat, theta=1j * b)
        return c.matrix()

    def single_filter(a):
        c = tc.Circuit(1)
        c.exp1(0, unitary=x_mat, theta=1j * a)
        return c.matrix()

    def energy_density(params):
        a, b = params
        psi = psi0
        for layer in range(n_layers):
            al = a[layer // 2, layer % 2]
            bl = b[layer // 2, layer % 2]
            c = tc.Circuit(n, inputs=psi)
            gate = pair_filter(al, bl)
            if layer % 2 == 0:
                for i in range(0, n, 2):
                    c.any(i, i + 1, unitary=gate)
            else:
                edge = single_filter(al)
                c.any(0, unitary=edge)
                for i in range(1, n - 1, 2):
                    c.any(i, i + 1, unitary=gate)
                c.any(n - 1, unitary=edge)
            psi = c.state()
            psi = psi / backend.norm(psi)
        final_c = tc.Circuit(n, inputs=psi)
        return tc.templates.measurements.sparse_expectation(final_c, ham) / n

    params = (
        jnp.full((n_blocks, 2), init_s, dtype=jnp.float32),
        jnp.full((n_blocks, 2), init_s, dtype=jnp.float32),
    )
    optimizer = optax.adam(lr)

    def train(init_params):
        opt_state = optimizer.init(init_params)

        def step(carry, _):
            current, opt_state = carry
            loss, grads = jax.value_and_grad(energy_density)(current)
            updates, opt_state = optimizer.update(grads, opt_state, current)
            current = optax.apply_updates(current, updates)
            return (current, opt_state), loss

        (final_params, _), history = jax.lax.scan(
            step, (init_params, opt_state), xs=None, length=max_steps
        )
        return final_params, history

    (final_a, final_b), history = jax.jit(train)(params)
    return {
        "final_a": np.asarray(final_a, dtype=np.float64),
        "final_b": np.asarray(final_b, dtype=np.float64),
        "energy_density_history": np.asarray(history, dtype=np.float64),
    }
