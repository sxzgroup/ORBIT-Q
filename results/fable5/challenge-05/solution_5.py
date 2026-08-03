"""Challenge 5: custom non-unitary gate cooling of an 18-qubit TFIM.

Dense statevector simulation with TensorCircuit-NG (JAX backend). Ten cooling
layers: the task-defined non-unitary one-qubit filter exp(a_l X) on every
qubit, then the two-qubit filter exp(b_l Z Z) on the layer's brickwork bonds,
applied through the framework circuit API; the state is rescaled to unit
norm after every layer and the rescaling is differentiated through. The
energy density <H>/n of H = -sum ZZ - h sum X is minimized with Adam for
exactly ``max_steps`` updates.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")

import jax
import jax.numpy as jnp
import optax
from tensorcircuit.templates.measurements import operator_expectation


def _tfim_hamiltonian(n, field):
    # H = -sum_i Z_i Z_{i+1} - field * sum_i X_i
    ls, ws = [], []
    for i in range(n - 1):
        s = [0] * n
        s[i] = 3
        s[i + 1] = 3
        ls.append(s)
        ws.append(-1.0)
    for i in range(n):
        s = [0] * n
        s[i] = 1
        ls.append(s)
        ws.append(-field)
    return tc.quantum.PauliStringSum2COO(ls, ws)


def _one_q_filter(a):
    # exp(a X) = cosh(a) I + sinh(a) X (non-unitary cooling filter)
    ch, sh = jnp.cosh(a), jnp.sinh(a)
    return jnp.array([[ch, sh], [sh, ch]], dtype=jnp.complex64)


def _two_q_filter(b):
    # exp(b Z Z) = diag(e^b, e^-b, e^-b, e^b)
    eb, emb = jnp.exp(b), jnp.exp(-b)
    return jnp.diag(jnp.array([eb, emb, emb, eb], dtype=jnp.complex64))


def run_solution(config):
    n = int(config["n_qubits"])
    field = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    init_strength = float(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    ham = _tfim_hamiltonian(n, field)
    even_bonds = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd_bonds = [(i, i + 1) for i in range(1, n - 1, 2)]

    def plus_state():
        c = tc.Circuit(n)
        for i in range(n):
            c.h(i)
        return c.state()

    psi0 = plus_state()

    def energy_density(params):
        state = psi0
        for layer in range(n_layers):
            a, b = params["a"][layer], params["b"][layer]
            f1 = _one_q_filter(a)
            # Layer map: exp(a X) on every qubit, then exp(b ZZ) on the
            # layer's bonds. On each bond the two filters act on the same
            # disjoint support, so they compose exactly into one two-qubit
            # operator exp(b ZZ) (exp(a X) x exp(a X)) applied through the
            # framework; qubits not covered by a bond keep their standalone
            # one-qubit filter.
            bonds = even_bonds if layer % 2 == 0 else odd_bonds
            covered = {q for bond in bonds for q in bond}
            fused = _two_q_filter(b) @ jnp.kron(f1, f1)
            c = tc.Circuit(n, inputs=state)
            for i in range(n):
                if i not in covered:
                    c.any(i, unitary=f1)
            for (i, j) in bonds:
                c.any(i, j, unitary=fused)
            state = c.state()
            # rescale to unit length after every cooling layer (differentiable)
            state = state / jnp.linalg.norm(state)
        c = tc.Circuit(n, inputs=state)
        return tc.backend.real(operator_expectation(c, ham)) / n

    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        e, grad = jax.value_and_grad(energy_density)(params)
        updates, opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), opt_state, e

    params = {
        "a": jnp.full((n_layers,), init_strength, dtype=jnp.float32),
        "b": jnp.full((n_layers,), init_strength, dtype=jnp.float32),
    }
    opt_state = opt.init(params)

    es = []
    for k in range(max_steps):
        params, opt_state, e = step(params, opt_state)
        es.append(e)
    e_hist = np.asarray(jax.device_get(jnp.stack(es)), dtype=np.float64)

    # (5, 2) layout: row = even+odd block, columns = (even, odd) sublayer
    final_a = np.asarray(params["a"], dtype=np.float64).reshape(n_layers // 2, 2)
    final_b = np.asarray(params["b"], dtype=np.float64).reshape(n_layers // 2, 2)
    return {
        "final_a": final_a,
        "final_b": final_b,
        "energy_density_history": e_hist,
    }
