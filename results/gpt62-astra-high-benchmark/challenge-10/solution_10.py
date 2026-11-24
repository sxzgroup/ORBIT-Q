import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
from cotengra import RandomGreedyOptimizer
from tensornetwork.matrixproductstates.mpo import FiniteMPO


def _hamiltonian(n, zz, field):
    # Bond states are "no term yet", "pending Z", and "term complete".
    eye, x, z = (np.asarray(g().tensor) for g in
                 (tc.gates.i, tc.gates.x, tc.gates.z))
    w = np.zeros((3, 3, 2, 2), dtype=eye.dtype)
    w[0, 0] = w[2, 2] = eye
    w[0, 1], w[1, 2], w[0, 2] = z, -zz * z, -field * x
    sites = [w[:1]] + [w] * (n - 2) + [w[:, 2:]]
    return tc.quantum.tn2qop(FiniteMPO(sites, backend="jax"))


def _rotations(angles):
    local = tc.Circuit(1)
    local.rx(0, theta=angles[0])
    local.rz(0, theta=angles[1])
    local.ry(0, theta=angles[2])
    return local.matrix()


def run_solution(config):
    n, layers = config["n_qubits"], config["n_layers"]
    shape = (layers, n, 3)
    initial = np.random.default_rng(config["seed"]).normal(
        scale=config["initial_parameter_scale"], size=shape)
    adam = optax.adam(config["learning_rate"])
    path = RandomGreedyOptimizer(max_repeats=16, seed=0, parallel=False)
    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"), tc.runtime_contractor(
        "custom", optimizer=lambda i, o, s, **kw: path(i, o, s), use_primitives=True
    ):
        h = _hamiltonian(n, config["zz_strength"], config["x_strength"])

        def density(params):
            c = tc.Circuit(n)
            for q in config["initial_ones"]:
                c.x(q)
            blocks = jax.vmap(jax.vmap(_rotations))(params)
            for layer in range(layers):
                for q in range(n):
                    c.unitary(q, unitary=blocks[layer, q])
                c.cmz(*config["selected_qubits"])
            return tc.templates.measurements.mpo_expectation(c, h) / n

        def update(carry, _):
            params, state = carry
            energy, grad = jax.value_and_grad(density)(params)
            delta, state = adam.update(grad, state, params)
            return (optax.apply_updates(params, delta), state), energy

        params = jnp.asarray(initial)
        (params, _), history = jax.lax.scan(
            update, (params, adam.init(params)), None, length=config["max_steps"])
        return {"energy_history": np.asarray(history), "final_parameters": np.asarray(params)}
