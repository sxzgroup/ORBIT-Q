import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _kraus(probabilities):
    p01, p10 = probabilities
    k0 = jnp.diag(jnp.sqrt(1.0 - probabilities))
    k1 = jnp.sqrt(p10) * jnp.array([[0.0, 1.0], [0.0, 0.0]])
    k2 = jnp.sqrt(p01) * jnp.array([[0.0, 0.0], [1.0, 0.0]])
    return k0, k1, k2


def _prepare_probe(circuit, probe, n):
    if probe == 0:
        circuit.ry(0, theta=np.pi / 2)
        for i in range(n - 1):
            circuit.cnot(i, i + 1)
    elif probe == 1:
        for i in range(0, n - 1, 2):
            circuit.ry(i, theta=np.pi / 2)
            circuit.cnot(i, i + 1)
            circuit.x(i + 1)
    elif probe == 3:
        for i in range(n):
            circuit.ry(i, theta=np.pi / 2)


def run_solution(config):
    n = config["n_qubits"]
    angle = config["entangler_angle"]
    bonds = [(i, i + 1) for offset in (0, 1) for i in range(offset, n - 1, 2)]

    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        identity = tc.gates.i().tensor
        z = tc.gates.z().tensor
        # Identity factors let all measurements share one contraction topology.
        masks = jnp.concatenate((jnp.eye(n, dtype=bool), jnp.ones((1, n), dtype=bool)))
        observables = jnp.where(masks[..., None, None], z, identity)

        @jax.jit
        def expectation_table(probabilities):
            kraus = _kraus(probabilities)
            rows = []
            for probe in range(4):
                circuit = tc.DMCircuit(n)
                _prepare_probe(circuit, probe, n)
                for left, right in bonds:
                    circuit.rxx(left, right, theta=angle)
                    circuit.apply_general_kraus(kraus, left)
                    circuit.apply_general_kraus(kraus, right)

                def measure(operators):
                    return circuit.expectation(
                        *((op, [i]) for i, op in enumerate(operators)), reuse=False
                    ).real

                rows.append(jax.vmap(measure)(observables))
            return jnp.stack(rows)

        target = expectation_table(jnp.array([config["[REDACTED]_p01"], config["[REDACTED]_p10"]]))
        initial = jnp.array([config["initial_p01"], config["initial_p10"]])
        logits = jnp.log(initial) - jnp.log1p(-initial)
        adam = optax.adam(config["learning_rate"])

        def loss(parameters, target_table):
            residual = expectation_table(jax.nn.sigmoid(parameters)) - target_table
            return jnp.mean(jnp.square(residual))

        loss_and_grad = jax.value_and_grad(loss)

        @jax.jit
        def train(parameters, target_table):
            def step(carry, _):
                parameters, state = carry
                value, gradient = loss_and_grad(parameters, target_table)
                updates, state = adam.update(gradient, state)
                return (optax.apply_updates(parameters, updates), state), value

            (parameters, _), history = jax.lax.scan(
                step, (parameters, adam.init(parameters)), None, length=config["max_steps"]
            )
            return jax.nn.sigmoid(parameters), history

        probabilities, history = train(logits, target)
        return {
            "loss_history": np.asarray(history),
            "final_probabilities": np.asarray(probabilities),
            "fitted_expectations": np.asarray(expectation_table(probabilities)),
        }
