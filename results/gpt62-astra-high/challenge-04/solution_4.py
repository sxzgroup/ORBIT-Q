import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _probe(n, kind):
    circuit = tc.DMCircuit(n)
    # Ry(pi/2) prepares |+> from |0> in the active circuit precision.
    if kind == "ghz":
        circuit.ry(0, theta=np.pi / 2)
        for i in range(n - 1):
            circuit.cnot(i, i + 1)
    elif kind == "pairs":
        for i in range(0, n, 2):
            circuit.ry(i, theta=np.pi / 2)
            circuit.x(i + 1)
            circuit.cnot(i, i + 1)
    elif kind == "plus":
        for i in range(n):
            circuit.ry(i, theta=np.pi / 2)
    return circuit


def _kraus(probabilities):
    p01, p10 = probabilities
    return [
        jnp.diag(jnp.sqrt(1 - probabilities)),
        jnp.sqrt(p10) * jnp.array([[0, 1], [0, 0]]),
        jnp.sqrt(p01) * jnp.array([[0, 0], [1, 0]]),
    ]


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    n = config["n_qubits"]
    angle = config["entangler_angle"]
    masks = jnp.concatenate([jnp.eye(n, dtype=bool), jnp.ones((1, n), bool)])
    observables = jnp.where(
        masks[:, :, None, None], tc.gates.z().tensor, tc.gates.i().tensor
    )

    @jax.jit
    def table(probabilities):
        rows = []
        kraus = _kraus(probabilities)
        for kind in ("ghz", "pairs", "zero", "plus"):
            circuit = _probe(n, kind)
            for offset in (0, 1):
                for i in range(offset, n - 1, 2):
                    circuit.rxx(i, i + 1, theta=angle)
                    circuit.general_kraus(kraus, i)
                    circuit.general_kraus(kraus, i + 1)

            def measure(operators):
                ops = [(operators[i], [i]) for i in range(n)]
                return circuit.expectation(*ops, reuse=False).real

            rows.append(jax.vmap(measure)(observables))
        return jnp.stack(rows)

    target = table(jnp.array([config["[REDACTED]_p01"], config["[REDACTED]_p10"]]))
    initial = jnp.array([config["initial_p01"], config["initial_p10"]])
    logits = jnp.log(initial) - jnp.log1p(-initial)
    optimizer = optax.adam(config["learning_rate"])

    def objective(raw):
        loss = jnp.mean((table(jax.nn.sigmoid(raw)) - target) ** 2)
        return loss, loss

    def step(carry, _):
        raw, state = carry
        gradient, loss = jax.jacfwd(objective, has_aux=True)(raw)
        updates, state = optimizer.update(gradient, state)
        return (optax.apply_updates(raw, updates), state), loss

    (logits, _), history = jax.lax.scan(
        step, (logits, optimizer.init(logits)), None, length=config["max_steps"]
    )
    fitted = jax.nn.sigmoid(logits)
    return {
        "loss_history": np.asarray(history),
        "final_probabilities": np.asarray(fitted),
        "fitted_expectations": np.asarray(table(fitted)),
    }
