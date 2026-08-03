import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def _init_tensors(n_qubits, initial_ones):
    ones = set(initial_ones)
    tensors = []
    for q in range(n_qubits):
        t = np.zeros((1, 2, 1), dtype=np.complex64)
        t[0, 1 if q in ones else 0, 0] = 1.0
        tensors.append(t)
    return tensors[::-1]


def _cmz_mpo(selected_qubits):
    selected = sorted(selected_qubits)
    left, right = selected[0], selected[-1]
    selected_set = set(selected)
    eye = np.eye(2, dtype=np.complex64)
    tensors = []
    for q in range(left, right + 1):
        if q not in selected_set:
            w = np.zeros((2, 2, 2, 2), dtype=np.complex64)
            w[0, :, :, 0] = eye
            w[1, :, :, 1] = eye
        elif q == left:
            w = np.zeros((1, 2, 2, 2), dtype=np.complex64)
            w[0, :, :, 0] = eye
            w[0, 1, 1, 1] = -2.0
        elif q == right:
            w = np.zeros((2, 2, 2, 1), dtype=np.complex64)
            w[0, :, :, 0] = eye
            w[1, 1, 1, 0] = 1.0
        else:
            w = np.zeros((2, 2, 2, 2), dtype=np.complex64)
            w[0, :, :, 0] = eye
            w[1, 1, 1, 1] = 1.0
        tensors.append(w)
    return tensors, left


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")

    n = config["n_qubits"]
    n_layers = config["n_layers"]
    max_steps = config["max_steps"]
    lr = config["learning_rate"]
    zz = config["zz_strength"]
    xs = config["x_strength"]
    selected = config["selected_qubits"]

    init_tensors = _init_tensors(n, config["initial_ones"])
    cmz_tensors, cmz_left = _cmz_mpo(selected)

    rng = np.random.default_rng(config["seed"])
    params0 = rng.normal(
        0.0, config["initial_parameter_scale"], size=(n_layers, n, 3)
    ).astype(np.float32)

    def energy(params):
        c = tc.MPSCircuit(n, tensors=init_tensors, center_position=0)
        for layer in range(n_layers):
            for q in range(n):
                c.rx(q, theta=params[layer, q, 0])
                c.rz(q, theta=params[layer, q, 1])
                c.ry(q, theta=params[layer, q, 2])
            c.apply_MPO(cmz_tensors, cmz_left, center_left=True, split={})
        e = 0.0
        for i in range(n - 1):
            e += -zz * c.expectation(
                (tc.gates.z(), [i]), (tc.gates.z(), [i + 1])
            )
        for i in range(n):
            e += -xs * c.expectation((tc.gates.x(), [i]))
        return tc.backend.real(e)

    energy_grad = jax.value_and_grad(energy)

    def adam_scan(params0):
        def body(carry, step):
            params, m, v = carry
            value, grad = energy_grad(params)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            m_hat = m / (1.0 - 0.9 ** step)
            v_hat = v / (1.0 - 0.999 ** step)
            params = params - lr * m_hat / (jnp.sqrt(v_hat) + 1e-8)
            return (params, m, v), value / n

        return jax.lax.scan(
            body,
            (params0, jnp.zeros_like(params0), jnp.zeros_like(params0)),
            jnp.arange(1, max_steps + 1, dtype=jnp.float32),
        )

    (final_params, _, _), energy_history = jax.jit(adam_scan)(params0)

    return {
        "energy_history": np.asarray(energy_history),
        "final_parameters": np.asarray(final_params),
    }
