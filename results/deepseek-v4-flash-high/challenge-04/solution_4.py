import numpy as np

import tensorcircuit as tc
import jax
import jax.numpy as jnp

CD = jnp.complex128


def _superop(u):
    return jnp.kron(u, jnp.conj(u))


def _two_su(u):
    s = _superop(u)
    return s.reshape([2] * 8).transpose(0, 2, 1, 3, 4, 6, 5, 7).reshape(4, 4, 4, 4)


def _make_table(n, theta):
    even = [(i, i + 1) for i in range(0, n, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    h = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=CD) / jnp.sqrt(2)
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=CD)
    cnot = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=CD
    ).reshape(2, 2, 2, 2)
    xx = jnp.kron(x, x)
    u_rxx = jnp.cos(theta / 2) * jnp.eye(4, dtype=CD) - 1j * jnp.sin(theta / 2) * xx
    rxx = _two_su(u_rxx)
    iv = jnp.array([1.0, 0.0, 0.0, 1.0], dtype=CD)
    zv = jnp.array([1.0, 0.0, 0.0, -1.0], dtype=CD)

    def obs(i):
        w = [zv if j == i else iv for j in range(n)]
        return tc.MPSCircuit(
            n, dim=4, tensors=[x[None, :, None] for x in w], center_position=0
        )

    def noise_su(p01, p10):
        a = jnp.sqrt(1 - p01)
        b = jnp.sqrt(1 - p10)
        k0 = jnp.array([[a, 0], [0, b]], dtype=CD)
        k1 = jnp.array([[0, jnp.sqrt(p10)], [0, 0]], dtype=CD)
        k2 = jnp.array([[0, 0], [jnp.sqrt(p01), 0]], dtype=CD)
        return jnp.kron(k0, jnp.conj(k0)) + jnp.kron(k1, jnp.conj(k1)) + jnp.kron(
            k2, jnp.conj(k2)
        )

    def probe(kind, p01, p10):
        c = tc.MPSCircuit(n, dim=4)
        if kind == 0:
            c.apply_general_gate(tc.gates.Gate(_superop(h)), 0)
            for i in range(1, n):
                c.apply_general_gate(
                    tc.gates.Gate(_superop(cnot).reshape(4, 4, 4, 4)), i - 1, i
                )
        elif kind == 1:
            for i in range(0, n, 2):
                c.apply_general_gate(tc.gates.Gate(_superop(x)), i + 1)
                c.apply_general_gate(tc.gates.Gate(_superop(h)), i)
                c.apply_general_gate(
                    tc.gates.Gate(_superop(cnot).reshape(4, 4, 4, 4)), i, i + 1
                )
        elif kind == 3:
            for i in range(n):
                c.apply_general_gate(tc.gates.Gate(_superop(h)), i)
        e = noise_su(p01, p10)
        for i, j in even:
            c.apply_general_gate(tc.gates.Gate(rxx), i, j)
            c.apply_general_gate(tc.gates.Gate(e), i)
            c.apply_general_gate(tc.gates.Gate(e), j)
        for i, j in odd:
            c.apply_general_gate(tc.gates.Gate(rxx), i, j)
            c.apply_general_gate(tc.gates.Gate(e), i)
            c.apply_general_gate(tc.gates.Gate(e), j)
        vals = [jnp.real(c.proj_with_mps(obs(i))) for i in range(n)]
        parity = tc.MPSCircuit(
            n, dim=4, tensors=[zv[None, :, None] for _ in range(n)], center_position=0
        )
        vals.append(jnp.real(c.proj_with_mps(parity)))
        return jnp.stack(vals)

    def table(p01, p10):
        return jnp.stack([probe(k, p01, p10) for k in range(4)])

    return jax.jit(table)


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    n = config["n_qubits"]
    theta = config["entangler_angle"]
    table = _make_table(n, theta)
    true_p = (config["true_p01"], config["true_p10"])
    target = jnp.asarray(np.asarray(table(*true_p)))

    def loss(r):
        p = jax.nn.sigmoid(r)
        return jnp.mean((table(p[0], p[1]) - target) ** 2)

    r = jnp.array(
        [
            np.log(config["initial_p01"] / (1 - config["initial_p01"])),
            np.log(config["initial_p10"] / (1 - config["initial_p10"])),
        ],
        dtype=jnp.float64,
    )
    m = jnp.zeros(2, dtype=jnp.float64)
    v = jnp.zeros(2, dtype=jnp.float64)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = config["learning_rate"]
    loss_history = np.empty(config["max_steps"], dtype=np.float64)
    step = jax.jit(jax.value_and_grad(loss))
    for t in range(1, config["max_steps"] + 1):
        val, g = step(r)
        loss_history[t - 1] = np.asarray(val)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        r = r - lr * mhat / (jnp.sqrt(vhat) + eps)

    p = jax.nn.sigmoid(r)
    return {
        "loss_history": loss_history,
        "final_probabilities": np.asarray(p, dtype=np.float64),
        "fitted_expectations": np.asarray(table(p[0], p[1])),
    }
