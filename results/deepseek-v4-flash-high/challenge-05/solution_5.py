import numpy as np
import jax
import jax.numpy as jnp
import tensornetwork as tn
import tensorcircuit as tc


tc.set_backend("jax")
tc.set_dtype("complex64")

_I2 = np.eye(2, dtype=np.complex64)
_XM = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex64)
_ZM = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex64)


def _h_mpo_energy(state, h_field, n):
    """Contract <state|H|state> in one tensor-network pass (H as 3-state MPO)."""
    ket = tn.Node(tc.backend.reshape(state, [2] * n))
    bra = tn.Node(tc.backend.conj(tc.backend.reshape(state, [2] * n)))
    nodes = [ket, bra]
    prev_out = None
    first = None
    for i in range(n):
        if i == 0:
            t = np.zeros((1, 2, 2, 3), dtype=np.complex64)
            t[0, :, :, 0] = _I2
            t[0, :, :, 1] = _ZM
            t[0, :, :, 2] = -h_field * _XM
        elif i == n - 1:
            t = np.zeros((3, 2, 2, 1), dtype=np.complex64)
            t[0, :, :, 0] = -h_field * _XM
            t[1, :, :, 0] = -_ZM
            t[2, :, :, 0] = _I2
        else:
            t = np.zeros((3, 2, 2, 3), dtype=np.complex64)
            t[0, :, :, 0] = _I2
            t[0, :, :, 1] = _ZM
            t[0, :, :, 2] = -h_field * _XM
            t[1, :, :, 2] = -_ZM
            t[2, :, :, 2] = _I2
        node = tn.Node(t)
        if prev_out is not None:
            prev_out ^ node[0]
        node[1] ^ bra[i]
        node[2] ^ ket[i]
        nodes.append(node)
        prev_out = node[3]
        if first is None:
            first = node
    first[0] ^ prev_out
    return tc.contractor(nodes, output_edge_order=[]).tensor


def _energy(a, b, n_qubits, transverse_field, n_layers):
    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.h(i)
    for l in range(n_layers):
        al = a[l // 2, l % 2]
        bl = b[l // 2, l % 2]
        ga = jnp.array([[jnp.cosh(al), jnp.sinh(al)], [jnp.sinh(al), jnp.cosh(al)]])
        for i in range(n_qubits):
            c.apply_general_gate(tc.gates.any(ga), i)
        ebb = jnp.exp(bl)
        em = jnp.exp(-bl)
        gb = jnp.diag(jnp.array([ebb, em, em, ebb]))
        start = l % 2
        for i in range(start, n_qubits - 1, 2):
            c.apply_general_gate(tc.gates.any(gb), i, i + 1)
        s = c.state()
        norm = jnp.sqrt(jnp.sum(jnp.abs(s) ** 2))
        c = tc.Circuit(n_qubits, inputs=s / norm)
    return jnp.real(_h_mpo_energy(c.state(), transverse_field, n_qubits) / n_qubits)


def run_solution(config):
    n_qubits = config["n_qubits"]
    transverse_field = config["transverse_field"]
    n_layers = config["n_layers"]
    init = config["initial_filter_strength"]
    max_steps = config["max_steps"]
    lr = config["learning_rate"]

    loss_grad = jax.jit(
        jax.value_and_grad(
            lambda p: _energy(p[0], p[1], n_qubits, transverse_field, n_layers)
        )
    )
    nblocks = n_layers // 2

    def step(carry, t):
        a, b, ma, va, mb, vb, hist = carry
        loss, (ga, gb) = loss_grad((a, b))
        hist = hist.at[t].set(loss)
        tt = t + 1.0
        ma = 0.9 * ma + 0.1 * ga
        va = 0.999 * va + 0.001 * ga * ga
        mb = 0.9 * mb + 0.1 * gb
        vb = 0.999 * vb + 0.001 * gb * gb
        mah = ma / (1 - 0.9 ** tt)
        vah = va / (1 - 0.999 ** tt)
        mbh = mb / (1 - 0.9 ** tt)
        vbh = vb / (1 - 0.999 ** tt)
        a2 = a - lr * mah / (jnp.sqrt(vah) + 1e-8)
        b2 = b - lr * mbh / (jnp.sqrt(vbh) + 1e-8)
        return (a2, b2, ma, va, mb, vb, hist)

    step_jit = jax.jit(step)
    a0 = np.full((nblocks, 2), init, dtype=np.float32)
    b0 = np.full((nblocks, 2), init, dtype=np.float32)
    zeros = jnp.zeros((nblocks, 2))
    carry = (a0, b0, zeros, zeros, zeros, zeros, jnp.zeros(max_steps))
    for t in range(max_steps):
        carry = step_jit(carry, t)
    a_f, b_f, *_ = carry
    hist = np.asarray(carry[6])
    return {
        "final_a": np.asarray(a_f),
        "final_b": np.asarray(b_f),
        "energy_density_history": hist,
    }
