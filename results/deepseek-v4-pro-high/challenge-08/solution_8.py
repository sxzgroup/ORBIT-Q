import math

import numpy as np
import jax
import jax.numpy as jnp

import tensorcircuit as tc


def run_solution(config):
    s = int(config["grid_side"])
    n_samples = int(config["n_samples"])

    tc.set_backend("numpy")
    backend = tc.backend

    def alpha(r, c):
        return (
            config["ry_offset"]
            + config["ry_row_sin_scale"] * math.sin(config["ry_row_sin_frequency"] * (r + 1))
            + config["ry_col_cos_scale"] * math.cos(config["ry_col_cos_frequency"] * (c + 1))
            + config["ry_diag_sin_scale"] * math.sin(config["ry_diag_sin_frequency"] * (r + c + 2))
        )

    def gamma(r, c):
        return (
            config["rx_offset"]
            + config["rx_row_cos_scale"] * math.cos(config["rx_row_cos_frequency"] * (r + 1))
            - config["rx_col_sin_scale"] * math.sin(config["rx_col_sin_frequency"] * (c + 1))
            + config["rx_diag_cos_scale"] * math.cos(config["rx_diag_cos_frequency"] * (r + c + 2))
        )

    def beta_h(r, c, k):
        return (
            config["rzz_offset"]
            + config["rzz_edge_sin_scale"] * math.sin(config["rzz_edge_sin_frequency"] * (k + 1))
            + config["rzz_site_cos_scale"] * math.cos(config["rzz_site_cos_frequency"] * (2 * r + c + 1))
        )

    def beta_v(r, c, k):
        return (
            config["rxx_offset"]
            + config["rxx_edge_cos_scale"] * math.cos(config["rxx_edge_cos_frequency"] * (k + 1))
            + config["rxx_site_sin_scale"] * math.sin(config["rxx_site_sin_frequency"] * (r + 2 * c + 1))
        )

    def labels(n):
        return "a", "b", list("cdefg")[:n]

    def apply_one(gate, cur):
        old, new, rest = labels(len(cur.shape) - 1)
        return np.einsum(
            new + old + "," + old + "".join(rest) + "->" + new + "".join(rest),
            gate,
            cur,
        )

    def apply_left(gate, cur):
        nb = len(cur.shape) - 1
        old, new, rest = labels(nb)
        bond = chr(104 + nb)
        return np.einsum(
            new + old + bond + "," + old + "".join(rest) + "->" + new + "".join(rest) + bond,
            gate,
            cur,
        )

    def apply_right(gate, cur):
        nb = len(cur.shape) - 1
        old, new, rest = labels(nb)
        bond = chr(104 + nb)
        return np.einsum(
            bond + new + old + "," + old + "".join(rest) + "->" + new + "".join(rest) + bond,
            gate,
            cur,
        )

    def split_gate(gate, left_site):
        arr = np.asarray(gate.tensor)
        arr = np.transpose(arr, (0, 2, 1, 3)).reshape(4, 4)
        u, sv, vh, _ = backend.svd(arr)
        u = u[:, :2].reshape(2, 2, 2) * sv[:2][None, None, :]
        vh = vh[:2, :].reshape(2, 2, 2)
        return u if left_site else vh

    def full_site(cur, has_l, has_r, has_u, has_d):
        nb = len(cur.shape) - 1
        arr = cur.reshape([2] + [2] * nb)
        pos = 1
        for present in (has_l, has_r, has_u, has_d):
            if not present:
                arr = np.expand_dims(arr, pos)
            pos += 1
        return arr

    def build_site(r, c):
        cur = np.array([1, 0], dtype=np.complex64)
        cur = apply_one(np.asarray(tc.gates.ry(theta=alpha(r, c)).tensor), cur)

        if c > 0:
            k = r * (s - 1) + c - 1
            cur = apply_right(split_gate(tc.gates.rzz(theta=beta_h(r, c - 1, k)), False), cur)
        if c < s - 1:
            k = r * (s - 1) + c
            cur = apply_left(split_gate(tc.gates.rzz(theta=beta_h(r, c, k)), True), cur)
        if r > 0:
            k = (r - 1) * s + c
            cur = apply_right(split_gate(tc.gates.rxx(theta=beta_v(r - 1, c, k)), False), cur)
        if r < s - 1:
            k = r * s + c
            cur = apply_left(split_gate(tc.gates.rxx(theta=beta_v(r, c, k)), True), cur)

        cur = apply_one(np.asarray(tc.gates.rx(theta=gamma(r, c)).tensor), cur)
        return full_site(cur, c > 0, c < s - 1, r > 0, r < s - 1)

    def build_row(r):
        arrs = [build_site(r, c) for c in range(s)]
        ins, out_p, out_u, out_d = [], [], [], []
        for c in range(s):
            pl = chr(97 + c)
            ul = chr(104 + c)
            dl = chr(111 + c)
            left = chr(65 + c - 1) if c > 0 else "x"
            right = chr(65 + c) if c < s - 1 else "y"
            ins.append(pl + left + right + ul + dl)
            out_p.append(pl)
            out_u.append(ul)
            out_d.append(dl)
        sub = ",".join(ins) + "->" + "".join(out_p + out_u + out_d)
        return np.einsum(sub, *arrs)

    rows = [build_row(r) for r in range(s)]

    mps = []
    for r, row in enumerate(rows):
        shape_p = [2] * s
        shape_u = [2] * s if r > 0 else [1] * s
        shape_d = [2] * s if r < s - 1 else [1] * s
        row = row.reshape(shape_p + shape_u + shape_d)
        perm = list(range(s, 2 * s)) + list(range(s)) + list(range(2 * s, 3 * s))
        row = np.transpose(row, perm)
        din = 1 << s if r > 0 else 1
        dout = 1 << s if r < s - 1 else 1
        mps.append(row.reshape(din, 1 << s, dout))

    mps_j = [jnp.asarray(x) for x in mps]
    right_env = [None] * s
    right_env[s - 1] = jnp.ones((1,), dtype=jnp.complex64)
    for i in range(s - 2, -1, -1):
        right_env[i] = jnp.einsum("axc,c->a", mps_j[i + 1], right_env[i + 1])

    def sample_batch(key, tensors, envs, batch):
        left = jnp.ones((batch, 1), dtype=jnp.complex64)
        rows_out = []
        for i in range(len(tensors)):
            amp = jnp.einsum("ba,axc,c->bx", left, tensors[i], envs[i])
            prob = jnp.abs(amp) ** 2
            key, subkey = jax.random.split(key)
            x = jax.random.categorical(subkey, jnp.log(prob + 1e-30))
            selected = jnp.transpose(tensors[i], (1, 0, 2))[x]
            new_left = jnp.einsum("ba,bac->bc", left, selected)
            norm = jnp.linalg.norm(new_left, axis=1, keepdims=True)
            left = new_left / jnp.maximum(norm, 1e-30)
            rows_out.append(x)
        return jnp.stack(rows_out, axis=1)

    sampler = jax.jit(sample_batch, static_argnums=(3,))
    key = jax.random.PRNGKey(0)
    row_samples = np.asarray(sampler(key, tuple(mps_j), tuple(right_env), n_samples))

    cols = np.arange(s, dtype=np.int64)
    bits = (row_samples[:, :, None] >> cols[None, None, :]) & 1
    samples = bits.reshape(n_samples, s * s).astype(np.int32)
    return {"samples": samples}
