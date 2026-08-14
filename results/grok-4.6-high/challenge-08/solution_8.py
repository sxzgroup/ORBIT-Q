import numpy as np
import tensorcircuit as tc
import tensornetwork as tn


def _shared_edge(a, b):
    for edge in a.edges:
        if edge.node1 is b or edge.node2 is b:
            return edge
    raise RuntimeError("missing split bond")


def _build_circuit(cfg):
    side = int(cfg["grid_side"])
    n = side * side
    circuit = tc.Circuit(n, split={"max_singular_values": 2, "fixed_choice": 1})
    for r in range(side):
        for col in range(side):
            alpha = (
                cfg["ry_offset"]
                + cfg["ry_row_sin_scale"] * np.sin(cfg["ry_row_sin_frequency"] * (r + 1))
                + cfg["ry_col_cos_scale"] * np.cos(cfg["ry_col_cos_frequency"] * (col + 1))
                + cfg["ry_diag_sin_scale"] * np.sin(cfg["ry_diag_sin_frequency"] * (r + col + 2))
            )
            circuit.ry(r * side + col, theta=float(alpha))
    kh = 0
    for r in range(side):
        for col in range(side - 1):
            beta = (
                cfg["rzz_offset"]
                + cfg["rzz_edge_sin_scale"] * np.sin(cfg["rzz_edge_sin_frequency"] * (kh + 1))
                + cfg["rzz_site_cos_scale"] * np.cos(cfg["rzz_site_cos_frequency"] * (2 * r + col + 1))
            )
            circuit.rzz(r * side + col, r * side + col + 1, theta=float(beta))
            kh += 1
    kv = 0
    for r in range(side - 1):
        for col in range(side):
            beta = (
                cfg["rxx_offset"]
                + cfg["rxx_edge_cos_scale"] * np.cos(cfg["rxx_edge_cos_frequency"] * (kv + 1))
                + cfg["rxx_site_sin_scale"] * np.sin(cfg["rxx_site_sin_frequency"] * (r + 2 * col + 1))
            )
            circuit.rxx(r * side + col, (r + 1) * side + col, theta=float(beta))
            kv += 1
    for r in range(side):
        for col in range(side):
            gamma = (
                cfg["rx_offset"]
                + cfg["rx_row_cos_scale"] * np.cos(cfg["rx_row_cos_frequency"] * (r + 1))
                - cfg["rx_col_sin_scale"] * np.sin(cfg["rx_col_sin_frequency"] * (col + 1))
                + cfg["rx_diag_cos_scale"] * np.cos(cfg["rx_diag_cos_frequency"] * (r + col + 2))
            )
            circuit.rx(r * side + col, theta=float(gamma))
    return circuit, side


def _peps_sites(circuit, side):
    n = side * side
    nodes = circuit._nodes
    fronts = circuit._front
    n_h = side * (side - 1)
    n_v = (side - 1) * side
    rzz = nodes[2 * n : 2 * n + 2 * n_h]
    rxx = nodes[2 * n + 2 * n_h : 2 * n + 2 * n_h + 2 * n_v]
    rx0 = 2 * n + 2 * n_h + 2 * n_v
    site_nodes = [[] for _ in range(n)]
    for q in range(n):
        site_nodes[q].extend([nodes[q], nodes[n + q], nodes[rx0 + q]])
    left_e = [[None] * side for _ in range(side)]
    right_e = [[None] * side for _ in range(side)]
    up_e = [[None] * side for _ in range(side)]
    down_e = [[None] * side for _ in range(side)]
    kh = 0
    for r in range(side):
        for col in range(side - 1):
            n1, n2 = rzz[2 * kh], rzz[2 * kh + 1]
            bond = _shared_edge(n1, n2)
            site_nodes[r * side + col].append(n1)
            site_nodes[r * side + col + 1].append(n2)
            right_e[r][col] = bond
            left_e[r][col + 1] = bond
            kh += 1
    kv = 0
    for r in range(side - 1):
        for col in range(side):
            n1, n2 = rxx[2 * kv], rxx[2 * kv + 1]
            bond = _shared_edge(n1, n2)
            site_nodes[r * side + col].append(n1)
            site_nodes[(r + 1) * side + col].append(n2)
            down_e[r][col] = bond
            up_e[r + 1][col] = bond
            kv += 1
    sites = []
    for r in range(side):
        row = []
        for col in range(side):
            q = r * side + col
            local = site_nodes[q]
            out_edges = [fronts[q]]
            present = []
            for name, edge in (
                ("L", left_e[r][col]),
                ("R", right_e[r][col]),
                ("U", up_e[r][col]),
                ("D", down_e[r][col]),
            ):
                if edge is not None:
                    out_edges.append(edge)
                    present.append(name)
            copied, edges = tn.copy(local)
            tensor = np.asarray(
                tc.contractor(
                    [copied[node] for node in local],
                    output_edge_order=[edges[edge] for edge in out_edges],
                ).tensor
            )
            axis = {"p": 0}
            for i, name in enumerate(present):
                axis[name] = i + 1
            for name in ("L", "R", "U", "D"):
                if name not in axis:
                    tensor = np.expand_dims(tensor, axis=-1)
                    axis[name] = tensor.ndim - 1
            tensor = np.transpose(tensor, [axis[name] for name in ("p", "L", "R", "U", "D")])
            row.append(np.ascontiguousarray(tensor))
        sites.append(row)
    return sites


def _column_mps(sites):
    side = len(sites)
    cols = []
    for col in range(side):
        tensor = np.squeeze(sites[0][col], axis=3)
        for r in range(1, side):
            nxt = sites[r][col]
            tensor = np.tensordot(tensor, nxt, axes=([3], [3]))
            p, left, right, p2, left2, right2, down = tensor.shape
            tensor = np.transpose(tensor, (0, 3, 1, 4, 2, 5, 6)).reshape(
                p * p2, left * left2, right * right2, down
            )
        tensor = np.squeeze(tensor, axis=3)
        cols.append(np.ascontiguousarray(np.transpose(tensor, (1, 0, 2))))
    return cols


def _right_envs(mts):
    env = np.eye(mts[-1].shape[2], dtype=np.complex128)
    envs = [env]
    for tensor in mts[-1:0:-1]:
        tmp = np.tensordot(tensor, env, axes=([2], [0]))
        env = np.einsum("lpr,mpr->lm", tmp, np.conj(tensor), optimize=True)
        envs.append(env)
    return envs[::-1]


def _sample_mps(mts, n_samples, rng):
    envs = _right_envs(mts)
    n_cols = len(mts)
    dim = mts[0].shape[1]
    packed = np.empty((n_samples, n_cols), dtype=np.int64)
    left0 = np.ones((mts[0].shape[0],), dtype=np.complex128)
    for s in range(n_samples):
        left = left0
        for i, tensor in enumerate(mts):
            phys, right = tensor.shape[1], tensor.shape[2]
            work = left @ tensor.reshape(tensor.shape[0], phys * right)
            work = work.reshape(phys, right)
            probs = np.sum((work @ envs[i]) * np.conj(work), axis=1).real
            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total <= 0.0:
                probs = np.full(phys, 1.0 / phys)
            else:
                probs = probs / total
            choice = int(rng.choice(dim, p=probs))
            packed[s, i] = choice
            vec = work[choice]
            norm = np.linalg.norm(vec)
            left = vec if norm == 0.0 else vec / norm
    return packed


def _unpack_columns(packed, side):
    n_samples = packed.shape[0]
    samples = np.empty((n_samples, side * side), dtype=np.int64)
    shifts = np.arange(side - 1, -1, -1)
    for col in range(side):
        samples[:, col::side] = (packed[:, col, None] >> shifts) & 1
    return samples


def run_solution(config):
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    circuit, side = _build_circuit(config)
    sites = _peps_sites(circuit, side)
    mts = _column_mps(sites)
    rng = np.random.default_rng()
    packed = _sample_mps(mts, int(config["n_samples"]), rng)
    samples = _unpack_columns(packed, side)
    return {"samples": samples}
