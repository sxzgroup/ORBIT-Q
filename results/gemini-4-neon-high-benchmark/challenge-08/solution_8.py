"""7x7 mixed-axis grid tensor-network sampling with TensorCircuit-NG."""

from __future__ import annotations

from typing import Any, Dict, List

import cotengra as ctg
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc
import tensornetwork as tn
from tensorcircuit.simplify import _full_light_cone_cancel


def _build_grid_circuit(cfg: Dict[str, Any]) -> tc.Circuit:
    side = int(cfg["grid_side"])
    n_qubits = int(cfg["n_qubits"])
    if n_qubits != side * side:
        raise ValueError(f"Expected n_qubits == grid_side**2, got {n_qubits} vs {side**2}")

    circ = tc.Circuit(n_qubits)
    for r in range(side):
        for c in range(side):
            alpha = (
                cfg["ry_offset"]
                + cfg["ry_row_sin_scale"] * np.sin(cfg["ry_row_sin_frequency"] * (r + 1))
                + cfg["ry_col_cos_scale"] * np.cos(cfg["ry_col_cos_frequency"] * (c + 1))
                + cfg["ry_diag_sin_scale"] * np.sin(cfg["ry_diag_sin_frequency"] * (r + c + 2))
            )
            circ.ry(r * side + c, theta=float(alpha))

    k_h = 0
    for r in range(side):
        for c in range(side - 1):
            beta_h = (
                cfg["rzz_offset"]
                + cfg["rzz_edge_sin_scale"] * np.sin(cfg["rzz_edge_sin_frequency"] * (k_h + 1))
                + cfg["rzz_site_cos_scale"] * np.cos(cfg["rzz_site_cos_frequency"] * (2 * r + c + 1))
            )
            circ.rzz(r * side + c, r * side + c + 1, theta=float(beta_h))
            k_h += 1

    k_v = 0
    for r in range(side - 1):
        for c in range(side):
            beta_v = (
                cfg["rxx_offset"]
                + cfg["rxx_edge_cos_scale"] * np.cos(cfg["rxx_edge_cos_frequency"] * (k_v + 1))
                + cfg["rxx_site_sin_scale"] * np.sin(cfg["rxx_site_sin_frequency"] * (r + 2 * c + 1))
            )
            circ.rxx(r * side + c, (r + 1) * side + c, theta=float(beta_v))
            k_v += 1

    for r in range(side):
        for c in range(side):
            gamma = (
                cfg["rx_offset"]
                + cfg["rx_row_cos_scale"] * np.cos(cfg["rx_row_cos_frequency"] * (r + 1))
                - cfg["rx_col_sin_scale"] * np.sin(cfg["rx_col_sin_frequency"] * (c + 1))
                + cfg["rx_diag_cos_scale"] * np.cos(cfg["rx_diag_cos_frequency"] * (r + c + 2))
            )
            circ.rx(r * side + c, theta=float(gamma))

    return circ


def _merge_small_nodes(nodes: List[tn.Node]) -> List[tn.Node]:
    nodes = list(nodes)
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if isinstance(node, tn.CopyNode) or len(node.shape) > 2:
                continue
            for edge in node.edges:
                if edge.is_dangling():
                    continue
                other = edge.node2 if edge.node1 is node else edge.node1
                if other is node or isinstance(other, tn.CopyNode):
                    continue
                n1, n2 = edge.node1, edge.node2
                merged = tn.contract_between(n1, n2)
                nodes = [n for n in nodes if n is not n1 and n is not n2]
                nodes.append(merged)
                changed = True
                break
            if changed:
                break
    return nodes


def _row_distribution(
    circ: tc.Circuit, side: int, r_meas: int, prev_bits: Any, basis: Any
) -> Any:
    n_qubits = side * side
    if r_meas == side - 1:
        ket_nodes, ket_front = circ._copy()
        nodes, out_edges = list(ket_nodes), []
        for q in range(n_qubits):
            row, col = divmod(q, side)
            if row < r_meas:
                proj = tn.Node(basis[prev_bits[row, col]])
                ket_front[q] ^ proj[0]
                nodes.append(proj)
            else:
                out_edges.append(ket_front[q])
        amp = tc.contractor(_merge_small_nodes(nodes), output_edge_order=out_edges).tensor
        return jnp.real(amp * jnp.conj(amp)).reshape(-1)

    ket_nodes, ket_front = circ._copy()
    bra_nodes, bra_front = circ._copy(conj=True)
    nodes, out_edges = list(ket_nodes) + list(bra_nodes), []
    for q in range(n_qubits):
        row, col = divmod(q, side)
        if row < r_meas:
            vec = basis[prev_bits[row, col]]
            pk, pb = tn.Node(vec), tn.Node(vec)
            circ.coloring_nodes([pk], flag="measurement")
            circ.coloring_nodes([pb], flag="measurement", is_dagger=True)
            ket_front[q] ^ pk[0]
            bra_front[q] ^ pb[0]
            nodes.extend([pk, pb])
        elif row == r_meas:
            cn = tn.CopyNode(3, 2)
            circ.coloring_nodes([cn], flag="measurement")
            ket_front[q] ^ cn[0]
            bra_front[q] ^ cn[1]
            nodes.append(cn)
            out_edges.append(cn[2])
        else:
            ket_front[q] ^ bra_front[q]

    nodes = _merge_small_nodes(_full_light_cone_cancel(nodes))
    return jnp.real(tc.contractor(nodes, output_edge_order=out_edges).tensor).reshape(-1)


def _sample_from_probs(probs: Any, u: Any, shifts: Any, side: int) -> Any:
    cdf = jnp.cumsum(jnp.clip(probs, 0.0, None), axis=-1)
    cdf = cdf / cdf[..., -1:]
    idx = jnp.clip(jnp.sum(cdf < u[..., None], axis=-1).astype(jnp.int32), 0, (1 << side) - 1)
    return (idx[..., None] >> shifts) & 1


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    side = int(config["grid_side"])
    n_samples = int(config["n_samples"])
    seed = int(config.get("seed", 0))
    greedy_opt = ctg.GreedyOptimizer()
    optimizer = lambda inputs, output, size_dict, **_kws: greedy_opt(inputs, output, size_dict)

    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"), tc.runtime_contractor(
        "custom", optimizer=optimizer, preprocessing=True
    ):
        circ = _build_grid_circuit(config)
        basis = jnp.eye(2, dtype=jnp.complex128)
        shifts = jnp.arange(side - 1, -1, -1, dtype=jnp.int32)
        u_all = jnp.asarray(np.random.default_rng(seed).random((n_samples, side)), dtype=jnp.float64)

        p0 = _row_distribution(circ, side, 0, jnp.zeros((0, side), dtype=jnp.int32), basis)
        r0_bits = _sample_from_probs(p0[None, :], u_all[:, 0], shifts, side)
        if side == 1:
            return {"samples": np.asarray(r0_bits, dtype=np.int64)}

        def sample_remaining(b0: Any, u_row: Any) -> Any:
            rows = [b0]
            for r_meas in range(1, side):
                probs = _row_distribution(circ, side, r_meas, jnp.stack(rows, axis=0), basis)
                rows.append(_sample_from_probs(probs, u_row[r_meas], shifts, side))
            return jnp.concatenate(rows, axis=0)

        sample_batch = tc.backend.jit(
            tc.backend.vmap(sample_remaining, vectorized_argnums=(0, 1))
        )
        batch_size = min(1024, n_samples)
        chunks: List[Any] = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            b0_c, u_c = r0_bits[start:end], u_all[start:end]
            pad = batch_size - (end - start)
            if pad > 0:
                b0_c = jnp.pad(b0_c, ((0, pad), (0, 0)))
                u_c = jnp.pad(u_c, ((0, pad), (0, 0)))
                chunks.append(sample_batch(b0_c, u_c)[: end - start])
            else:
                chunks.append(sample_batch(b0_c, u_c))

        samples = np.asarray(jnp.concatenate(chunks, axis=0), dtype=np.int64)

    return {"samples": samples}
