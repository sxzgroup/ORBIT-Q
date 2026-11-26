"""Tensor-network sampling for the 7x7 mixed-axis shallow 2D circuit."""

from typing import Any, Dict

import numpy as np
import omeco
import tensorcircuit as tc


def _build_grid_circuit(config: Dict[str, Any]) -> tc.Circuit:
    """Build the 2D shallow circuit (Ry -> horizontal RZZ -> vertical RXX -> Rx)."""
    side = int(config["grid_side"])
    n_qubits = int(config["n_qubits"])
    circuit = tc.Circuit(n_qubits)

    for r in range(side):
        for c in range(side):
            alpha = (
                config["ry_offset"]
                + config["ry_row_sin_scale"] * np.sin(config["ry_row_sin_frequency"] * (r + 1))
                + config["ry_col_cos_scale"] * np.cos(config["ry_col_cos_frequency"] * (c + 1))
                + config["ry_diag_sin_scale"] * np.sin(config["ry_diag_sin_frequency"] * (r + c + 2))
            )
            circuit.ry(side * r + c, theta=alpha)

    k_h = 0
    for r in range(side):
        for c in range(side - 1):
            beta_h = (
                config["rzz_offset"]
                + config["rzz_edge_sin_scale"] * np.sin(config["rzz_edge_sin_frequency"] * (k_h + 1))
                + config["rzz_site_cos_scale"] * np.cos(config["rzz_site_cos_frequency"] * (2 * r + c + 1))
            )
            circuit.rzz(side * r + c, side * r + c + 1, theta=beta_h)
            k_h += 1

    k_v = 0
    for r in range(side - 1):
        for c in range(side):
            beta_v = (
                config["rxx_offset"]
                + config["rxx_edge_cos_scale"] * np.cos(config["rxx_edge_cos_frequency"] * (k_v + 1))
                + config["rxx_site_sin_scale"] * np.sin(config["rxx_site_sin_frequency"] * (r + 2 * c + 1))
            )
            circuit.rxx(side * r + c, side * (r + 1) + c, theta=beta_v)
            k_v += 1

    for r in range(side):
        for c in range(side):
            gamma = (
                config["rx_offset"]
                + config["rx_row_cos_scale"] * np.cos(config["rx_row_cos_frequency"] * (r + 1))
                - config["rx_col_sin_scale"] * np.sin(config["rx_col_sin_frequency"] * (c + 1))
                + config["rx_diag_cos_scale"] * np.cos(config["rx_diag_cos_frequency"] * (r + c + 2))
            )
            circuit.rx(side * r + c, theta=gamma)

    return circuit


def _sample_single_shot(circuit: tc.Circuit, n_qubits: int, status: Any) -> Any:
    """Draw one computational-basis sample via conditional tensor-network marginals."""
    sampled_bits = []
    ket_0 = tc.gates.array_to_tensor(np.array([1.0, 0.0]))
    ket_1 = tc.gates.array_to_tensor(np.array([0.0, 1.0]))

    for k in range(n_qubits):
        nodes1, edge1 = circuit._copy()
        nodes2, edge2 = circuit._copy(conj=True)
        newnodes = nodes1 + nodes2
        for i, edge in enumerate(edge1):
            if i != k:
                edge ^ edge2[i]
        for i in range(k):
            proj = (1 - sampled_bits[i]) * ket_0 + sampled_bits[i] * ket_1
            g1 = tc.gates.Gate(proj)
            circuit.coloring_nodes([g1], flag="measurement")
            newnodes.append(g1)
            g1.get_edge(0) ^ edge1[i]

            g2 = tc.gates.Gate(proj)
            circuit.coloring_nodes([g2], flag="measurement", is_dagger=True)
            newnodes.append(g2)
            g2.get_edge(0) ^ edge2[i]

        newnodes = tc.simplify._full_light_cone_cancel(newnodes)
        rho = tc.cons.contractor(newnodes, output_edge_order=[edge1[k], edge2[k]]).tensor
        p0 = tc.backend.relu(tc.backend.real(rho[0, 0]))
        p1 = tc.backend.relu(tc.backend.real(rho[1, 1]))
        pu = p0 / (p0 + p1)
        r = tc.backend.cast(status[k], tc.rdtypestr)
        bit = tc.backend.cast(r >= pu, tc.dtypestr)
        sampled_bits.append(bit)

    return tc.backend.cast(tc.backend.real(tc.backend.stack(sampled_bits)), "int32")


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Sample computational-basis bitstrings from the 2D tensor-network circuit."""
    n_qubits = int(config["n_qubits"])
    n_samples = int(config["n_samples"])
    opt = omeco.TreeSA(ntrials=2, niters=8)

    with (
        tc.runtime_backend("jax"),
        tc.runtime_dtype("complex64"),
        tc.runtime_contractor("custom", optimizer=opt, preprocessing=True),
    ):
        circuit = _build_grid_circuit(config)
        sample_batch = tc.backend.jit(
            tc.backend.vmap(lambda s: _sample_single_shot(circuit, n_qubits, s))
        )

        chunk_size = min(512, n_samples)
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        rng = np.random.default_rng()
        status = tc.backend.convert_to_tensor(
            rng.uniform(size=(n_chunks, chunk_size, n_qubits)), dtype=tc.rdtypestr
        )

        chunks = [np.asarray(sample_batch(status[i]), dtype=np.int32) for i in range(n_chunks)]
        samples = np.concatenate(chunks, axis=0)[:n_samples]

    return {"samples": np.asarray(samples, dtype=np.int32)}
