"""Challenge 8: 7x7 mixed-axis grid tensor-network sampling (49 qubits).

The fixed shallow 2D circuit (position-dependent RY, horizontal RZZ layer,
vertical RXX layer, position-dependent RX) is built as a TensorCircuit-NG
circuit and sampled in the computational basis directly from the circuit
tensor network (``Circuit.sample`` with ``allow_state=False``), so neither
the 2^49 statevector nor a dense probability vector is ever materialized.
The lazy network contractions use the framework's omeco contraction-path
optimizer, which exploits the two-dimensional lightcone structure.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex64")
tc.set_contractor("omeco")

import jax


def _build_circuit(config):
    s = int(config["grid_side"])
    n = int(config["n_qubits"])
    c = tc.Circuit(n)
    for r in range(s):
        for co in range(s):
            alpha = (
                config["ry_offset"]
                + config["ry_row_sin_scale"] * np.sin(config["ry_row_sin_frequency"] * (r + 1))
                + config["ry_col_cos_scale"] * np.cos(config["ry_col_cos_frequency"] * (co + 1))
                + config["ry_diag_sin_scale"] * np.sin(config["ry_diag_sin_frequency"] * (r + co + 2))
            )
            c.ry(s * r + co, theta=alpha)
    kh = 0
    for r in range(s):
        for co in range(s - 1):
            beta_h = (
                config["rzz_offset"]
                + config["rzz_edge_sin_scale"] * np.sin(config["rzz_edge_sin_frequency"] * (kh + 1))
                + config["rzz_site_cos_scale"] * np.cos(config["rzz_site_cos_frequency"] * (2 * r + co + 1))
            )
            # rzz(theta) = exp(-i theta Z Z / 2), matching the task convention
            c.rzz(s * r + co, s * r + co + 1, theta=beta_h)
            kh += 1
    kv = 0
    for r in range(s - 1):
        for co in range(s):
            beta_v = (
                config["rxx_offset"]
                + config["rxx_edge_cos_scale"] * np.cos(config["rxx_edge_cos_frequency"] * (kv + 1))
                + config["rxx_site_sin_scale"] * np.sin(config["rxx_site_sin_frequency"] * (r + 2 * co + 1))
            )
            c.rxx(s * r + co, s * (r + 1) + co, theta=beta_v)
            kv += 1
    for r in range(s):
        for co in range(s):
            gamma = (
                config["rx_offset"]
                + config["rx_row_cos_scale"] * np.cos(config["rx_row_cos_frequency"] * (r + 1))
                - config["rx_col_sin_scale"] * np.sin(config["rx_col_sin_frequency"] * (co + 1))
                + config["rx_diag_cos_scale"] * np.cos(config["rx_diag_cos_frequency"] * (r + co + 2))
            )
            c.rx(s * r + co, theta=gamma)
    return c


def run_solution(config):
    n = int(config["n_qubits"])
    n_samples = int(config["n_samples"])
    c = _build_circuit(config)
    out = c.sample(
        batch=n_samples,
        allow_state=False,
        format="sample_bin",
        random_generator=jax.random.PRNGKey(42),
        jittable=True,
    )
    samples = np.asarray(out).reshape(n_samples, n).astype(np.int64)
    return {"samples": samples}
