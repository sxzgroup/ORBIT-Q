import numpy as np
import jax
import jax.numpy as jnp
import omeco
import tensorcircuit as tc
from tensorcircuit.simplify import _full_light_cone_cancel

_COMPILED_CACHE = {}
_DEFAULT_RNG = np.random.default_rng(0)


def _circuit_signature(config):
    keys = (
        "grid_side",
        "n_qubits",
        "ry_offset",
        "ry_row_sin_scale",
        "ry_row_sin_frequency",
        "ry_col_cos_scale",
        "ry_col_cos_frequency",
        "ry_diag_sin_scale",
        "ry_diag_sin_frequency",
        "rzz_offset",
        "rzz_edge_sin_scale",
        "rzz_edge_sin_frequency",
        "rzz_site_cos_scale",
        "rzz_site_cos_frequency",
        "rxx_offset",
        "rxx_edge_cos_scale",
        "rxx_edge_cos_frequency",
        "rxx_site_sin_scale",
        "rxx_site_sin_frequency",
        "rx_offset",
        "rx_row_cos_scale",
        "rx_row_cos_frequency",
        "rx_col_sin_scale",
        "rx_col_sin_frequency",
        "rx_diag_cos_scale",
        "rx_diag_cos_frequency",
    )
    return tuple((k, float(config[k]) if isinstance(config[k], (int, float, np.number)) else config[k]) for k in keys)


def _build_circuit(config):
    grid_side = int(config["grid_side"])
    n_qubits = int(config["n_qubits"])
    if n_qubits != grid_side * grid_side:
        raise ValueError("n_qubits must equal grid_side ** 2.")

    c = tc.Circuit(n_qubits)

    for r in range(grid_side):
        for col in range(grid_side):
            q = grid_side * r + col
            alpha = (
                float(config["ry_offset"])
                + float(config["ry_row_sin_scale"]) * np.sin(float(config["ry_row_sin_frequency"]) * (r + 1))
                + float(config["ry_col_cos_scale"]) * np.cos(float(config["ry_col_cos_frequency"]) * (col + 1))
                + float(config["ry_diag_sin_scale"]) * np.sin(float(config["ry_diag_sin_frequency"]) * (r + col + 2))
            )
            c.ry(q, theta=alpha)

    k_h = 0
    for r in range(grid_side):
        for col in range(grid_side - 1):
            q1 = grid_side * r + col
            q2 = grid_side * r + col + 1
            beta_h = (
                float(config["rzz_offset"])
                + float(config["rzz_edge_sin_scale"]) * np.sin(float(config["rzz_edge_sin_frequency"]) * (k_h + 1))
                + float(config["rzz_site_cos_scale"]) * np.cos(float(config["rzz_site_cos_frequency"]) * (2 * r + col + 1))
            )
            c.rzz(q1, q2, theta=beta_h)
            k_h += 1

    k_v = 0
    for r in range(grid_side - 1):
        for col in range(grid_side):
            q1 = grid_side * r + col
            q2 = grid_side * (r + 1) + col
            beta_v = (
                float(config["rxx_offset"])
                + float(config["rxx_edge_cos_scale"]) * np.cos(float(config["rxx_edge_cos_frequency"]) * (k_v + 1))
                + float(config["rxx_site_sin_scale"]) * np.sin(float(config["rxx_site_sin_frequency"]) * (r + 2 * col + 1))
            )
            c.rxx(q1, q2, theta=beta_v)
            k_v += 1

    for r in range(grid_side):
        for col in range(grid_side):
            q = grid_side * r + col
            gamma = (
                float(config["rx_offset"])
                + float(config["rx_row_cos_scale"]) * np.cos(float(config["rx_row_cos_frequency"]) * (r + 1))
                - float(config["rx_col_sin_scale"]) * np.sin(float(config["rx_col_sin_frequency"]) * (col + 1))
                + float(config["rx_diag_cos_scale"]) * np.cos(float(config["rx_diag_cos_frequency"]) * (r + col + 2))
            )
            c.rx(q, theta=gamma)

    return c


def _get_compiled_sampler(config, chunk_size):
    sig = (_circuit_signature(config), int(chunk_size))
    if sig in _COMPILED_CACHE:
        return _COMPILED_CACHE[sig]

    n_qubits = int(config["n_qubits"])
    old_contractor = tc.cons.contractor
    score = omeco.ScoreFunction(tc_weight=1.0, sc_weight=0.0, rw_weight=64.0, sc_target=20.0)
    opt = omeco.TreeSA(ntrials=2, niters=4, score=score)
    base_cf = tc.get_contractor("custom", optimizer=opt, preprocessing=True, use_primitives=True)

    def lightcone_contractor(nodes, **kws):
        return base_cf(_full_light_cone_cancel(nodes), **kws)

    tc.cons._set_global_contractor(lightcone_contractor)
    try:
        circuit = _build_circuit(config)
        wires = tuple(range(n_qubits))
        batch_fn = jax.jit(jax.vmap(lambda st: circuit.measure_jit(*wires, with_prob=False, status=st)[0]))
        dummy = jnp.full((chunk_size, n_qubits), 0.5, dtype=jnp.float64)
        batch_fn = batch_fn.lower(dummy).compile()
    finally:
        tc.cons._set_global_contractor(old_contractor)

    _COMPILED_CACHE[sig] = batch_fn
    return batch_fn


def run_solution(config):
    n_qubits = int(config["n_qubits"])
    n_samples = int(config["n_samples"])
    if n_samples <= 0:
        return {"samples": np.empty((0, n_qubits), dtype=np.int64)}

    chunk_size = min(n_samples, 1024)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    padded_samples = n_chunks * chunk_size

    seed = config.get("seed", None)
    rng = _DEFAULT_RNG if seed is None else np.random.default_rng(seed)
    status = rng.random((padded_samples, n_qubits), dtype=np.float64)

    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        sampler = _get_compiled_sampler(config, chunk_size)
        chunks = [sampler(status[i * chunk_size : (i + 1) * chunk_size]) for i in range(n_chunks)]
        raw_samples = np.asarray(jnp.concatenate(chunks, axis=0))[:n_samples]

    samples = np.rint(raw_samples).astype(np.int64)
    return {"samples": samples}
