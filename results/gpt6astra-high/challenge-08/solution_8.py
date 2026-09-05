"""Exact sampling by a moving, two-row boundary of the square-grid network.

All vertical XX gates commute. After applying the gates between rows r and
r+1, measurement of row r commutes with every remaining gate. Contracting
and measuring that strip therefore leaves an exact conditional boundary on
row r+1. Every horizontal and vertical edge is retained, without an MPS or
truncation; the largest open strip has 14 qubits rather than 49.
"""
import numpy as np
import tensorcircuit as tc


def _angles(config):
    s = config['grid_side']
    r, c = np.indices((s, s))
    a = (config['ry_offset']
         + config['ry_row_sin_scale'] * np.sin(config['ry_row_sin_frequency'] * (r + 1))
         + config['ry_col_cos_scale'] * np.cos(config['ry_col_cos_frequency'] * (c + 1))
         + config['ry_diag_sin_scale'] * np.sin(config['ry_diag_sin_frequency'] * (r + c + 2)))
    g = (config['rx_offset']
         + config['rx_row_cos_scale'] * np.cos(config['rx_row_cos_frequency'] * (r + 1))
         - config['rx_col_sin_scale'] * np.sin(config['rx_col_sin_frequency'] * (c + 1))
         + config['rx_diag_cos_scale'] * np.cos(config['rx_diag_cos_frequency'] * (r + c + 2)))
    r, c = np.indices((s, s - 1))
    h = (config['rzz_offset']
         + config['rzz_edge_sin_scale'] * np.sin(config['rzz_edge_sin_frequency'] * (r * (s - 1) + c + 1))
         + config['rzz_site_cos_scale'] * np.cos(config['rzz_site_cos_frequency'] * (2 * r + c + 1)))
    r, c = np.indices((s - 1, s))
    v = (config['rxx_offset']
         + config['rxx_edge_cos_scale'] * np.cos(config['rxx_edge_cos_frequency'] * (r * s + c + 1))
         + config['rxx_site_sin_scale'] * np.sin(config['rxx_site_sin_frequency'] * (r + 2 * c + 1)))
    return a, g, h, v


def run_solution(config):
    import jax
    import jax.numpy as jnp
    tc.set_backend('jax')
    tc.set_dtype('complex64')
    tc.set_contractor('greedy', preprocessing=True)
    s, shots = int(config['grid_side']), int(config['n_samples'])
    if config['n_qubits'] != s * s:
        raise ValueError('n_qubits must equal grid_side squared')
    a, g, h, v = map(jnp.asarray, _angles(config))

    def prepare(alpha, beta):
        row = tc.Circuit(s)
        for col in range(s):
            row.ry(col, theta=alpha[col])
        for col in range(s - 1):
            row.rzz(col, col + 1, theta=beta[col])
        return row

    initial = prepare(a[0], h[0]).state()

    def step(frontier, data):
        alpha, beta, vertical, gamma, randoms = data
        # QuVector tensor products attach a fresh row to the contracted boundary.
        boundary = tc.quantum.QuVector.from_tensor(frontier.reshape([2] * s))
        row = prepare(alpha, beta).quvector()
        circuit = tc.Circuit(2 * s, mps_inputs=boundary | row)
        for col in range(s):
            circuit.rxx(col, s + col, theta=vertical[col])
            circuit.rx(col, theta=gamma[col])
        # Contract only this two-row strip, never the 49-qubit state.
        strip = tc.Circuit(2 * s, inputs=circuit.state())
        bits, _ = strip.measure_jit(*range(s), status=randoms)
        fixed = jnp.concatenate([bits, jnp.zeros(s)])
        frontier = strip.projected_subsystem(fixed, tuple(range(s, 2 * s)))
        return frontier, bits

    def sample(randoms):
        frontier, bits = jax.lax.scan(
            step, initial, (a[1:], h[1:], v, g[:-1], randoms[:-1]))
        last = tc.Circuit(s, inputs=frontier)
        for col in range(s):
            last.rx(col, theta=g[-1, col])
        tail, _ = last.measure_jit(*range(s), status=randoms[-1])
        return jnp.concatenate([bits.reshape(-1), tail]).astype(jnp.int8)

    batch = 64
    sample_batch = jax.jit(jax.vmap(sample))
    rng = np.random.default_rng(1808)
    result = np.empty((shots, s * s), dtype=np.int8)
    for start in range(0, shots, batch):
        randoms = rng.random((batch, s, s), dtype=np.float32)
        out = np.asarray(sample_batch(randoms))
        count = min(batch, shots - start)
        result[start:start + count] = out[:count]
    return {'samples': result}
