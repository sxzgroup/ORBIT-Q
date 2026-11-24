import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc


def _angles(config, side):
    r, c = np.indices((side, side))
    sin, cos = np.sin, np.cos

    def term(name, trig, index):
        return config[name + "_scale"] * trig(config[name + "_frequency"] * index)

    ry = (config["ry_offset"] + term("ry_row_sin", sin, r + 1)
          + term("ry_col_cos", cos, c + 1) + term("ry_diag_sin", sin, r + c + 2))
    rx = (config["rx_offset"] + term("rx_row_cos", cos, r + 1)
          - term("rx_col_sin", sin, c + 1) + term("rx_diag_cos", cos, r + c + 2))
    zz = (config["rzz_offset"] + term("rzz_edge_sin", sin, r * (side - 1) + c + 1)
          + term("rzz_site_cos", cos, 2 * r + c + 1))
    xx = (config["rxx_offset"] + term("rxx_edge_cos", cos, r * side + c + 1)
          + term("rxx_site_sin", sin, r + 2 * c + 1))
    zz[:, -1] = 0
    xx[-1, :] = 0
    return ry.ravel(), zz.ravel(), xx.ravel(), rx.ravel()


@tc.set_function_backend("jax")
@tc.set_function_dtype("complex128")
def run_solution(config):
    side = config["grid_side"]
    n = side * side
    shots = config["n_samples"]
    ry, zz, xx, rx = _angles(config, side)

    # The boundary runs from the site being measured to the site below it,
    # inclusive, in row-major order. Zero padding flushes the final row.
    width = side + 1
    ry = np.pad(ry, (0, width))
    zz = np.pad(zz, (0, width))
    first = tc.Circuit(width)
    for i in range(width):
        first.ry(i, theta=ry[i])
    for i in range(side - 1):
        first.rzz(i, i + 1, theta=zz[i])
    initial = first.state()
    angles = jnp.asarray(np.column_stack((ry[width:], zz[side:side + n], xx, rx)))
    keep = tuple(range(1, width + 1))

    def measure_site(boundary, uniform, angles_i):
        alpha, horizontal, vertical, gamma = angles_i
        fresh = tc.Circuit(1)
        fresh.ry(0, theta=alpha)
        inputs = tc.Circuit(width, inputs=boundary).quvector() | fresh.quvector()
        patch = tc.Circuit(width + 1, mps_inputs=inputs)
        # Finish both horizontal interactions of the lower neighbor before
        # applying its vertical gate to the current site.
        patch.rzz(side, width, theta=horizontal)
        patch.rxx(0, side, theta=vertical)
        patch.rx(0, theta=gamma)
        measured = tc.Circuit(width + 1, inputs=patch.state())
        bit, _ = measured.measure(0, status=uniform[None])
        digits = jnp.broadcast_to(bit, (width + 1,))
        return measured.projected_subsystem(digits, keep), bit[0].astype(jnp.int8)

    batched_step = jax.vmap(measure_site, in_axes=(0, 0, None))

    @jax.jit
    def sample_batch(uniforms):
        def advance(boundary, data):
            params, u = data
            return batched_step(boundary, u, params)

        start = jnp.broadcast_to(initial, (uniforms.shape[1], initial.size))
        _, bits = jax.lax.scan(advance, start, (angles, uniforms))
        return bits.T

    rng = np.random.default_rng(8)
    samples = np.empty((shots, n), dtype=np.int8)
    for start in range(0, shots, 256):
        batch = samples[start:start + 256]
        batch[:] = np.asarray(sample_batch(rng.random((n, len(batch)))))
    return {"samples": samples}
