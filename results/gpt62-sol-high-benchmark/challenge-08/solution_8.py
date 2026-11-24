import jax
import numpy as np
import tensorcircuit as tc


def _angles(config, side):
    r, c = np.indices((side, side))

    def term(gate, name, function, coordinate):
        return config[f"{gate}_{name}_scale"] * function(
            config[f"{gate}_{name}_frequency"] * coordinate
        )

    alpha = (
        config["ry_offset"]
        + term("ry", "row_sin", np.sin, r + 1)
        + term("ry", "col_cos", np.cos, c + 1)
        + term("ry", "diag_sin", np.sin, r + c + 2)
    )
    gamma = (
        config["rx_offset"]
        + term("rx", "row_cos", np.cos, r + 1)
        - term("rx", "col_sin", np.sin, c + 1)
        + term("rx", "diag_cos", np.cos, r + c + 2)
    )
    hr, hc = r[:, :-1], c[:, :-1]
    horizontal = (
        config["rzz_offset"]
        + term("rzz", "edge_sin", np.sin, hr * (side - 1) + hc + 1)
        + term("rzz", "site_cos", np.cos, 2 * hr + hc + 1)
    )
    vr, vc = r[:-1], c[:-1]
    vertical = (
        config["rxx_offset"]
        + term("rxx", "edge_cos", np.cos, vr * side + vc + 1)
        + term("rxx", "site_sin", np.sin, vr + 2 * vc + 1)
    )
    return tuple(x.T.astype(np.float32) for x in (alpha, horizontal, vertical, gamma))


def _finish_column(circuit, vertical, rx, side):
    for row in range(side - 1):
        circuit.rxx(row, row + 1, theta=vertical[row])
    for row in range(side):
        circuit.rx(row, theta=rx[row])


def _sampler(backend, side):
    column_size = 1 << side
    measured = tuple(range(side))

    def advance(boundary, next_ry, crossing, vertical, rx, uniform):
        # Pass a single frontier tensor and a fresh product column as a QuVector.
        frontier = tc.QuVector.from_tensor(backend.reshape(boundary, (2,) * side))
        inputs = frontier.tensor_product(tc.Circuit(side).quvector())
        circuit = tc.Circuit(2 * side, mps_inputs=inputs)
        for row in measured:
            circuit.ry(side + row, theta=next_ry[row])
            circuit.rzz(row, side + row, theta=crossing[row])
        _finish_column(circuit, vertical, rx, side)

        # Future gates act only on unmeasured columns and cancel in this marginal.
        # The open grid strip contains only the current column and its neighbor.
        ket = circuit.state()
        joint = tc.quantum.measurement_counts(ket, counts=None)
        marginal = backend.sum(backend.reshape(joint, (column_size, column_size)), 1)
        outcome = tc.quantum.measurement_counts(
            marginal, counts=1, is_prob=True, status=uniform[None], format="sample_bin"
        )[0]
        boundary = tc.quantum.reduced_wavefunction(
            ket, measure=outcome, subsystems_to_trace_out=measured
        )
        return boundary / backend.norm(boundary), outcome

    def sample(uniforms, ry, horizontal, vertical, rx):
        first = tc.Circuit(side)
        for row in measured:
            first.ry(row, theta=ry[0, row])

        def step(boundary, args):
            return advance(boundary, *args)

        boundary, columns = jax.lax.scan(
            step, first.state(), (ry[1:], horizontal, vertical[:-1], rx[:-1], uniforms[:-1])
        )
        last = tc.Circuit(side, inputs=boundary)
        _finish_column(last, vertical[-1], rx[-1], side)
        last_column = last.sample(
            batch=1, allow_state=True, status=uniforms[-1:], format="sample_bin"
        )
        return backend.concat((columns, last_column), axis=0)

    return backend.jit(backend.vmap(sample))


def run_solution(config):
    side = config["grid_side"]
    n_samples = config["n_samples"]
    angles = _angles(config, side)
    rng = np.random.default_rng(8)
    uniforms = rng.random((n_samples, side), dtype=np.float32)
    samples = np.empty((n_samples, side, side), dtype=np.int8)

    with tc.runtime_backend("jax") as backend, tc.runtime_dtype("complex64"):
        sample_batch = _sampler(backend, side)
        batch_size = 64
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            columns = np.asarray(sample_batch(uniforms[start:stop], *angles))
            samples[start:stop] = columns.transpose(0, 2, 1)

    return {"samples": samples.reshape(n_samples, config["n_qubits"])}
