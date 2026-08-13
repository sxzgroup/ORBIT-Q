import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc


def _mps_tensors(mps):
    n = mps.L
    tensors = []
    for i in range(n):
        data = np.asarray(mps[i].data)
        inds = list(mps[i].inds)
        phys = mps.site_ind(i)
        left = mps.bond(i - 1, i) if i > 0 else None
        right = mps.bond(i, i + 1) if i < n - 1 else None
        extra = [ix for ix in inds if ix not in {phys, left, right}]
        if left is None:
            if extra:
                left = extra.pop(0)
            else:
                data = np.expand_dims(data, 0)
                inds = ["_L"] + inds
                left = "_L"
        if right is None:
            extra = [ix for ix in inds if ix not in {phys, left, right}]
            if extra:
                right = extra[0]
            else:
                data = np.expand_dims(data, -1)
                inds = inds + ["_R"]
                right = "_R"
        tensors.append(
            np.ascontiguousarray(
                np.transpose(data, (inds.index(left), inds.index(phys), inds.index(right))),
                dtype=np.complex64,
            )
        )
    return tuple(tensors)


def _neel_tensors(n_qubits):
    tensors = []
    for i in range(n_qubits):
        tensor = np.zeros((1, 2, 1), dtype=np.complex64)
        tensor[0, i % 2, 0] = 1.0
        tensors.append(tensor)
    return tuple(tensors)


def _gate_count(n_qubits, n_layers):
    return int(sum(len(range(layer % 2, n_qubits - 1, 2)) for layer in range(n_layers)))


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")

    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    scale = float(config["initial_parameter_scale"])
    seed = int(config["seed"])
    n_gates = _gate_count(n_qubits, n_layers)
    init_tensors = _neel_tensors(n_qubits)
    target_tensors = _mps_tensors(config["dmrg_state"])

    def loss_and_aux(params):
        angles = params.reshape((n_gates, 15))
        circuit = tc.MPSCircuit(n_qubits, tensors=list(init_tensors), center_position=0)
        gate = 0
        for layer in range(n_layers):
            for site in range(layer % 2, n_qubits - 1, 2):
                circuit.su4(site, site + 1, theta=angles[gate])
                gate += 1
        target = tc.MPSCircuit(n_qubits, tensors=list(target_tensors), center_position=0)
        overlap = circuit.proj_with_mps(target)
        fidelity = jnp.real(jnp.conj(overlap) * overlap)
        return jnp.real(1.0 - fidelity), (fidelity, overlap)

    def train(params):
        def step(carry, _):
            params, moment1, moment2, tstep = carry
            (loss, (fidelity, _overlap)), grad = jax.value_and_grad(
                loss_and_aux, has_aux=True
            )(params)
            tstep = tstep + 1.0
            moment1 = 0.9 * moment1 + 0.1 * grad
            moment2 = 0.999 * moment2 + 0.001 * (grad * grad)
            hat1 = moment1 / (1.0 - 0.9**tstep)
            hat2 = moment2 / (1.0 - 0.999**tstep)
            params = params - learning_rate * hat1 / (jnp.sqrt(hat2) + 1e-8)
            return (params, moment1, moment2, tstep), (loss, fidelity)

        carry = (
            params,
            jnp.zeros_like(params),
            jnp.zeros_like(params),
            jnp.float32(0.0),
        )
        (params, _moment1, _moment2, _tstep), (losses, fidelities) = jax.lax.scan(
            step, carry, xs=None, length=max_steps
        )
        _loss, (_fidelity, overlap) = loss_and_aux(params)
        return params, losses, fidelities, overlap

    rng = np.random.default_rng(seed)
    params0 = rng.normal(0.0, scale, size=(n_gates * 15,)).astype(np.float32)
    params, losses, fidelities, overlap = jax.jit(train)(jnp.asarray(params0))
    overlap = np.asarray(overlap)
    return {
        "loss_history": np.asarray(losses, dtype=np.float64),
        "fidelity_history": np.asarray(fidelities, dtype=np.float64),
        "final_parameters": np.asarray(params, dtype=np.float64),
        "final_overlap_phase": float(np.angle(overlap)),
    }
