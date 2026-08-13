import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc
from tensorcircuit.channels import _sqrt, kraus_to_super_gate


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")
    backend = tc.backend

    n_qubits = int(config["n_qubits"])
    theta = float(config["entangler_angle"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    g00 = tc.gates.Gate(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex64))
    g01 = tc.gates.Gate(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex64))
    g10 = tc.gates.Gate(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex64))
    g11 = tc.gates.Gate(np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex64))

    def bitflip_kraus(p01, p10):
        return [
            _sqrt(1.0 - p01) * g00 + _sqrt(1.0 - p10) * g11,
            _sqrt(p10) * g01,
            _sqrt(p01) * g10,
        ]

    def channel_super(p01, p10):
        return backend.reshape(kraus_to_super_gate(bitflip_kraus(p01, p10)), [4, 4])

    def rxx_super(angle):
        unitary = tc.gates.rxx_gate(theta=angle).tensor
        superop = backend.einsum("ijkl,abcd->iajbkcld", unitary, backend.conj(unitary))
        return backend.reshape(superop, [4, 4, 4, 4])

    def product_tensors(local_vec):
        tensor = np.asarray(local_vec, dtype=np.complex64).reshape(1, 4, 1)
        return [tensor.copy() for _ in range(n_qubits)]

    def ghz_tensors():
        left = np.zeros((1, 4, 4), dtype=np.complex64)
        middle = np.zeros((4, 4, 4), dtype=np.complex64)
        right = np.zeros((4, 4, 1), dtype=np.complex64)
        for state in range(4):
            left[0, state, state] = 0.5
            middle[state, state, state] = 1.0
            right[state, state, 0] = 1.0
        return [left] + [middle.copy() for _ in range(n_qubits - 2)] + [right]

    def bell_pair_tensors():
        prep = tc.Circuit(2)
        prep.x(1)
        prep.h(0)
        prep.cnot(0, 1)
        psi = np.asarray(prep.state(), dtype=np.complex64).reshape(2, 2)
        rho = np.einsum("ij,kl->ijkl", psi, np.conj(psi))
        matrix = np.transpose(rho, (0, 2, 1, 3)).reshape(4, 4)
        uu, ss, vh = np.linalg.svd(matrix, full_matrices=False)
        left = (uu * ss).reshape(1, 4, -1).astype(np.complex64)
        right = vh.reshape(-1, 4, 1).astype(np.complex64)
        tensors = []
        for _ in range(n_qubits // 2):
            tensors.extend([left.copy(), right.copy()])
        return tensors

    def observable_tensors(which):
        identity_vec = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex64)
        z_vec = np.array([1.0, 0.0, 0.0, -1.0], dtype=np.complex64)
        tensors = []
        for site in range(n_qubits):
            local = z_vec if (which == "all" or which == site) else identity_vec
            tensors.append(local.reshape(1, 4, 1))
        return tensors

    init_states = [
        ghz_tensors(),
        bell_pair_tensors(),
        product_tensors([1.0, 0.0, 0.0, 0.0]),
        product_tensors([0.5, 0.5, 0.5, 0.5]),
    ]
    init_states = [
        [backend.convert_to_tensor(tensor) for tensor in tensors]
        for tensors in init_states
    ]
    z_states = [
        [backend.convert_to_tensor(tensor) for tensor in observable_tensors(which)]
        for which in list(range(n_qubits)) + ["all"]
    ]

    def probe_expectations(p01, p10, tensors):
        circuit = tc.MPSCircuit(n_qubits, dim=4, tensors=tensors, center_position=0)
        noise = tc.gates.Gate(channel_super(p01, p10))
        entangler = tc.gates.Gate(rxx_super(theta))
        for left in range(0, n_qubits - 1, 2):
            circuit.apply_general_gate(entangler, left, left + 1)
            circuit.apply_general_gate(noise, left)
            circuit.apply_general_gate(noise, left + 1)
        for left in range(1, n_qubits - 1, 2):
            circuit.apply_general_gate(entangler, left, left + 1)
            circuit.apply_general_gate(noise, left)
            circuit.apply_general_gate(noise, left + 1)
        values = []
        for obs_tensors in z_states:
            observable = tc.MPSCircuit(
                n_qubits, dim=4, tensors=obs_tensors, center_position=0
            )
            values.append(backend.real(circuit.proj_with_mps(observable, conj=False)))
        return backend.stack(values)

    def observable_table(probs):
        p01, p10 = probs[0], probs[1]
        rows = [probe_expectations(p01, p10, tensors) for tensors in init_states]
        return backend.stack(rows)

    true_probs = jnp.asarray(
        [config["true_p01"], config["true_p10"]], dtype=jnp.float32
    )
    init_probs = jnp.asarray(
        [config["initial_p01"], config["initial_p10"]], dtype=jnp.float32
    )
    raw0 = jnp.log(init_probs / (1.0 - init_probs))
    target = observable_table(true_probs)

    def loss_fn(raw, target_table):
        fitted = observable_table(jax.nn.sigmoid(raw))
        residual = fitted - target_table
        return jnp.mean(jnp.square(residual))

    def train(raw, target_table):
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(raw)

        def step(carry, _):
            raw_params, state = carry
            loss, grads = jax.value_and_grad(loss_fn)(raw_params, target_table)
            updates, state = optimizer.update(grads, state)
            raw_params = optax.apply_updates(raw_params, updates)
            return (raw_params, state), loss

        (raw, _), losses = jax.lax.scan(
            step, (raw, opt_state), xs=None, length=max_steps
        )
        probs = jax.nn.sigmoid(raw)
        fitted = observable_table(probs)
        return losses, probs, fitted

    losses, probs, fitted = jax.jit(train)(raw0, target)
    return {
        "loss_history": np.asarray(losses, dtype=np.float64).reshape((max_steps,)),
        "final_probabilities": np.asarray(probs, dtype=np.float64).reshape((2,)),
        "fitted_expectations": np.asarray(fitted, dtype=np.float64).reshape(
            (4, n_qubits + 1)
        ),
    }
