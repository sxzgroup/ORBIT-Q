import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc
from tensorcircuit.templates.measurements import mpo_expectation


def _ising_mpo(n, zz, x):
    # MPO paths select one X term or a Z term followed immediately by a Z.
    local = np.zeros((3, 3, 2, 2), dtype=tc.npdtype)
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_z = np.diag([1, -1])
    local[0, 0] = local[2, 2] = np.eye(2)
    local[0, 2] = -x * pauli_x
    local[0, 1] = pauli_z
    local[1, 2] = -zz * pauli_z

    nodes = [tc.gates.Gate(local) for _ in range(n)]
    for left, right in zip(nodes, nodes[1:]):
        left[1] ^ right[0]
    begin = tc.gates.Gate(np.array([1, 0, 0], dtype=tc.npdtype))
    end = tc.gates.Gate(np.array([0, 0, 1], dtype=tc.npdtype))
    begin[0] ^ nodes[0][0]
    nodes[-1][1] ^ end[0]
    return tc.quantum.QuOperator([node[2] for node in nodes], [node[3] for node in nodes])


@tc.set_function_backend("jax")
@tc.set_function_dtype("complex64")
def run_solution(config):
    n = config["n_qubits"]
    layers = config["n_layers"]
    selected = tuple(config["selected_qubits"])
    hamiltonian = _ising_mpo(n, config["zz_strength"], config["x_strength"])
    parameters = np.random.default_rng(config["seed"]).normal(
        scale=config["initial_parameter_scale"], size=(layers, n, 3)
    ).astype(np.float32)
    adam = optax.adam(config["learning_rate"])

    def energy_density(angles):
        circuit = tc.Circuit(n)
        for qubit in config["initial_ones"]:
            circuit.x(qubit)
        for layer in range(layers):
            for qubit in range(n):
                rx, rz, ry = angles[layer, qubit]
                circuit.rx(qubit, theta=rx)
                circuit.rz(qubit, theta=rz)
                circuit.ry(qubit, theta=ry)
            circuit.cmz(*selected)
        # TensorCircuit applies the MPO to the circuit's QuVector before
        # contracting with the bra. Both vectors stay in tensor-network form.
        return mpo_expectation(circuit, hamiltonian) / n

    value_and_grad = jax.value_and_grad(energy_density)

    def step(carry, _):
        angles, optimizer_state = carry
        energy, gradient = value_and_grad(angles)
        updates, optimizer_state = adam.update(gradient, optimizer_state, angles)
        return (optax.apply_updates(angles, updates), optimizer_state), energy

    parameters = jnp.asarray(parameters)
    (parameters, _), history = jax.lax.scan(
        step, (parameters, adam.init(parameters)), None, length=config["max_steps"]
    )
    return {
        "energy_history": np.asarray(history),
        "final_parameters": np.asarray(parameters),
    }
