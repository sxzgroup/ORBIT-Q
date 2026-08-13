import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
import tensornetwork as tn
from tensorcircuit.quantum import QuOperator
from tensorcircuit.templates.measurements import mpo_expectation


def _make_tfim_mpo(n_qubits, zz_strength, x_strength):
    i_mat = jnp.eye(2, dtype=jnp.complex64)
    x_mat = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex64)
    z_mat = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex64)
    hzz = -zz_strength
    hx = -x_strength
    bond_dim = 4

    tensors = []
    for _ in range(n_qubits):
        local = jnp.zeros((2, 2, bond_dim, bond_dim), dtype=jnp.complex64)
        local = local.at[..., 0, 0].set(i_mat)
        local = local.at[..., 3, 3].set(i_mat)
        local = local.at[..., 0, 1].set(hx * x_mat)
        local = local.at[..., 1, 3].set(i_mat)
        local = local.at[..., 0, 2].set(z_mat)
        local = local.at[..., 2, 3].set(hzz * z_mat)
        tensors.append(local)

    tensors[-1] = tensors[-1].at[..., 0, 3].add(hx * x_mat)

    nodes = [tn.Node(t) for t in tensors]
    out_edges = []
    in_edges = []
    for i, node in enumerate(nodes):
        out_edges.append(node[0])
        in_edges.append(node[1])
        if i < n_qubits - 1:
            node[3] ^ nodes[i + 1][2]

    left_vec = jnp.zeros((bond_dim,), dtype=jnp.complex64).at[0].set(1.0)
    right_vec = jnp.zeros((bond_dim,), dtype=jnp.complex64).at[3].set(1.0)
    left_node = tn.Node(left_vec)
    right_node = tn.Node(right_vec)
    left_node[0] ^ nodes[0][2]
    right_node[0] ^ nodes[-1][3]

    return QuOperator(out_edges, in_edges, ref_nodes=[left_node, right_node])


def run_solution(config):
    tc.set_backend("jax")
    tc.set_dtype("complex64")

    n_qubits = int(config["n_qubits"])
    n_layers = int(config["n_layers"])
    selected_qubits = [int(q) for q in config["selected_qubits"]]
    initial_ones = [int(q) for q in config["initial_ones"]]
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])
    initial_scale = float(config["initial_parameter_scale"])
    zz_strength = float(config["zz_strength"])
    x_strength = float(config["x_strength"])
    seed = int(config["seed"])

    hamiltonian_mpo = _make_tfim_mpo(n_qubits, zz_strength, x_strength)

    def energy_density(params):
        circuit = tc.Circuit(n_qubits)
        for q in initial_ones:
            circuit.x(q)

        for layer in range(n_layers):
            for q in range(n_qubits):
                circuit.rx(q, theta=params[layer, q, 0])
                circuit.rz(q, theta=params[layer, q, 1])
                circuit.ry(q, theta=params[layer, q, 2])
            circuit.cmz(*selected_qubits)

        energy = tc.backend.real(mpo_expectation(circuit, hamiltonian_mpo))
        return energy / n_qubits

    rng = np.random.default_rng(seed)
    params = rng.normal(
        0.0, initial_scale, size=(n_layers, n_qubits, 3)
    ).astype(np.float32)
    params = jnp.asarray(params)

    value_and_grad = jax.jit(jax.value_and_grad(energy_density))
    _, _ = value_and_grad(params)

    first_moment = jnp.zeros_like(params)
    second_moment = jnp.zeros_like(params)
    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8

    energy_history = np.empty((max_steps,), dtype=np.float32)

    for step in range(max_steps):
        value, grad = value_and_grad(params)
        value.block_until_ready()
        grad.block_until_ready()
        energy_history[step] = np.asarray(value, dtype=np.float32)

        first_moment = beta1 * first_moment + (1.0 - beta1) * grad
        second_moment = beta2 * second_moment + (1.0 - beta2) * (grad * grad)
        first_hat = first_moment / (1.0 - beta1 ** (step + 1))
        second_hat = second_moment / (1.0 - beta2 ** (step + 1))
        params = params - learning_rate * first_hat / (
            jnp.sqrt(second_hat) + adam_eps
        )

    return {
        "energy_history": energy_history,
        "final_parameters": np.asarray(params).reshape(n_layers, n_qubits, 3),
    }
