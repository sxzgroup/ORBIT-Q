import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import networkx as nx
import tensorcircuit as tc
import jax.numpy as jnp
from tensorcircuit import quantum as qu


def run_solution(config):
    n = int(config["n_qubits"])
    field = float(config["transverse_field"])
    n_layers = int(config["n_layers"])
    init = float(config["initial_filter_strength"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])

    tc.set_backend("jax")
    tc.set_dtype("complex128")

    graph = nx.path_graph(n)
    hamiltonian = qu.heisenberg_hamiltonian(
        graph,
        hzz=-1.0,
        hx=-field,
        hxx=0.0,
        hyy=0.0,
        hy=0.0,
        hz=0.0,
        sparse=True,
        numpy=False,
    )

    eye1 = jnp.eye(2, dtype=jnp.complex128)
    xmat = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    eye2 = jnp.eye(4, dtype=jnp.complex128)
    zzmat = jnp.diag(jnp.array([1.0, -1.0, -1.0, 1.0], dtype=jnp.complex128))

    def one_qubit_filter(a):
        return jnp.cosh(a) * eye1 + jnp.sinh(a) * xmat

    def two_qubit_filter(b):
        matrix = jnp.cosh(b) * eye2 + jnp.sinh(b) * zzmat
        return tc.backend.reshape(matrix, [2, 2, 2, 2])

    def build_circuit(params):
        circuit = tc.Circuit(n)
        for i in range(n):
            circuit.H(i)
        for layer in range(n_layers):
            a = params[2 * layer]
            b = params[2 * layer + 1]
            for i in range(n):
                circuit.apply_general_gate(
                    tc.Gate(one_qubit_filter(a)), i
                )
            if layer % 2 == 0:
                bonds = range(0, n - 1, 2)
            else:
                bonds = range(1, n - 1, 2)
            for i in bonds:
                circuit.apply_general_gate(
                    tc.Gate(two_qubit_filter(b)), i, i + 1
                )
        return circuit

    def energy_density(params, hamiltonian):
        state = build_circuit(params).state()
        norm2 = jnp.real(jnp.vdot(state, state))
        projected = tc.backend.sparse_dense_matmul(hamiltonian, state)
        value = jnp.real(jnp.vdot(state, projected))
        return value / norm2 / n

    value_and_grad = jax.jit(
        jax.value_and_grad(energy_density, argnums=0)
    )

    params = jnp.full((2 * n_layers,), init, dtype=jnp.float64)
    first_moment = jnp.zeros_like(params)
    second_moment = jnp.zeros_like(params)
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    history = []

    for step in range(max_steps):
        value, gradient = value_and_grad(params, hamiltonian)
        history.append(float(value))

        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        first_hat = first_moment / (1.0 - beta1 ** (step + 1))
        second_hat = second_moment / (1.0 - beta2 ** (step + 1))
        params = params - lr * first_hat / (jnp.sqrt(second_hat) + epsilon)

    final_a = np.asarray(params[0::2]).reshape(n_layers // 2, 2)
    final_b = np.asarray(params[1::2]).reshape(n_layers // 2, 2)

    return {
        "final_a": final_a,
        "final_b": final_b,
        "energy_density_history": np.asarray(history),
    }
