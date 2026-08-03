"""
Task Suite Problem 11: spin-1 Haldane-chain VQE.

The TensorCircuit-NG baseline uses QuditCircuit and direct qudit unitary APIs for
all variational gates. Repeated layers are staged through scan to reduce JIT
tracing overhead.
"""

import numpy as np
import optax
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")

DIM = 3
SQRT2 = np.sqrt(2.0).astype(np.float32)

SX = K.convert_to_tensor(
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex64)
    / SQRT2
)
SY = K.convert_to_tensor(
    np.array([[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]], dtype=np.complex64)
    / SQRT2
)
SZ = K.convert_to_tensor(np.diag([1.0, 0.0, -1.0]).astype(np.complex64))
SZ2_DIAG = np.array([1.0, 0.0, 1.0], dtype=np.float32)
STRING_MIDDLE = K.convert_to_tensor(np.diag([-1.0, 1.0, -1.0]).astype(np.complex64))

DOT_BOND = K.kron(SX, SX) + K.kron(SY, SY) + K.kron(SZ, SZ)
DOT_BOND_SQUARED = DOT_BOND @ DOT_BOND
ZZ_BOND = K.kron(SZ, SZ)


def n_even_bonds(config):
    return config["n_sites"] // 2

def n_odd_bonds(config):
    return (config["n_sites"] - 1) // 2

def string_pairs(config):
    n_sites = config["n_sites"]
    return tuple((i, n_sites - 1 - i) for i in range(3))


def initial_parameters(config):
    rng = np.random.default_rng(config["seed"])
    scale = config["initial_parameter_scale"]
    n_layers = config["n_layers"]
    n_sites = config["n_sites"]
    return {
        "single_rz1": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_sites)).astype(np.float32)
        ),
        "single_ry": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_sites)).astype(np.float32)
        ),
        "single_rz2": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_sites)).astype(np.float32)
        ),
        "even_theta": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_even_bonds(config))).astype(np.float32)
        ),
        "even_phi": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_even_bonds(config))).astype(np.float32)
        ),
        "odd_theta": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_odd_bonds(config))).astype(np.float32)
        ),
        "odd_phi": K.convert_to_tensor(
            rng.normal(scale=scale, size=(n_layers, n_odd_bonds(config))).astype(np.float32)
        ),
    }


def initial_state(config):
    neel = np.zeros(DIM ** config["n_sites"], dtype=np.complex64)
    digits = [0 if i % 2 == 0 else 2 for i in range(config["n_sites"])]
    index = 0
    for digit in digits:
        index = index * DIM + digit
    neel[index] = 1.0
    return K.convert_to_tensor(neel)


def string_orders_from_state(state, config):
    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
    values = []
    for i, j in string_pairs(config):
        operators = [(tc.gates.Gate(SZ), [i])]
        for site in range(i + 1, j):
            operators.append((tc.gates.Gate(STRING_MIDDLE), [site]))
        operators.append((tc.gates.Gate(SZ), [j]))
        values.append(K.real(circuit.expectation(*operators)))
    return K.stack(values)


def basis_digit_table(n_sites):
    digits = np.zeros((DIM**n_sites, n_sites), dtype=np.int8)
    values = np.arange(DIM**n_sites, dtype=np.int64)
    for site in range(n_sites - 1, -1, -1):
        digits[:, site] = values % DIM
        values //= DIM
    return digits


def rz_batch(theta):
    theta = K.cast(theta, "complex64")
    zero, one = K.zeros_like(theta), K.ones_like(theta)
    return K.stack(
        [
            K.stack([K.exp(-1.0j * theta), zero, zero], axis=-1),
            K.stack([zero, one, zero], axis=-1),
            K.stack([zero, zero, K.exp(1.0j * theta)], axis=-1),
        ],
        axis=-2,
    )


def ry_batch(theta):
    c, s = K.cos(theta), K.sin(theta)
    return K.cast(
        K.stack(
            [
                K.stack([(1.0 + c) / 2.0, -s / SQRT2, (1.0 - c) / 2.0], axis=-1),
                K.stack([s / SQRT2, c, -s / SQRT2], axis=-1),
                K.stack([(1.0 - c) / 2.0, s / SQRT2, (1.0 + c) / 2.0], axis=-1),
            ],
            axis=-2,
        ),
        "complex64",
    )


def expm_pade33_fixed(a, s=5):
    eye = K.eye(a.shape[-1], dtype=a.dtype)
    a = a / (2**s)
    a2 = a @ a
    r = K.solve(12.0 * a2 + 120.0 * eye - a @ (a2 + 60.0 * eye), 12.0 * a2 + 120.0 * eye + a @ (a2 + 60.0 * eye))
    for _ in range(s):
        r = r @ r
    return r


def entangler_batch(theta, phi, beta):
    generator = (
        K.cast(theta, "complex64")[:, None, None] * DOT_BOND
        + K.cast(phi - theta, "complex64")[:, None, None] * ZZ_BOND
        + K.cast(beta, "complex64") * DOT_BOND_SQUARED
    )
    return expm_pade33_fixed(-1.0j * generator)


def apply_layer(state, layer_params, config):
    singles = K.einsum(
        "sab,sbc,scd->sad",
        rz_batch(layer_params["single_rz2"]),
        ry_batch(layer_params["single_ry"]),
        rz_batch(layer_params["single_rz1"]),
    )
    pair = K.reshape(
        K.einsum("kac,kbd->kabcd", singles[0::2], singles[1::2]),
        [-1, DIM * DIM, DIM * DIM],
    )
    even = entangler_batch(layer_params["even_theta"], layer_params["even_phi"], config["beta"]) @ pair
    odd = entangler_batch(layer_params["odd_theta"], layer_params["odd_phi"], config["beta"])

    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
    even_index = 0
    for left in range(0, config["n_sites"] - 1, 2):
        circuit.unitary(
            left, left + 1,
            unitary=tc.gates.Gate(K.reshape(even[even_index], (DIM, DIM, DIM, DIM))),
            name="spin1_even",
        )
        even_index += 1
    odd_index = 0
    for left in range(1, config["n_sites"] - 1, 2):
        circuit.unitary(
            left, left + 1,
            unitary=tc.gates.Gate(K.reshape(odd[odd_index], (DIM, DIM, DIM, DIM))),
            name="spin1_odd",
        )
        odd_index += 1
    return circuit.state()


def build_state(params, config):
    return K.scan(lambda s, p: apply_layer(s, p, config), params, initial_state(config))


def bond_hamiltonian(config):
    return DOT_BOND + config["beta"] * DOT_BOND_SQUARED


def energy_density_from_state(state, config, onsite_coeffs):
    circuit = tc.QuditCircuit(config["n_sites"], dim=DIM, inputs=state)
    bond_op = tc.gates.Gate(K.reshape(bond_hamiltonian(config), (DIM, DIM, DIM, DIM)))
    energy = K.cast(0.0, "complex64")
    for left in range(config["n_sites"] - 1):
        energy += circuit.expectation((bond_op, [left, left + 1]))
    return (K.real(energy) + K.sum(onsite_coeffs * K.abs(state) ** 2)) / config["n_sites"]


def run_solution(config):
    params = initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)
    digits = basis_digit_table(config["n_sites"])
    onsite_coeffs = K.convert_to_tensor(
        (config["single_ion_anisotropy"] * SZ2_DIAG[digits].sum(axis=1)).astype(np.float32)
    )

    def loss_fn(p):
        return energy_density_from_state(build_state(p, config), config, onsite_coeffs)

    def train_step(carry, _):
        p, state = carry
        value, grads = K.value_and_grad(loss_fn)(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return (p, state), value

    def train(p, state):
        return K.jaxy_scan(train_step, (p, state), K.zeros([config["max_steps"]]))

    def finalize(p):
        final_state = build_state(p, config)
        return energy_density_from_state(final_state, config, onsite_coeffs), string_orders_from_state(final_state, config)

    train = K.jit(train)
    finalize = K.jit(finalize)
    (params, _), history = train(params, opt_state)
    e, s = finalize(params)
    return {
        "energy_density_history": K.numpy(history),
        "final_energy_density": K.numpy(e),
        "final_string_orders": K.numpy(s),
    }
