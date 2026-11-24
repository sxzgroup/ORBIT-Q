import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _energy_function(n_qubits, field, n_layers):
    # In the Hadamard basis the input is the fermionic vacuum. Both Z_i
    # and X_i X_{i+1} are i times products of adjacent TC Majoranas.
    vacuum = tc.FGSSimulator(n_qubits)
    w = vacuum.wtransform
    sites = np.arange(n_qubits)
    bonds = np.arange(n_qubits - 1)

    def generator(left):
        skew = np.zeros((2 * n_qubits, 2 * n_qubits))
        skew[left, left + 1] = 1
        skew[left + 1, left] = -1
        # TC imaginary time implements exp(-c† h c / 2) in Nambu notation.
        return -1j * (w.conj().T @ skew @ w)

    x_generator = generator(2 * sites)
    zz_generators = jnp.stack([generator(2 * bonds[p::2] + 1) for p in range(2)])
    brickwork = zz_generators[np.arange(n_layers) % 2]

    def layer(alpha, arguments):
        (a, b), zz_generator = arguments
        state = tc.FGSSimulator(n_qubits, alpha=alpha)
        # A complex angle lets TC apply the X filter before normalization.
        state.evol_hamiltonian(1j * a * x_generator)
        # TC applies the ZZ filter and restores an orthonormal frame with QR.
        # The frame represents the normalized many-body ket, up to global phase.
        state.evol_ihamiltonian(b * zz_generator)
        return state.get_alpha(), None

    def energy(parameters):
        alpha, _ = jax.lax.scan(layer, vacuum.get_alpha(), (parameters, brickwork))
        correlations = tc.FGSSimulator(n_qubits, alpha=alpha).get_cmatrix_majorana()
        # The Hamiltonian's minus sign turns -Im(<gamma_p gamma_q>) into +Im.
        x = correlations[2 * sites, 2 * sites + 1].imag.sum()
        zz = correlations[2 * bonds + 1, 2 * bonds + 2].imag.sum()
        return (field * x + zz) / n_qubits

    return energy


def run_solution(config):
    n_layers = config["n_layers"]
    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        energy = _energy_function(config["n_qubits"], config["transverse_field"], n_layers)
        loss_and_grad = jax.value_and_grad(energy)
        optimizer = optax.adam(config["learning_rate"])
        parameters = jnp.full((n_layers, 2), config["initial_filter_strength"])

        def update(carry, _):
            parameters, optimizer_state = carry
            loss, gradient = loss_and_grad(parameters)
            updates, optimizer_state = optimizer.update(gradient, optimizer_state)
            return (optax.apply_updates(parameters, updates), optimizer_state), loss

        (parameters, _), history = jax.lax.scan(
            update, (parameters, optimizer.init(parameters)), None, length=config["max_steps"]
        )
        filters = np.asarray(parameters)
        return {
            "final_a": filters[:, 0].reshape(-1, 2),
            "final_b": filters[:, 1].reshape(-1, 2),
            "energy_density_history": np.asarray(history),
        }
