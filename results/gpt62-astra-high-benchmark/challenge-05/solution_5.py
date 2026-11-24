"""Variational Ising cooling in TensorCircuit's exact Gaussian representation."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _energy_function(n, field):
    # In the Hadamard basis |+> becomes vacuum, X becomes 1-2n_i,
    # and ZZ becomes nearest-neighbor XX (fermion hopping plus pairing).
    fgs = tc.FGSSimulator
    vacuum = fgs(n).get_alpha()
    # evol_ihamiltonian(h) applies exp(-H/2) to the many-body ket.
    onsite = sum(fgs.chemical_potential(4., i, n) for i in range(n))
    bonds = []
    for parity in range(2):
        generator = jnp.zeros_like(onsite)
        for i in range(parity, n - 1, 2):
            generator += fgs.hopping(-2., i, i + 1, n)
            generator += fgs.sc_pairing(-2., i, i + 1, n)
        bonds.append(generator)
    bonds = jnp.stack(bonds)

    def layer(alpha, inputs):
        (a, b), parity = inputs
        state = fgs(n, alpha=alpha)
        # Within each group all terms commute, so one exponential applies
        # exactly their product. Each call QR-normalizes the Gaussian state.
        state.evol_ihamiltonian(a * onsite)
        state.evol_ihamiltonian(b * bonds[parity])
        return state.get_alpha(), None

    def energy(params):
        parity = jnp.arange(len(params)) % 2
        alpha, _ = jax.lax.scan(layer, vacuum, (params, parity))
        corr = fgs(n, alpha=alpha).get_cmatrix_majorana()
        # TensorCircuit uses i(c-c†) for the second Majorana of each site.
        # Consecutive Majorana pairs alternate between physical X and ZZ.
        observables = -jnp.imag(jnp.diag(corr, k=1))
        return -(field * observables[::2].sum() + observables[1::2].sum()) / n

    return energy


def run_solution(config):
    with tc.runtime_backend("jax"), tc.runtime_dtype("complex128"):
        energy = _energy_function(config["n_qubits"], config["transverse_field"])
        adam = optax.adam(config["learning_rate"])
        params = jnp.full((config["n_layers"], 2), config["initial_filter_strength"],
                          dtype=jnp.float64)

        def update(carry, _):
            params, moments = carry
            loss, grad = jax.value_and_grad(energy)(params)
            delta, moments = adam.update(grad, moments)
            return (optax.apply_updates(params, delta), moments), loss

        (params, _), history = jax.lax.scan(
            update, (params, adam.init(params)), None, length=config["max_steps"]
        )
        a, b = np.asarray(params).T
        return {
            "final_a": a.reshape(-1, 2),
            "final_b": b.reshape(-1, 2),
            "energy_density_history": np.asarray(history),
        }
