"""Exact Gaussian simulation of the prescribed non-unitary Ising circuit."""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc


def _objective(n, layers, field):
    # Work in the Hadamard-rotated basis: |+> -> vacuum, X -> Z,
    # ZZ -> XX. Adjacent XX and on-site Z are quadratic fermion operators.
    fgs = tc.FGSSimulator
    vacuum = fgs(n).get_alpha()
    # evol_ihamiltonian(h) implements exp(-hat(h)/2), including normalization.
    # Z = 1 - 2 c^dagger c; its scalar identity part cancels on normalization.
    onsite = sum(fgs.chemical_potential(4.0, i, n) for i in range(n))
    bonds = []
    for parity in (0, 1):
        h = jnp.zeros((2 * n, 2 * n), dtype=onsite.dtype)
        for i in range(parity, n - 1, 2):
            h = h + fgs.hopping(-2.0, i, i + 1, n)
            h = h + fgs.sc_pairing(-2.0, i, i + 1, n)
        bonds.append(h)
    bonds = jnp.stack(bonds)

    def layer(alpha, data):
        strengths, parity = data
        state = fgs(n, alpha=alpha)
        state.evol_ihamiltonian(strengths[0] * onsite)
        state.evol_ihamiltonian(strengths[1] * bonds[parity])
        # Native QR normalization represents the full normalized many-body
        # state. The additional normalization between commuting gate groups
        # cancels exactly in the layer-end normalization. AD includes both.
        return state.get_alpha(), None

    def loss(parameters):
        alpha, _ = jax.lax.scan(
            layer, vacuum, (parameters, jnp.arange(layers) % 2)
        )
        state = fgs(n, alpha=alpha)
        e = 0.0
        for i in range(n):
            z = 1.0 - 2.0 * state.expectation_2body(i + n, i)
            e = e - field * z
        for i in range(n - 1):
            j = i + 1
            xx = (
                state.expectation_2body(i + n, j + n)
                + state.expectation_2body(i + n, j)
                - state.expectation_2body(i, j + n)
                - state.expectation_2body(i, j)
            )
            e = e - xx
        return jnp.real(e) / n

    return loss


def run_solution(config):
    jax.config.update("jax_enable_x64", True)
    tc.set_backend("jax")
    tc.set_dtype("complex128")
    n = int(config["n_qubits"])
    layers = int(config["n_layers"])
    steps = int(config["max_steps"])
    parameters = jnp.full(
        (layers, 2), config["initial_filter_strength"], dtype=jnp.float64
    )
    loss = _objective(n, layers, float(config["transverse_field"]))
    optimizer = optax.adam(float(config["learning_rate"]))

    def update(carry, _):
        parameters, optimizer_state = carry
        energy, gradient = jax.value_and_grad(loss)(parameters)
        updates, optimizer_state = optimizer.update(
            gradient, optimizer_state, parameters
        )
        parameters = optax.apply_updates(parameters, updates)
        return (parameters, optimizer_state), energy

    @jax.jit
    def train(parameters):
        return jax.lax.scan(
            update, (parameters, optimizer.init(parameters)), None, length=steps
        )

    (parameters, _), history = train(parameters)
    parameters = np.asarray(parameters)
    return {
        "final_a": parameters[:, 0].reshape(layers // 2, 2).copy(),
        "final_b": parameters[:, 1].reshape(layers // 2, 2).copy(),
        "energy_density_history": np.asarray(history),
    }
