"""Digital-analog hybrid VQE with trainable analog blocks in TensorCircuit-NG."""

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc


def _build_hamiltonian_mvps(
    n_qubits: int,
) -> Tuple[Any, Any, Any]:
    """Construct TensorCircuit Pauli-string MVPs for analog and target Hamiltonians."""
    ls_xy: List[List[int]] = []
    ls_zz: List[List[int]] = []
    ls_z: List[List[int]] = []
    w_z: List[float] = []

    for i in range(n_qubits - 1):
        for pauli in (1, 2):
            term = [0] * n_qubits
            term[i] = pauli
            term[i + 1] = pauli
            ls_xy.append(term)
        zz_term = [0] * n_qubits
        zz_term[i] = 3
        zz_term[i + 1] = 3
        ls_zz.append(zz_term)

    for i in range(n_qubits):
        z_term = [0] * n_qubits
        z_term[i] = 3
        ls_z.append(z_term)
        w_z.append(float((-1) ** i))

    mvp_xy = tc.quantum.PauliStringSum2MVP(ls_xy, [1.0] * len(ls_xy))
    mvp_z = tc.quantum.PauliStringSum2MVP(ls_z, w_z)
    mvp_target = tc.quantum.PauliStringSum2MVP(
        ls_xy + ls_zz + ls_z,
        [0.7] * len(ls_xy) + [1.1] * len(ls_zz) + [0.25 * w for w in w_z],
    )
    return mvp_xy, mvp_z, mvp_target


def _neel_state(n_qubits: int) -> Any:
    """Prepare the |0101...> Neel state using a TensorCircuit Circuit."""
    circuit = tc.Circuit(n_qubits)
    for i in range(1, n_qubits, 2):
        circuit.x(i)
    return circuit.state()


def run_solution(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Optimize the hybrid digital-analog VQE ansatz for the given config."""
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = int(config["n_qubits"])
    n_blocks = int(config["n_blocks"])
    t_min = float(config["t_min"])
    t_max = float(config["t_max"])
    ode_rtol = float(config["ode_rtol"])
    ode_atol = float(config["ode_atol"])
    ode_max_steps = int(config["ode_max_steps"])
    max_steps = int(config["max_steps"])
    learning_rate = float(config["learning_rate"])

    mvp_xy, mvp_z, mvp_target = _build_hamiltonian_mvps(n_qubits)
    psi0 = _neel_state(n_qubits)

    def block_step(
        psi: Any, block_params: Tuple[Any, Any, Any, Any]
    ) -> Tuple[Any, None]:
        s_l, j_l, d_l, angles_l = block_params
        t_l = t_min + (t_max - t_min) * jax.nn.sigmoid(s_l)
        j_coupling = jnp.tanh(j_l)
        delta_l = jnp.tanh(d_l)

        def h_analog(_t: Any) -> Any:
            return lambda state: j_coupling * mvp_xy(state) + delta_l * mvp_z(state)

        psi = tc.timeevol.ode_evol_global(
            h_analog,
            psi,
            tc.backend.stack([0.0, t_l]),
            rtol=ode_rtol,
            atol=ode_atol,
            max_steps=ode_max_steps,
            ode_backend="jaxode",
        )[-1]

        digital_layer = tc.Circuit(n_qubits, inputs=psi)
        for k in range(n_qubits):
            digital_layer.rz(k, theta=angles_l[k, 2])
            digital_layer.ry(k, theta=angles_l[k, 1])
            digital_layer.rz(k, theta=angles_l[k, 0])
        return digital_layer.state(), None

    def loss_fn(params: Tuple[Any, Any, Any, Any]) -> Any:
        psi, _ = jax.lax.scan(block_step, psi0, params)
        energy = tc.backend.real(tc.backend.sum(tc.backend.conj(psi) * mvp_target(psi)))
        return energy / n_qubits

    value_and_grad_fn = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(learning_rate)

    @jax.jit
    def train(
        init_params: Tuple[Any, Any, Any, Any],
    ) -> Tuple[Tuple[Any, Any, Any, Any], Any]:
        init_opt_state = optimizer.init(init_params)

        def opt_step(
            carry: Tuple[Tuple[Any, Any, Any, Any], Any], _: Any
        ) -> Tuple[Tuple[Tuple[Any, Any, Any, Any], Any], Any]:
            params, opt_state = carry
            energy_density, grads = value_and_grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), energy_density

        (final_params, _), history = jax.lax.scan(
            opt_step, (init_params, init_opt_state), None, length=max_steps
        )
        return final_params, history

    rng = np.random.default_rng(42)
    init_params = (
        jnp.zeros(n_blocks, dtype=jnp.float64),
        jnp.full(n_blocks, 0.1, dtype=jnp.float64),
        jnp.full(n_blocks, 0.1, dtype=jnp.float64),
        jnp.asarray(
            rng.normal(0.0, 0.1, size=(n_blocks, n_qubits, 3)), dtype=jnp.float64
        ),
    )

    final_params, energy_history = train(init_params)
    s_final, j_final, d_final, _ = final_params

    final_times = np.asarray(
        t_min + (t_max - t_min) * jax.nn.sigmoid(s_final), dtype=np.float64
    )
    final_couplings = np.asarray(jnp.tanh(j_final), dtype=np.float64)
    final_detunings = np.asarray(jnp.tanh(d_final), dtype=np.float64)
    history_np = np.asarray(energy_history, dtype=np.float64)

    return {
        "final_analog_times": final_times,
        "final_analog_couplings": final_couplings,
        "final_analog_detunings": final_detunings,
        "energy_density_history": history_np,
    }
