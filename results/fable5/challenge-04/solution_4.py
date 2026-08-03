"""Challenge 4: trainable Kraus noise calibration from multi-circuit data.

Vectorized density-matrix simulation on TensorCircuit-NG's MPS circuit
simulator. Qubit q of the 12-qubit register maps to two MPS sites: site 2q
carries the ket copy and site 2q+1 the bra copy of rho. Probe states are
prepared with framework gates (H/CNOT/X) applied to both copies, the RXX
entanglers act through the native rxx gate (theta on the ket copy, -theta on
the bra copy for the conjugate), and the asymmetric bit-flip channel is
expressed as explicit Kraus tensor algebra: the (3, 2, 2) Kraus stack is
contracted into the one-qubit superoperator sum_a K_a (x) K_a^*, which acts
on the adjacent (ket, bra) site pair after every entangler. Observable
traces Tr[O K(rho)] are overlaps (proj_with_mps) with product Bell-type MPS
prepared by framework gates, normalized by the trace overlap. Everything is
differentiable in the two channel probabilities; Adam fits them against a
synthetic target table generated first from the true probabilities.
"""
import numpy as np
import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax
import jax.numpy as jnp
import optax

_SPLIT = {"max_singular_values": 128}


def _kraus_stack(p01, p10):
    # K0 = sqrt(1-p01)|0><0| + sqrt(1-p10)|1><1|; K1 = sqrt(p10)|0><1|;
    # K2 = sqrt(p01)|1><0|. Index order (a, out, in).
    k = jnp.zeros((3, 2, 2), dtype=jnp.complex128)
    k = k.at[0, 0, 0].set(jnp.sqrt(1.0 - p01))
    k = k.at[0, 1, 1].set(jnp.sqrt(1.0 - p10))
    k = k.at[1, 0, 1].set(jnp.sqrt(p10))
    k = k.at[2, 1, 0].set(jnp.sqrt(p01))
    return k


def _channel_superop(p01, p10):
    # rho' = sum_a K_a rho K_a^dagger, vectorized on the (ket, bra) pair.
    k = _kraus_stack(p01, p10)
    s = jnp.einsum("aij,akl->ikjl", k, jnp.conj(k))
    return s.reshape(4, 4)


def _prep_probe(c, which, n):
    # Framework gates applied to ket sites (2q) and bra sites (2q+1); all
    # preparation matrices are real, so the bra copy uses the same gates.
    if which == "ghz":
        c.h(0)
        c.h(1)
        for q in range(n - 1):
            c.cnot(2 * q, 2 * q + 2)
            c.cnot(2 * q + 1, 2 * q + 3)
    elif which == "bell":
        for a in range(0, n - 1, 2):
            c.h(2 * a)
            c.h(2 * a + 1)
            c.cnot(2 * a, 2 * a + 2)
            c.cnot(2 * a + 1, 2 * a + 3)
            c.x(2 * a + 2)
            c.x(2 * a + 3)
    elif which == "plus":
        for q in range(n):
            c.h(2 * q)
            c.h(2 * q + 1)
    # "zero": |0...0> needs no gates


def _evolved(p01, p10, which, n, theta):
    s4 = _channel_superop(p01, p10)
    c = tc.MPSCircuit(2 * n, split=_SPLIT)
    _prep_probe(c, which, n)
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    for bonds in (even, odd):
        for (a, b) in bonds:
            c.rxx(2 * a, 2 * b, theta=theta)  # rxx(theta) = exp(-i theta XX / 2)
            c.rxx(2 * a + 1, 2 * b + 1, theta=-theta)  # conjugate on the bra copy
            c.any(2 * a, 2 * a + 1, unitary=s4)
            c.any(2 * b, 2 * b + 1, unitary=s4)
    return c


def _obs_mps(z_sites, n):
    # Product MPS encoding vec(prod Z)/2^(n/2): Bell pair per (ket, bra) site
    # pair, with a Z on the ket site where the observable acts.
    c = tc.MPSCircuit(2 * n, split=_SPLIT)
    for q in range(n):
        c.h(2 * q)
        c.cnot(2 * q, 2 * q + 1)
        if q in z_sites:
            c.z(2 * q)
    return c


def run_solution(config):
    n = int(config["n_qubits"])
    theta = float(config["entangler_angle"])
    max_steps = int(config["max_steps"])
    lr = float(config["learning_rate"])
    probes = ("ghz", "bell", "zero", "plus")

    def table(p01, p10):
        rows = []
        for which in probes:
            rho = _evolved(p01, p10, which, n, theta)
            denom = rho.proj_with_mps(_obs_mps(set(), n))
            vals = [
                tc.backend.real(rho.proj_with_mps(_obs_mps({i}, n)) / denom)
                for i in range(n)
            ]
            vals.append(
                tc.backend.real(rho.proj_with_mps(_obs_mps(set(range(n)), n)) / denom)
            )
            rows.append(jnp.stack(vals))
        return jnp.stack(rows)

    table_jit = jax.jit(table)
    # 1) generate the synthetic target table at the true probabilities
    target = table_jit(float(config["true_p01"]), float(config["true_p10"]))

    # 2) fit sigmoid-parameterized probabilities against the target table
    def loss_fn(r):
        p = jax.nn.sigmoid(r)
        return jnp.mean((table(p[0], p[1]) - target) ** 2)

    vg = jax.jit(jax.value_and_grad(loss_fn))
    p_init = np.array([float(config["initial_p01"]), float(config["initial_p10"])])
    r = jnp.asarray(np.log(p_init / (1.0 - p_init)))
    opt = optax.adam(lr)
    opt_state = opt.init(r)

    loss_hist = np.empty(max_steps, dtype=np.float64)
    for k in range(max_steps):
        loss, grad = vg(r)
        loss_hist[k] = float(loss)
        updates, opt_state = opt.update(grad, opt_state)
        r = optax.apply_updates(r, updates)

    p_final = jax.nn.sigmoid(r)
    fitted = table_jit(p_final[0], p_final[1])
    return {
        "loss_history": loss_hist,
        "final_probabilities": np.asarray(p_final, dtype=np.float64),
        "fitted_expectations": np.asarray(fitted, dtype=np.float64),
    }
