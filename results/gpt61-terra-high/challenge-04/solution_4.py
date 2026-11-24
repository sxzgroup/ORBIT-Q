import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

jax.config.update('jax_enable_x64', True)
tc.set_backend('jax')
tc.set_dtype('complex128')
tc.set_contractor('greedy')

def run_solution(config):
    n = config['n_qubits']
    theta = config['entangler_angle']
    z = tc.gates.z()
    def table(p):
        a,b=p
        k=[jnp.array([[jnp.sqrt(1-a),0],[0,jnp.sqrt(1-b)]],dtype=jnp.complex128),
           jnp.array([[0,jnp.sqrt(b)],[0,0]],dtype=jnp.complex128),
           jnp.array([[0,0],[jnp.sqrt(a),0]],dtype=jnp.complex128)]
        rows=[]
        for state in range(4):
            c=tc.DMCircuit(n)
            if state == 0:
                c.h(0)
                for i in range(1,n): c.cnot(0,i)
            elif state == 1:
                for i in range(0,n,2):
                    c.x(i+1); c.h(i); c.cnot(i,i+1)
            elif state == 3:
                for i in range(n): c.h(i)
            for start in (0,1):
                for i in range(start,n-1,2):
                    c.rxx(i,i+1,theta=theta)
                    c.apply_general_kraus(k,i)
                    c.apply_general_kraus(k,i+1)
            rows.append(jnp.stack([jnp.real(c.expectation((tc.gates.z(),[i]),reuse=False)) for i in range(n)] + [jnp.real(c.expectation(*[(tc.gates.z(),[i]) for i in range(n)],reuse=False))]))
        return jnp.stack(rows)
    target=table(jnp.array([config['[REDACTED]_p01'],config['[REDACTED]_p10']]))
    def loss(r):
        return jnp.mean((table(jax.nn.sigmoid(r))-target)**2)
    grad=jax.jit(jax.value_and_grad(loss))
    init=jnp.array([config['initial_p01'],config['initial_p10']])
    r=jnp.log(init/(1-init))
    optim=optax.adam(config['learning_rate'])
    optstate=optim.init(r)
    history=[]
    for _ in range(config['max_steps']):
        l,g=grad(r)
        updates,optstate=optim.update(g,optstate,r)
        r=optax.apply_updates(r,updates)
        history.append(l)
    return {'loss_history':np.asarray(history), 'final_probabilities':np.asarray(jax.nn.sigmoid(r)), 'fitted_expectations':np.asarray(table(jax.nn.sigmoid(r)))}
