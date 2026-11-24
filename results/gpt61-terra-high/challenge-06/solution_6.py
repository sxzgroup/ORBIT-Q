import os
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import tensorcircuit as tc

tc.set_backend('jax')
jax.config.update('jax_enable_x64', True)
tc.set_dtype('complex128')

def run_solution(config):
    n, blocks = config['n_qubits'], config['n_blocks']
    def terms(kind):
        ps, ws = [], []
        for i in range(n-1):
            for a, w in kind:
                p = [0]*n
                p[i] = p[i+1] = a
                ps.append(p); ws.append(w)
        return ps, ws
    xy, xyw = terms([(1, 1.), (2, 1.)])
    zz, zzw = terms([(3, 1.1)])
    z, zw = [], []
    for i in range(n):
        p = [0]*n; p[i] = 3
        z.append(p); zw.append((-1.)**i)
    xyop = tc.quantum.PauliStringSum2MVP(xy, xyw)
    zop = tc.quantum.PauliStringSum2MVP(z, zw)
    target = tc.quantum.PauliStringSum2MVP(xy+zz+z, [0.7]*len(xy)+zzw+[0.25*w for w in zw])
    c = tc.Circuit(n)
    for i in range(1,n,2): c.x(i)
    init = c.state()
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=config['ode_rtol'], atol=config['ode_atol'])
    def loss(params):
        a, rot = params
        def block(state, inputs):
            p, angles = inputs
            t = config['t_min']+(config['t_max']-config['t_min'])*jax.nn.sigmoid(p[0])
            coupling, detuning = jnp.tanh(p[1]), jnp.tanh(p[2])
            def rhs(time, y, args):
                return -1j*t*(coupling*xyop(y)+detuning*zop(y))
            sol = diffrax.diffeqsolve(diffrax.ODETerm(rhs), solver, t0=0., t1=1., dt0=None, y0=state, stepsize_controller=controller, max_steps=config['ode_max_steps'])
            circ = tc.Circuit(n, inputs=sol.ys[0])
            for k in range(n):
                circ.rz(k, theta=angles[k,2]); circ.ry(k, theta=angles[k,1]); circ.rz(k, theta=angles[k,0])
            return circ.state(), None
        state, _ = jax.lax.scan(block, init, (a, rot))
        return jnp.real(jnp.vdot(state, target(state)))/n
    rng = np.random.default_rng(42)
    params = (jnp.array(np.tile([0.,0.1,0.1], (blocks,1))), jnp.array(rng.normal(0,0.1,(blocks,n,3))))
    value_grad = jax.jit(jax.value_and_grad(loss))
    m = jax.tree.map(jnp.zeros_like, params); v = jax.tree.map(jnp.zeros_like, params)
    @jax.jit
    def update(params,m,v,step):
        value, grad = value_grad(params)
        m = jax.tree.map(lambda x,g: .9*x+.1*g,m,grad)
        v = jax.tree.map(lambda x,g: .999*x+.001*g*g,v,grad)
        params = jax.tree.map(lambda p,x,y: p-config['learning_rate']*(x/(1-.9**step))/(jnp.sqrt(y/(1-.999**step))+1e-8),params,m,v)
        return params,m,v,value
    history=[]
    for i in range(config['max_steps']):
        params,m,v,value=update(params,m,v,i+1)
        history.append(float(value))
    a=np.asarray(params[0])
    return {'final_analog_times': np.asarray(config['t_min']+(config['t_max']-config['t_min'])*jax.nn.sigmoid(a[:,0])), 'final_analog_couplings': np.tanh(a[:,1]), 'final_analog_detunings': np.tanh(a[:,2]), 'energy_density_history': np.array(history)}
