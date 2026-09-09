import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
from jax_lbfgsb import BatchedLbfgsb

def scipy_each(fun,starts,lo,hi):
    return [minimize(fun,x,method="L-BFGS-B",bounds=list(zip(lo,hi))) for x in np.asarray(starts)]

def test_interior_quadratic_per_start():
    target=jnp.array([.25,-.5,1.25])
    def f(x): return .5*jnp.sum((x-target)**2)
    def fn(x): return .5*np.sum((x-np.asarray(target))**2)
    lo=np.full(3,-3.); hi=np.full(3,3.); starts=np.array([[-2,2,0],[2.5,-2,2],[.3,-.4,1.1]])
    r=BatchedLbfgsb(f,lo,hi).run(starts); refs=scipy_each(fn,starts,lo,hi)
    np.testing.assert_allclose(np.asarray(r.fun),[x.fun for x in refs],atol=1e-8)

def test_multiple_active_bounds():
    target=jnp.array([-2.,3.,.5])
    def f(x): return jnp.sum((x-target)**2)
    lo=np.array([-1.,-1.,0.]); hi=np.array([1.,2.,1.]); starts=np.array([[0,0,.2],[.9,1.8,.9],[-.8,-.5,.1]])
    r=BatchedLbfgsb(f,lo,hi).run(starts); expected=np.array([-1.,2.,.5])
    np.testing.assert_allclose(np.asarray(r.x),np.broadcast_to(expected,r.x.shape),atol=2e-4)
    assert np.all(np.asarray(r.active_lower)[:,0]); assert np.all(np.asarray(r.active_upper)[:,1])

def test_coupled_quadratic():
    A=jnp.array([[5.,2.,.5],[2.,3.,1.],[.5,1.,2.]]); b=jnp.array([4.,-3.,2.])
    def f(x): return .5*x@A@x-b@x
    An,bn=np.asarray(A),np.asarray(b)
    def fn(x): return .5*x@An@x-bn@x
    lo=np.array([-.5,-.75,-.25]); hi=np.array([.8,.6,.7]); starts=np.array([[-.4,.5,.6],[.7,-.6,-.1],[0,0,0]])
    r=BatchedLbfgsb(f,lo,hi,gtol=1e-8).run(starts); refs=scipy_each(fn,starts,lo,hi)
    np.testing.assert_allclose(np.asarray(r.fun),[x.fun for x in refs],atol=2e-5)

def test_bounded_rosenbrock():
    def f(x): return jnp.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)
    starts=np.array([[-1.2,1.],[1.4,-.5],[0.,0.]])
    r=BatchedLbfgsb(f,[-1.5,-1.5],[1.5,1.5],gtol=1e-7,maxiter=1000).run(starts)
    assert np.all(np.asarray(r.fun)<1e-5)

def test_hierarchical_2c_plus_1():
    C=3; truth=jnp.array([.4,.8,.6,1.1,.9,.5,.7])
    def f(x,t):
        pair=x[:-1].reshape(C,2); tp=t[:-1].reshape(C,2); return jnp.mean(jnp.sum((pair-tp)**2,axis=1)+(x[-1]-t[-1])**2*jnp.arange(1,C+1))
    starts=jnp.array([[1.3,.2,.2,1.4,1.4,.2,1.4],[.2,1.4,1.4,.2,.2,1.4,.2]])
    r=BatchedLbfgsb(f,np.full(7,.1),np.full(7,1.5),gtol=1e-7).run(starts,truth)
    np.testing.assert_allclose(np.asarray(r.x),np.broadcast_to(np.asarray(truth),r.x.shape),atol=5e-4)

def test_compile_and_independent_counts():
    target=jnp.array([.2,-.4])
    def f(x): return jnp.sum((x-target)**2)
    starts=jnp.array([[.2,-.4],[.3,-.4],[2.,2.]])
    s=BatchedLbfgsb(f,[-3,-3],[3,3],gtol=1e-9); s.compile(starts); r=s.run(starts)
    assert int(r.iterations[0])==0 and int(r.evaluations[0])==1
