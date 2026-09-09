"""Batched, JAX-native L-BFGS-B.

Implements projected-gradient convergence, generalized Cauchy point, compact
limited-memory subspace minimization, and a bound-aware safeguarded line search.
"""
from __future__ import annotations

from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

RUNNING=jnp.int32(-1); CONVERGED_PGTOL=jnp.int32(0); CONVERGED_FTOL=jnp.int32(1)
ITERATION_LIMIT=jnp.int32(2); EVALUATION_LIMIT=jnp.int32(3); LINE_SEARCH_FAILED=jnp.int32(4); NONFINITE=jnp.int32(5)
STATUS_MESSAGES={0:"CONVERGENCE: NORM OF PROJECTED GRADIENT <= GTOL",1:"CONVERGENCE: REL_REDUCTION_OF_F <= FTOL",2:"ITERATION LIMIT REACHED",3:"FUNCTION/GRADIENT EVALUATION LIMIT REACHED",4:"ABNORMAL TERMINATION IN LINE SEARCH",5:"NON-FINITE OBJECTIVE OR GRADIENT"}

class _SingleResult(NamedTuple):
    initial_x:jax.Array; x:jax.Array; initial_fun:jax.Array; fun:jax.Array; status:jax.Array; iterations:jax.Array; evaluations:jax.Array; projected_grad_norm:jax.Array; active_lower:jax.Array; active_upper:jax.Array; ever_hit_lower:jax.Array; ever_hit_upper:jax.Array
class BatchedLbfgsbResult(NamedTuple):
    initial_x:jax.Array; x:jax.Array; initial_fun:jax.Array; fun:jax.Array; status:jax.Array; iterations:jax.Array; evaluations:jax.Array; projected_grad_norm:jax.Array; active_lower:jax.Array; active_upper:jax.Array; ever_hit_lower:jax.Array; ever_hit_upper:jax.Array
    def winner_index(self):
        losses=np.asarray(jax.device_get(self.fun)); finite=np.isfinite(losses)
        return 0 if not finite.any() else int(np.argmin(np.where(finite,losses,np.inf)))
    def messages(self): return [STATUS_MESSAGES.get(int(s),"UNKNOWN") for s in np.asarray(self.status)]
class _OuterState(NamedTuple):
    x:jax.Array; f:jax.Array; g:jax.Array; s_hist:jax.Array; y_hist:jax.Array; count:jax.Array; theta:jax.Array; iterations:jax.Array; evaluations:jax.Array; status:jax.Array; pg_norm:jax.Array; ever_hit_lower:jax.Array; ever_hit_upper:jax.Array

def projected_gradient(x,g,lower,upper):
    hl=jnp.isfinite(lower); hu=jnp.isfinite(upper)
    gi=jnp.where((g<0)&hu,jnp.maximum(x-upper,g),g)
    return jnp.where((g>=0)&hl,jnp.minimum(x-lower,gi),gi)
def projected_gradient_norm(x,g,lower,upper): return jnp.max(jnp.abs(projected_gradient(x,g,lower,upper)))

def _compact_parts(s,y,count,theta):
    m=s.shape[0]; valid=jnp.arange(m)<count; vf=valid.astype(s.dtype); s=s*vf[:,None]; y=y*vf[:,None]
    sy=s@y.T; d=jnp.diag(jnp.diag(sy)); ell=jnp.tril(sy,-1); sts=s@s.T
    k0=jnp.block([[-d,ell.T],[ell,theta*sts]]); active=jnp.concatenate([valid,valid]); aa=active[:,None]&active[None,:]
    k=jnp.where(aa,k0,jnp.zeros_like(k0))+jnp.diag((~active).astype(k0.dtype)); w=jnp.concatenate([y.T,theta*s.T],axis=1)
    return w,k
def _bmv(v,s,y,count,theta):
    w,k=_compact_parts(s,y,count,theta); return theta*v-w@jnp.linalg.solve(k,w.T@v)

def _cauchy(x,g,lo,hi,s,y,count,theta):
    n=x.shape[0]; hl=jnp.isfinite(lo); hu=jnp.isfinite(hi); d=-g
    blocked=(hl&(x<=lo)&(d<=0))|(hu&(x>=hi)&(d>=0))|(hl&hu&(lo==hi)); d=jnp.where(blocked,0.,d); free=~blocked
    tl=jnp.where(hl&(d<0),(lo-x)/d,jnp.inf); tu=jnp.where(hu&(d>0),(hi-x)/d,jnp.inf); br=jnp.minimum(tl,tu); order=jnp.argsort(br); sb=br[order]
    class C(NamedTuple): z:jax.Array; d:jax.Array; free:jax.Array; tp:jax.Array; done:jax.Array
    st=C(jnp.zeros_like(x),d,free,jnp.array(0.,x.dtype),jnp.array(False))
    def dtmin(z,d):
        der=jnp.dot(d,g+_bmv(z,s,y,count,theta)); curv=jnp.dot(d,_bmv(d,s,y,count,theta)); return jnp.where(curv>0,jnp.maximum(-der/curv,0.),0.)
    def step(j,st):
        tj=sb[j]; i=order[j]; gap=jnp.maximum(tj-st.tp,0.); dt=dtmin(st.z,st.d); stop=(~st.done)&(dt<=gap); cross=(~st.done)&(~stop)&jnp.isfinite(tj)
        z=jnp.where(stop,st.z+dt*st.d,st.z); z=jnp.where(cross,z+gap*st.d,z); di=st.d[i]; bz=jnp.where(di>0,hi[i]-x[i],lo[i]-x[i]); z=z.at[i].set(jnp.where(cross,bz,z[i]))
        return C(z,st.d.at[i].set(jnp.where(cross,0.,st.d[i])),st.free.at[i].set(jnp.where(cross,False,st.free[i])),jnp.where(cross,tj,st.tp),st.done|stop)
    st=lax.fori_loop(0,n,step,st); z=jnp.where(st.done,st.z,st.z+dtmin(st.z,st.d)*st.d)
    return jnp.clip(x+z,lo,hi),st.free

def _subspace(x,g,xcp,free,lo,hi,s,y,count,theta):
    r=g+_bmv(xcp-x,s,y,count,theta); rf=jnp.where(free,r,0.); w,k=_compact_parts(s,y,count,theta); wf=w*free[:,None].astype(w.dtype)
    h=k-(wf.T@wf)/theta; q=wf.T@rf; d=-(rf/theta+(wf@jnp.linalg.solve(h,q))/(theta*theta)); d=jnp.where(free,d,0.)
    projected=jnp.clip(xcp+d,lo,hi); hl=jnp.isfinite(lo); hu=jnp.isfinite(hi)
    al=jnp.where(free&hl&(d<0),(lo-xcp)/d,jnp.inf); au=jnp.where(free&hu&(d>0),(hi-xcp)/d,jnp.inf); a=jnp.maximum(0.,jnp.minimum(1.,jnp.min(jnp.minimum(al,au))))
    safe=jnp.clip(xcp+a*d,lo,hi); return jnp.where(jnp.dot(projected-x,g)<=0,projected,safe)

def _line_search(vg,x,f,g,z,lo,hi,maxls,budget,payload):
    """Bound-aware safeguarded strong-Wolfe search; returns best finite point on failure."""
    d=z-x; gd0=jnp.dot(g,d); dn=jnp.linalg.norm(d); hl=jnp.isfinite(lo); hu=jnp.isfinite(hi)
    al=jnp.where(hl&(d<0),(lo-x)/d,jnp.inf); au=jnp.where(hu&(d>0),(hi-x)/d,jnp.inf); stmax=jnp.maximum(0.,jnp.min(jnp.minimum(al,au))); stmax=jnp.where(jnp.any(hl|hu),stmax,1e10)
    class L(NamedTuple): k:jax.Array; a:jax.Array; alo:jax.Array; ahi:jax.Array; flo:jax.Array; glo:jax.Array; done:jax.Array; ok:jax.Array; nonfinite:jax.Array; xb:jax.Array; fb:jax.Array; gb:jax.Array; xl:jax.Array; fl:jax.Array; gl:jax.Array
    a0=jnp.minimum(1.,stmax); init=L(jnp.int32(0),a0,jnp.array(0.,f.dtype),stmax,f,gd0,jnp.array(False),jnp.array(False),jnp.array(False),x,f,g,x,f,g)
    limit=jnp.minimum(jnp.int32(maxls),jnp.maximum(jnp.int32(budget),0)); valid=(dn>0)&jnp.isfinite(dn)&(gd0<0)&jnp.isfinite(gd0)&(stmax>0)
    def cond(st): return (st.k<limit)&(~st.done)
    def body(st):
        xt=jnp.clip(x+st.a*d,lo,hi); ft,gt=vg(xt,*payload); finite=jnp.isfinite(ft)&jnp.all(jnp.isfinite(gt)); gd=jnp.dot(gt,d)
        better=finite&(ft<st.fb); xb=jnp.where(better,xt,st.xb); fb=jnp.where(better,ft,st.fb); gb=jnp.where(better,gt,st.gb)
        arm=ft<=f+1e-3*st.a*gd0; wolfe=jnp.abs(gd)<=0.9*(-gd0); ok=finite&arm&wolfe
        # Safeguarded bracket update: cubic interpolation is deliberately avoided only when its denominator is unstable.
        high=(~finite)|(~arm)|(ft>=st.flo); alo=jnp.where(high,st.alo,st.a); flo=jnp.where(high,st.flo,ft); glo=jnp.where(high,st.glo,gd); ahi=jnp.where(high,st.a,st.ahi)
        sign=(~high)&(gd>=0); ahi=jnp.where(sign,st.a,ahi); mid=0.5*(alo+ahi); grow=jnp.minimum(stmax,jnp.maximum(st.a*2.,st.a+0.1*(stmax-st.a))); bracket=ahi<stmax; an=jnp.where(bracket,mid,grow)
        return L(st.k+1,an,alo,ahi,flo,glo,ok|(~finite),ok,~finite,xb,fb,gb,xt,ft,gt)
    out=lax.cond(valid&(limit>0),lambda _:lax.while_loop(cond,body,init),lambda _:init,None)
    xr=jnp.where(out.ok,out.xl,out.xb); fr=jnp.where(out.ok,out.fl,out.fb); gr=jnp.where(out.ok,out.gl,out.gb)
    return xr,fr,gr,out.k,out.ok,out.nonfinite

def _push(s,y,count,sn,yn):
    m=s.shape[0]
    def add(_): return s.at[count].set(sn),y.at[count].set(yn),count+1
    def roll(_): return jnp.concatenate([s[1:],sn[None,:]]),jnp.concatenate([y[1:],yn[None,:]]),count
    return lax.cond(count<m,add,roll,None)

def _single(vg,x0,lo,hi,maxcor,ftol,gtol,maxiter,maxfun,maxls,payload):
    dtype=x0.dtype; lo=lo.astype(dtype); hi=hi.astype(dtype); x0=jnp.clip(x0,lo,hi); f0,g0=vg(x0,*payload); finite=jnp.isfinite(f0)&jnp.all(jnp.isfinite(g0)); pg=projected_gradient_norm(x0,g0,lo,hi)
    status=jnp.where(~finite,NONFINITE,jnp.where(pg<=gtol,CONVERGED_PGTOL,RUNNING)); status=jnp.where((status==RUNNING)&(maxfun<=1),EVALUATION_LIMIT,status); hitl=jnp.isfinite(lo)&(x0==lo); hitu=jnp.isfinite(hi)&(x0==hi)
    st=_OuterState(x0,f0,g0,jnp.zeros((maxcor,x0.size),dtype),jnp.zeros((maxcor,x0.size),dtype),jnp.int32(0),jnp.array(1.,dtype),jnp.int32(0),jnp.int32(1),status,pg,hitl,hitu); eps=jnp.finfo(dtype).eps
    def cond(st): return st.status==RUNNING
    def body(st):
        xcp,free=_cauchy(st.x,st.g,lo,hi,st.s_hist,st.y_hist,st.count,st.theta); z=lax.cond((st.count>0)&jnp.any(free),lambda _:_subspace(st.x,st.g,xcp,free,lo,hi,st.s_hist,st.y_hist,st.count,st.theta),lambda _:xcp,None)
        compact=jnp.all(jnp.isfinite(xcp))&jnp.all(jnp.isfinite(z))
        def refresh(_): return st._replace(s_hist=jnp.zeros_like(st.s_hist),y_hist=jnp.zeros_like(st.y_hist),count=jnp.int32(0),theta=jnp.array(1.,dtype))
        def search(_):
            xn,fn,gn,ne,ok,nf=_line_search(vg,st.x,st.f,st.g,z,lo,hi,maxls,maxfun-st.evaluations,payload); ev=st.evaluations+ne
            def fail(_):
                ss=jnp.where(nf,NONFINITE,jnp.where(ev>=maxfun,EVALUATION_LIMIT,LINE_SEARCH_FAILED)); p=projected_gradient_norm(xn,gn,lo,hi); return st._replace(x=xn,f=fn,g=gn,evaluations=ev,status=ss,pg_norm=p,ever_hit_lower=st.ever_hit_lower|(jnp.isfinite(lo)&(xn==lo)),ever_hit_upper=st.ever_hit_upper|(jnp.isfinite(hi)&(xn==hi)))
            def accept(_):
                nit=st.iterations+1; p=projected_gradient_norm(xn,gn,lo,hi); rel=(st.f-fn)/jnp.maximum(jnp.maximum(jnp.abs(st.f),jnp.abs(fn)),1.); ss=jnp.where(p<=gtol,CONVERGED_PGTOL,jnp.where(rel<=ftol,CONVERGED_FTOL,jnp.where(ev>=maxfun,EVALUATION_LIMIT,jnp.where(nit>=maxiter,ITERATION_LIMIT,RUNNING))))
                sn=xn-st.x; yn=gn-st.g; sy=jnp.dot(sn,yn); yy=jnp.dot(yn,yn); good=(sy>eps*yy)&jnp.isfinite(sy)&(sy>0)
                def yes(_): sh,yh,c=_push(st.s_hist,st.y_hist,st.count,sn,yn); return sh,yh,c,yy/sy
                sh,yh,c,th=lax.cond(good,yes,lambda _:(st.s_hist,st.y_hist,st.count,st.theta),None)
                return _OuterState(xn,fn,gn,sh,yh,c,th,nit,ev,ss,p,st.ever_hit_lower|(jnp.isfinite(lo)&(xn==lo)),st.ever_hit_upper|(jnp.isfinite(hi)&(xn==hi)))
            return lax.cond(ok,accept,fail,None)
        return lax.cond(compact,search,refresh,None)
    out=lax.while_loop(cond,body,st); return _SingleResult(x0,out.x,f0,out.f,out.status,out.iterations,out.evaluations,out.pg_norm,jnp.isfinite(lo)&(out.x==lo),jnp.isfinite(hi)&(out.x==hi),out.ever_hit_lower,out.ever_hit_upper)

class BatchedLbfgsb:
    def __init__(self,loss_fn:Callable[...,jax.Array],lower,upper,*,maxcor=10,ftol=2.220446049250313e-9,gtol=1e-5,maxiter=15000,maxfun=15000,maxls=20):
        lo=np.asarray(lower); hi=np.asarray(upper)
        if lo.ndim!=1 or hi.shape!=lo.shape: raise ValueError("lower and upper must be same-shape one-dimensional arrays")
        if np.any(lo>hi): raise ValueError("each lower bound must be <= its upper bound")
        if min(maxcor,maxiter,maxfun,maxls)<1: raise ValueError("limits must be positive")
        self.loss_fn=loss_fn; self.lower=jnp.asarray(lo); self.upper=jnp.asarray(hi); self.maxcor=int(maxcor); self.ftol=float(ftol); self.gtol=float(gtol); self.maxiter=int(maxiter); self.maxfun=int(maxfun); self.maxls=int(maxls); self._compiled=None; self._compiled_signature=None; self._build()
    def _build(self):
        vg=jax.value_and_grad(self.loss_fn); lo=self.lower; hi=self.upper; cfg=(self.maxcor,self.ftol,self.gtol,self.maxiter,self.maxfun,self.maxls)
        def batched(starts,*payload):
            def one(x,*pl): return _single(vg,x,lo,hi,*cfg,pl)
            return jax.vmap(one,in_axes=(0,)+(None,)*len(payload))(starts,*payload)
        self._jitted=jax.jit(batched)
    def _validate(self,s):
        if s.ndim!=2: raise ValueError("starts must have shape (n_starts, n_parameters)")
        if s.shape[1]!=self.lower.shape[0]: raise ValueError("starts/bounds dimension mismatch")
    @staticmethod
    def _sig(s,payload): return tuple((np.asarray(x).shape,np.asarray(x).dtype.str) for x in jax.tree.leaves((s,payload)))
    def compile(self,starts,*payload):
        starts=jnp.asarray(starts); self._validate(starts); self._compiled=self._jitted.lower(starts,*payload).compile(); self._compiled_signature=self._sig(starts,payload); return self
    def run(self,starts,*payload):
        starts=jnp.asarray(starts); self._validate(starts); raw=self._compiled(starts,*payload) if self._compiled is not None and self._compiled_signature==self._sig(starts,payload) else self._jitted(starts,*payload); return BatchedLbfgsbResult(*raw)
