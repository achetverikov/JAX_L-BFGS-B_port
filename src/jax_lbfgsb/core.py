"""Batched, JAX-native L-BFGS-B.

The implementation follows the L-BFGS-B algorithmic decomposition rather than
turning L-BFGS into a projected method: generalized Cauchy point, free-variable
subspace minimization with the compact limited-memory representation, and a
bound-aware More-Thuente line search.

The implementation intentionally uses recomputation of the compact B operator
inside the Cauchy breakpoint walk.  That is algebraically equivalent to the
incremental recurrence in the original Fortran but considerably easier to keep
inside shape-static JAX control flow.  It trades some arithmetic for a simpler,
accelerator-friendly state machine.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


RUNNING = jnp.int32(-1)
CONVERGED_PGTOL = jnp.int32(0)
CONVERGED_FTOL = jnp.int32(1)
ITERATION_LIMIT = jnp.int32(2)
EVALUATION_LIMIT = jnp.int32(3)
LINE_SEARCH_FAILED = jnp.int32(4)
NONFINITE = jnp.int32(5)

STATUS_MESSAGES = {
    0: "CONVERGENCE: NORM OF PROJECTED GRADIENT <= GTOL",
    1: "CONVERGENCE: REL_REDUCTION_OF_F <= FTOL",
    2: "ITERATION LIMIT REACHED",
    3: "FUNCTION/GRADIENT EVALUATION LIMIT REACHED",
    4: "ABNORMAL TERMINATION IN LINE SEARCH",
    5: "NON-FINITE OBJECTIVE OR GRADIENT",
}


class _SingleResult(NamedTuple):
    initial_x: jax.Array
    x: jax.Array
    initial_fun: jax.Array
    fun: jax.Array
    status: jax.Array
    iterations: jax.Array
    evaluations: jax.Array
    projected_grad_norm: jax.Array
    active_lower: jax.Array
    active_upper: jax.Array
    ever_hit_lower: jax.Array
    ever_hit_upper: jax.Array


class BatchedLbfgsbResult(NamedTuple):
    """One stable-order record per start, stored as batched arrays."""

    initial_x: jax.Array
    x: jax.Array
    initial_fun: jax.Array
    fun: jax.Array
    status: jax.Array
    iterations: jax.Array
    evaluations: jax.Array
    projected_grad_norm: jax.Array
    active_lower: jax.Array
    active_upper: jax.Array
    ever_hit_lower: jax.Array
    ever_hit_upper: jax.Array

    def winner_index(self) -> int:
        """Canonical winner: minimum finite final loss, ties by start order."""
        losses = np.asarray(jax.device_get(self.fun))
        finite = np.isfinite(losses)
        if not finite.any():
            return 0
        return int(np.argmin(np.where(finite, losses, np.inf)))

    def messages(self) -> list[str]:
        return [STATUS_MESSAGES.get(int(s), "UNKNOWN") for s in np.asarray(self.status)]


class _OuterState(NamedTuple):
    x: jax.Array
    f: jax.Array
    g: jax.Array
    s_hist: jax.Array  # (m, n), oldest -> newest
    y_hist: jax.Array
    count: jax.Array
    theta: jax.Array
    iterations: jax.Array
    evaluations: jax.Array
    status: jax.Array
    pg_norm: jax.Array
    ever_hit_lower: jax.Array
    ever_hit_upper: jax.Array


def projected_gradient(x: jax.Array, g: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
    """L-BFGS-B projected-gradient proxy, componentwise."""
    has_l = jnp.isfinite(lower)
    has_u = jnp.isfinite(upper)
    gi = jnp.where((g < 0) & has_u, jnp.maximum(x - upper, g), g)
    gi = jnp.where((g >= 0) & has_l, jnp.minimum(x - lower, gi), gi)
    return gi


def projected_gradient_norm(x: jax.Array, g: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
    return jnp.max(jnp.abs(projected_gradient(x, g, lower, upper)))


def _compact_parts(s_hist, y_hist, count, theta):
    """Return W and K for B = theta I - W K^{-1} W^T.

    History rows are chronological.  Inactive memory slots are embedded as an
    identity block so the solve retains a static (2m, 2m) shape under jit.
    """
    m = s_hist.shape[0]
    valid = jnp.arange(m) < count
    vf = valid.astype(s_hist.dtype)
    s = s_hist * vf[:, None]
    y = y_hist * vf[:, None]
    sy = s @ y.T
    d = jnp.diag(jnp.diag(sy))
    ell = jnp.tril(sy, -1)
    sts = s @ s.T
    k_raw = jnp.block([[-d, ell.T], [ell, theta * sts]])
    active = jnp.concatenate([valid, valid])
    aa = active[:, None] & active[None, :]
    k = jnp.where(aa, k_raw, jnp.zeros_like(k_raw))
    k = k + jnp.diag((~active).astype(k.dtype))
    w = jnp.concatenate([y.T, theta * s.T], axis=1)
    return w, k


def _bmv(v, s_hist, y_hist, count, theta):
    w, k = _compact_parts(s_hist, y_hist, count, theta)
    rhs = w.T @ v
    corr = jnp.linalg.solve(k, rhs)
    return theta * v - w @ corr


def _generalized_cauchy(x, g, lower, upper, s_hist, y_hist, count, theta):
    """First local minimizer along x(t)=P[x-t*g] over breakpoint segments."""
    n = x.shape[0]
    has_l = jnp.isfinite(lower)
    has_u = jnp.isfinite(upper)
    at_l = has_l & (x <= lower)
    at_u = has_u & (x >= upper)
    d0 = -g
    blocked_l = at_l & (d0 <= 0)
    blocked_u = at_u & (d0 >= 0)
    d0 = jnp.where(blocked_l | blocked_u, 0.0, d0)
    free0 = ~(blocked_l | blocked_u | (has_l & has_u & (lower == upper)))

    t_l = jnp.where(has_l & (d0 < 0), (lower - x) / d0, jnp.inf)
    t_u = jnp.where(has_u & (d0 > 0), (upper - x) / d0, jnp.inf)
    breaks = jnp.minimum(t_l, t_u)
    order = jnp.argsort(breaks)
    sorted_breaks = breaks[order]

    class CState(NamedTuple):
        z: jax.Array
        d: jax.Array
        free: jax.Array
        tprev: jax.Array
        done: jax.Array

    init = CState(jnp.zeros_like(x), d0, free0, jnp.array(0.0, x.dtype), jnp.array(False))

    def segment_min(z, d):
        grad_m = g + _bmv(z, s_hist, y_hist, count, theta)
        bd = _bmv(d, s_hist, y_hist, count, theta)
        der = jnp.dot(d, grad_m)
        curv = jnp.dot(d, bd)
        dt = jnp.where(curv > 0, jnp.maximum(-der / curv, 0.0), jnp.array(0.0, x.dtype))
        return dt

    def step(j, st: CState):
        tj = sorted_breaks[j]
        idx = order[j]
        gap = jnp.maximum(tj - st.tprev, 0.0)
        dtm = segment_min(st.z, st.d)
        stops_here = (~st.done) & (dtm <= gap)
        crosses = (~st.done) & (~stops_here) & jnp.isfinite(tj)
        z1 = jnp.where(stops_here, st.z + dtm * st.d, st.z)
        z1 = jnp.where(crosses, z1 + gap * st.d, z1)

        di = st.d[idx]
        bound_z = jnp.where(di > 0, upper[idx] - x[idx], lower[idx] - x[idx])
        z1 = z1.at[idx].set(jnp.where(crosses, bound_z, z1[idx]))
        d1 = st.d.at[idx].set(jnp.where(crosses, 0.0, st.d[idx]))
        free1 = st.free.at[idx].set(jnp.where(crosses, False, st.free[idx]))
        tprev1 = jnp.where(crosses, tj, st.tprev)
        return CState(z1, d1, free1, tprev1, st.done | stops_here)

    st = lax.fori_loop(0, n, step, init)
    dt_final = segment_min(st.z, st.d)
    z = jnp.where(st.done, st.z, st.z + dt_final * st.d)
    xcp = jnp.clip(x + z, lower, upper)
    return xcp, st.free


def _subspace_minimize(x, g, xcp, free, lower, upper, s_hist, y_hist, count, theta):
    """Compact-representation Newton step restricted to GCP free variables."""
    z = xcp - x
    r = g + _bmv(z, s_hist, y_hist, count, theta)
    rf = jnp.where(free, r, 0.0)
    w, k = _compact_parts(s_hist, y_hist, count, theta)
    wf = w * free[:, None].astype(w.dtype)
    h = k - (wf.T @ wf) / theta
    q = wf.T @ rf
    small = jnp.linalg.solve(h, q)
    d = -(rf / theta + (wf @ small) / (theta * theta))
    d = jnp.where(free, d, 0.0)

    projected = jnp.clip(xcp + d, lower, upper)
    dd_p = jnp.dot(projected - x, g)

    has_l = jnp.isfinite(lower)
    has_u = jnp.isfinite(upper)
    ratio_l = jnp.where(free & has_l & (d < 0), (lower - xcp) / d, jnp.inf)
    ratio_u = jnp.where(free & has_u & (d > 0), (upper - xcp) / d, jnp.inf)
    alpha = jnp.minimum(1.0, jnp.min(jnp.minimum(ratio_l, ratio_u)))
    alpha = jnp.maximum(alpha, 0.0)
    safeguarded = jnp.clip(xcp + alpha * d, lower, upper)
    return jnp.where(dd_p <= 0.0, projected, safeguarded)
