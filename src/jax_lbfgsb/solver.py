from __future__ import annotations

from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .core import (
    RUNNING, CONVERGED_PGTOL, CONVERGED_FTOL, ITERATION_LIMIT, EVALUATION_LIMIT,
    LINE_SEARCH_FAILED, NONFINITE, STATUS_MESSAGES, _SingleResult, BatchedLbfgsbResult,
    _OuterState, projected_gradient, projected_gradient_norm, _generalized_cauchy,
    _subspace_minimize,
)
from .line_search import _line_search

def _push_history(s_hist, y_hist, count, s, y):
    m = s_hist.shape[0]
    append_idx = jnp.minimum(count, m - 1)

    def not_full(_):
        sh = s_hist.at[append_idx].set(s)
        yh = y_hist.at[append_idx].set(y)
        return sh, yh, count + 1

    def full(_):
        sh = jnp.concatenate([s_hist[1:], s[None, :]], axis=0)
        yh = jnp.concatenate([y_hist[1:], y[None, :]], axis=0)
        return sh, yh, count

    return lax.cond(count < m, not_full, full, None)


def _single_solver(loss_fn, value_and_grad, x0, lower, upper, *, maxcor, ftol, gtol, maxiter, maxfun, maxls, payload):
    dtype = x0.dtype
    lower = lower.astype(dtype)
    upper = upper.astype(dtype)
    x0 = jnp.clip(x0, lower, upper)
    f0, g0 = value_and_grad(x0, *payload)
    finite0 = jnp.isfinite(f0) & jnp.all(jnp.isfinite(g0))
    pg0 = projected_gradient_norm(x0, g0, lower, upper)
    status0 = jnp.where(~finite0, NONFINITE, jnp.where(pg0 <= gtol, CONVERGED_PGTOL, RUNNING))
    status0 = jnp.where((status0 == RUNNING) & (maxfun <= 1), EVALUATION_LIMIT, status0)
    hit_l0 = jnp.isfinite(lower) & (x0 == lower)
    hit_u0 = jnp.isfinite(upper) & (x0 == upper)
    st0 = _OuterState(
        x0, f0, g0,
        jnp.zeros((maxcor, x0.shape[0]), dtype=dtype),
        jnp.zeros((maxcor, x0.shape[0]), dtype=dtype),
        jnp.int32(0), jnp.array(1.0, dtype), jnp.int32(0), jnp.int32(1), status0, pg0,
        hit_l0, hit_u0,
    )

    eps = jnp.finfo(dtype).eps

    def cond(st):
        return st.status == RUNNING

    def body(st):
        xcp, free = _generalized_cauchy(st.x, st.g, lower, upper, st.s_hist, st.y_hist, st.count, st.theta)

        def do_subspace(_):
            return _subspace_minimize(st.x, st.g, xcp, free, lower, upper, st.s_hist, st.y_hist, st.count, st.theta)

        z = lax.cond((st.count > 0) & jnp.any(free), do_subspace, lambda _: xcp, None)
        compact_ok = jnp.all(jnp.isfinite(xcp)) & jnp.all(jnp.isfinite(z))

        # Numerical compact-representation failure: refresh non-empty history
        # once.  If the history is already empty, retrying cannot change the
        # Cauchy point and would leave the outer while-loop spinning forever.
        def refresh(_):
            def clear_history(_):
                return st._replace(
                    s_hist=jnp.zeros_like(st.s_hist), y_hist=jnp.zeros_like(st.y_hist),
                    count=jnp.int32(0), theta=jnp.array(1.0, dtype),
                )

            return lax.cond(
                st.count > 0,
                clear_history,
                lambda _: st._replace(status=NONFINITE),
                None,
            )

        def search_and_update(_):
            budget = maxfun - st.evaluations
            xn, fn, gn, neval, ls_ok, ls_nonfinite = _line_search(
                value_and_grad, st.x, st.f, st.g, z, lower, upper, st.iterations,
                maxls, budget, payload,
            )
            evals = st.evaluations + neval

            def failed(_):
                # Reference L-BFGS-B restores the old point after a failed line search.
                # If curvature history exists, it clears the history and retries the
                # same outer iteration; only a failure with empty history is terminal.
                budget_exhausted = evals >= maxfun
                can_restart = (st.count > 0) & (~budget_exhausted) & (~ls_nonfinite)

                def restart(_):
                    return st._replace(
                        s_hist=jnp.zeros_like(st.s_hist),
                        y_hist=jnp.zeros_like(st.y_hist),
                        count=jnp.int32(0),
                        theta=jnp.array(1.0, dtype),
                        evaluations=evals,
                    )

                def terminate(_):
                    status = jnp.where(ls_nonfinite, NONFINITE,
                             jnp.where(budget_exhausted, EVALUATION_LIMIT, LINE_SEARCH_FAILED))
                    return st._replace(evaluations=evals, status=status)

                return lax.cond(can_restart, restart, terminate, None)

            def accepted(_):
                nit = st.iterations + 1
                pgn = projected_gradient_norm(xn, gn, lower, upper)
                rel = (st.f - fn) / jnp.maximum(jnp.maximum(jnp.abs(st.f), jnp.abs(fn)), 1.0)
                ftol_converged = (rel >= 0.0) & (rel <= ftol)
                status = jnp.where(pgn <= gtol, CONVERGED_PGTOL,
                         jnp.where(ftol_converged, CONVERGED_FTOL,
                         jnp.where(evals >= maxfun, EVALUATION_LIMIT,
                         jnp.where(nit >= maxiter, ITERATION_LIMIT, RUNNING))))
                s = xn - st.x
                y = gn - st.g
                sy = jnp.dot(s, y)
                yy = jnp.dot(y, y)
                # L-BFGS-B skips an update when y^T s is tiny relative to the
                # directional decrease scale -g_old^T s (not relative to y^T y).
                curvature_scale = jnp.maximum(-jnp.dot(st.g, s), 0.0)
                good = (sy > eps * curvature_scale) & jnp.isfinite(sy) & jnp.isfinite(yy) & (sy > 0)

                def hist_yes(_):
                    sh, yh, ct = _push_history(st.s_hist, st.y_hist, st.count, s, y)
                    th = yy / sy
                    return sh, yh, ct, th

                def hist_no(_):
                    return st.s_hist, st.y_hist, st.count, st.theta

                sh, yh, ct, th = lax.cond(good, hist_yes, hist_no, None)
                hl = st.ever_hit_lower | (jnp.isfinite(lower) & (xn == lower))
                hu = st.ever_hit_upper | (jnp.isfinite(upper) & (xn == upper))
                return _OuterState(xn, fn, gn, sh, yh, ct, th, nit, evals, status, pgn, hl, hu)

            return lax.cond(ls_ok, accepted, failed, None)

        return lax.cond(compact_ok, search_and_update, refresh, None)

    out = lax.while_loop(cond, body, st0)
    active_l = jnp.isfinite(lower) & (out.x == lower)
    active_u = jnp.isfinite(upper) & (out.x == upper)
    return _SingleResult(x0, out.x, f0, out.f, out.status, out.iterations, out.evaluations,
                         out.pg_norm, active_l, active_u, out.ever_hit_lower, out.ever_hit_upper)


class BatchedLbfgsb:
    """Optimize independent bounded starts as one JAX batch.

    Parameters mirror SciPy's useful L-BFGS-B controls.  ``ftol`` is the direct
    relative reduction threshold used by ``scipy.optimize.minimize``; SciPy's
    legacy ``factr`` equals ``ftol / finfo(float).eps``.
    """

    def __init__(
        self,
        loss_fn: Callable[..., jax.Array],
        lower,
        upper,
        *,
        maxcor: int = 10,
        ftol: float = 2.220446049250313e-9,
        gtol: float = 1.0e-5,
        maxiter: int = 15000,
        maxfun: int = 15000,
        maxls: int = 20,
    ):
        lower_np = np.asarray(lower)
        upper_np = np.asarray(upper)
        if lower_np.ndim != 1 or upper_np.ndim != 1 or lower_np.shape != upper_np.shape:
            raise ValueError("lower and upper must be same-shape one-dimensional arrays")
        if np.any(lower_np > upper_np):
            raise ValueError("each lower bound must be <= its upper bound")
        if maxcor < 1 or maxiter < 1 or maxfun < 1 or maxls < 1:
            raise ValueError("maxcor, maxiter, maxfun, and maxls must be positive")
        self.loss_fn = loss_fn
        self.lower = jnp.asarray(lower_np)
        self.upper = jnp.asarray(upper_np)
        self.maxcor = int(maxcor)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.maxiter = int(maxiter)
        self.maxfun = int(maxfun)
        self.maxls = int(maxls)
        self._compiled = None
        self._compiled_signature = None
        self._build_jit()

    def _build_jit(self):
        value_and_grad = jax.value_and_grad(self.loss_fn)
        lower = self.lower
        upper = self.upper
        cfg = dict(maxcor=self.maxcor, ftol=self.ftol, gtol=self.gtol,
                   maxiter=self.maxiter, maxfun=self.maxfun, maxls=self.maxls)

        def batched(starts, *payload):
            n_payload = len(payload)
            in_axes = (0,) + (None,) * n_payload

            def solve_one(x0, *pl):
                return _single_solver(self.loss_fn, value_and_grad, x0, lower, upper,
                                      payload=pl, **cfg)

            return jax.vmap(solve_one, in_axes=in_axes)(starts, *payload)

        self._jitted = jax.jit(batched)

    def _validate_starts(self, starts):
        if getattr(starts, "ndim", None) != 2:
            raise ValueError("starts must have shape (n_starts, n_parameters)")
        if starts.shape[1] != self.lower.shape[0]:
            raise ValueError(f"starts have {starts.shape[1]} parameters but bounds have {self.lower.shape[0]}")

    @staticmethod
    def _signature(starts, payload):
        leaves = jax.tree.leaves((starts, payload))
        sig = []
        for leaf in leaves:
            arr = np.asarray(leaf)
            sig.append((arr.shape, arr.dtype.str))
        return tuple(sig)

    def compile(self, starts, *payload):
        """Ahead-of-time compile for the supplied shapes/dtypes without running optimization."""
        starts = jnp.asarray(starts)
        self._validate_starts(starts)
        self._compiled = self._jitted.lower(starts, *payload).compile()
        self._compiled_signature = self._signature(starts, payload)
        return self

    def run(self, starts, *payload) -> BatchedLbfgsbResult:
        starts = jnp.asarray(starts)
        self._validate_starts(starts)
        signature = self._signature(starts, payload)
        if self._compiled is not None and self._compiled_signature == signature:
            raw = self._compiled(starts, *payload)
        else:
            raw = self._jitted(starts, *payload)
        return BatchedLbfgsbResult(*raw)
