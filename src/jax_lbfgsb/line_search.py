from __future__ import annotations

from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

class _MtState(NamedTuple):
    brackt: jax.Array
    stage: jax.Array
    ginit: jax.Array
    gtest: jax.Array
    gx: jax.Array
    gy: jax.Array
    finit: jax.Array
    fx: jax.Array
    fy: jax.Array
    stx: jax.Array
    sty: jax.Array
    stmin: jax.Array
    stmax: jax.Array
    width: jax.Array
    width1: jax.Array


class _LineLoop(NamedTuple):
    k: jax.Array
    mt: _MtState
    stp: jax.Array
    done: jax.Array
    term_code: jax.Array  # 0 none, 1 conv, 2 warning, 3 nonfinite
    x_last: jax.Array
    f_last: jax.Array
    g_last: jax.Array
    x_best: jax.Array
    f_best: jax.Array
    g_best: jax.Array


def _dcstep(stx, fx, dx, sty, fy, dy, stp, brackt, fp, dp, stpmin, stpmax):
    """JAX form of the four-case safeguarded More-Thuente step."""
    sgnd = dp * (dx / jnp.abs(dx))
    p66 = jnp.array(0.66, stp.dtype)

    def case1(_):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
        gamma = scale * jnp.sqrt(jnp.maximum(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale)))
        gamma = jnp.where(stp < stx, -gamma, gamma)
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        stpf = jnp.where(jnp.abs(stpc - stx) < jnp.abs(stpq - stx), stpc, stpc + (stpq - stpc) / 2.0)
        return stpf, jnp.array(True)

    def case2(_):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
        gamma = scale * jnp.sqrt(jnp.maximum(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale)))
        gamma = jnp.where(stp > stx, -gamma, gamma)
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        stpf = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)
        return stpf, jnp.array(True)

    def case3(_):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        scale = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
        gamma = scale * jnp.sqrt(jnp.maximum(0.0, (theta / scale) ** 2 - (dx / scale) * (dp / scale)))
        gamma = jnp.where(stp > stx, -gamma, gamma)
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        stpc = jnp.where((r < 0.0) & (gamma != 0.0), stp + r * (stx - stp), jnp.where(stp > stx, stpmax, stpmin))
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        inside = jnp.where(jnp.abs(stpc - stp) < jnp.abs(stpq - stp), stpc, stpq)
        inside = jnp.where(stp > stx, jnp.minimum(stp + p66 * (sty - stp), inside), jnp.maximum(stp + p66 * (sty - stp), inside))
        outside = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)
        outside = jnp.clip(outside, stpmin, stpmax)
        return jnp.where(brackt, inside, outside), brackt

    def case4(_):
        def bracketed(_):
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            scale = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dy)), jnp.abs(dp))
            gamma = scale * jnp.sqrt(jnp.maximum(0.0, (theta / scale) ** 2 - (dy / scale) * (dp / scale)))
            gamma = jnp.where(stp > sty, -gamma, gamma)
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            return stp + (p / q) * (sty - stp)
        stpf = lax.cond(brackt, bracketed, lambda _: jnp.where(stp > stx, stpmax, stpmin), operand=None)
        return stpf, brackt

    stpf, brackt_new = lax.cond(
        fp > fx,
        case1,
        lambda _: lax.cond(sgnd < 0.0, case2, lambda __: lax.cond(jnp.abs(dp) < jnp.abs(dx), case3, case4, None), None),
        None,
    )

    sty2 = jnp.where(fp > fx, stp, sty)
    fy2 = jnp.where(fp > fx, fp, fy)
    dy2 = jnp.where(fp > fx, dp, dy)
    opposite = (fp <= fx) & (sgnd < 0.0)
    sty2 = jnp.where(opposite, stx, sty2)
    fy2 = jnp.where(opposite, fx, fy2)
    dy2 = jnp.where(opposite, dx, dy2)
    improve = fp <= fx
    stx2 = jnp.where(improve, stp, stx)
    fx2 = jnp.where(improve, fp, fx)
    dx2 = jnp.where(improve, dp, dx)
    return stx2, fx2, dx2, sty2, fy2, dy2, stpf, brackt_new


def _mt_init(f, gd, stp, stpmin, stpmax):
    width = stpmax - stpmin
    return _MtState(
        jnp.array(False), jnp.int32(1), gd, 1.0e-3 * gd, gd, gd,
        f, f, f, jnp.array(0.0, f.dtype), jnp.array(0.0, f.dtype),
        jnp.array(0.0, f.dtype), stp + 4.0 * stp, width, width / 0.5,
    )


def _mt_update(mt: _MtState, f, gd, stp, stpmin, stpmax):
    # Keep the global DCSRCH step limits distinct from the evolving internal bracket.
    global_stpmin, global_stpmax = stpmin, stpmax
    ftest = mt.finit + stp * mt.gtest
    stage = jnp.where((mt.stage == 1) & (f <= ftest) & (gd >= 0.0), jnp.int32(2), mt.stage)

    warn = (
        (mt.brackt & ((stp <= mt.stmin) | (stp >= mt.stmax)))
        | (mt.brackt & ((mt.stmax - mt.stmin) <= 0.1 * mt.stmax))
        | ((stp == stpmax) & (f <= ftest) & (gd <= mt.gtest))
        | ((stp == stpmin) & ((f > ftest) | (gd >= mt.gtest)))
    )
    conv = (f <= ftest) & (jnp.abs(gd) <= 0.9 * (-mt.ginit))
    term = jnp.where(conv, jnp.int32(1), jnp.where(warn, jnp.int32(2), jnp.int32(0)))

    def terminate(_):
        return mt._replace(stage=stage), stp, term

    def advance(_):
        modified = (stage == 1) & (f <= mt.fx) & (f > ftest)

        def mod_step(_):
            fm = f - stp * mt.gtest
            fxm = mt.fx - mt.stx * mt.gtest
            fym = mt.fy - mt.sty * mt.gtest
            gm = gd - mt.gtest
            gxm = mt.gx - mt.gtest
            gym = mt.gy - mt.gtest
            vals = _dcstep(mt.stx, fxm, gxm, mt.sty, fym, gym, stp, mt.brackt, fm, gm, mt.stmin, mt.stmax)
            stx, fxm2, gxm2, sty, fym2, gym2, stpn, br = vals
            return stx, fxm2 + stx * mt.gtest, gxm2 + mt.gtest, sty, fym2 + sty * mt.gtest, gym2 + mt.gtest, stpn, br

        def plain_step(_):
            return _dcstep(mt.stx, mt.fx, mt.gx, mt.sty, mt.fy, mt.gy, stp, mt.brackt, f, gd, mt.stmin, mt.stmax)

        stx, fx, gx, sty, fy, gy, stpn, br = lax.cond(modified, mod_step, plain_step, None)
        shrink = br & (jnp.abs(sty - stx) >= 0.66 * mt.width1)
        stpn = jnp.where(shrink, stx + 0.5 * (sty - stx), stpn)
        width1 = jnp.where(br, mt.width, mt.width1)
        width = jnp.where(br, jnp.abs(sty - stx), mt.width)
        stmin = jnp.where(br, jnp.minimum(stx, sty), stpn + 1.1 * (stpn - stx))
        stmax = jnp.where(br, jnp.maximum(stx, sty), stpn + 4.0 * (stpn - stx))
        stpn = jnp.clip(stpn, global_stpmin, global_stpmax)
        impossible = br & (((stpn <= stmin) | (stpn >= stmax)) | ((stmax - stmin) <= 0.1 * stmax))
        stpn = jnp.where(impossible, stx, stpn)
        mt2 = _MtState(br, stage, mt.ginit, mt.gtest, gx, gy, mt.finit, fx, fy, stx, sty, stmin, stmax, width, width1)
        return mt2, stpn, jnp.int32(0)

    return lax.cond(term != 0, terminate, advance, None)


def _line_search(value_and_grad, x, f, g, z, lower, upper, outer_iter, maxls, eval_budget, payload):
    d = z - x
    dtd = jnp.dot(d, d)
    dnorm = jnp.sqrt(dtd)
    gdold = jnp.dot(g, d)
    has_l = jnp.isfinite(lower)
    has_u = jnp.isfinite(upper)
    constrained = jnp.any(has_l | has_u)
    boxed = jnp.all(has_l & has_u)

    ratio_l = jnp.where(has_l & (d < 0), (lower - x) / d, jnp.inf)
    ratio_u = jnp.where(has_u & (d > 0), (upper - x) / d, jnp.inf)
    stpmx_general = jnp.min(jnp.minimum(ratio_l, ratio_u))
    stpmx = jnp.where(constrained, jnp.where(outer_iter == 0, 1.0, stpmx_general), 1.0e10)
    stpmx = jnp.maximum(stpmx, 0.0)
    stp0 = jnp.where((outer_iter == 0) & (~boxed), jnp.minimum(1.0 / dnorm, stpmx), 1.0)
    stp0 = jnp.minimum(stp0, stpmx)
    mt0 = _mt_init(f, gdold, stp0, 0.0, stpmx)
    init = _LineLoop(
        jnp.int32(0), mt0, stp0, jnp.array(False), jnp.int32(0),
        x, f, g, x, f, g,
    )

    max_eval = jnp.minimum(jnp.int32(maxls), jnp.maximum(jnp.int32(eval_budget), 0))
    valid_direction = jnp.isfinite(dnorm) & (dnorm > 0) & jnp.isfinite(gdold) & (gdold < 0) & (stpmx > 0)

    def cond(st):
        return (st.k < max_eval) & (~st.done)

    def body(st):
        xt = jnp.where(st.stp == 1.0, z, x + st.stp * d)
        xt = jnp.clip(xt, lower, upper)
        ft, gt = value_and_grad(xt, *payload)
        finite = jnp.isfinite(ft) & jnp.all(jnp.isfinite(gt))
        better = finite & (ft < st.f_best)
        xb = jnp.where(better, xt, st.x_best)
        fb = jnp.where(better, ft, st.f_best)
        gb = jnp.where(better, gt, st.g_best)

        def finite_branch(_):
            gd = jnp.dot(gt, d)
            mt2, stp2, term = _mt_update(st.mt, ft, gd, st.stp, 0.0, stpmx)
            return _LineLoop(st.k + 1, mt2, stp2, term != 0, term, xt, ft, gt, xb, fb, gb)

        def nonfinite_branch(_):
            return _LineLoop(st.k + 1, st.mt, st.stp, jnp.array(True), jnp.int32(3), xt, ft, gt, xb, fb, gb)

        return lax.cond(finite, finite_branch, nonfinite_branch, None)

    out = lax.cond(valid_direction & (max_eval > 0), lambda _: lax.while_loop(cond, body, init), lambda _: init, None)
    success = valid_direction & ((out.term_code == 1) | (out.term_code == 2))
    nonfinite = out.term_code == 3
    xret = jnp.where(success, out.x_last, out.x_best)
    fret = jnp.where(success, out.f_last, out.f_best)
    gret = jnp.where(success, out.g_last, out.g_best)
    return xret, fret, gret, out.k, success, nonfinite
