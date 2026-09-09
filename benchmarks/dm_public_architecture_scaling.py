from __future__ import annotations

"""Scaling benchmark using the public demixing_model WNM architecture.

The trained checkpoint binary is not materialized in this clean runtime, so this
uses deterministic random weights with the same public compute architecture.
It mirrors the public MirrorAwareMu1Predictor shape, periodic 180x90 output,
density-asymmetry collapse, and 1-CCC objective.  This is a workload/scaling
proxy, not a WNM recovery/parity test.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from scipy.optimize import minimize

from jax_lbfgsb import BatchedLbfgsb

COUNTS = (1, 4, 8, 16, 32, 64, 128)
REPEATS = 3
SEED = 20260909
MU1_ROWS = 180
FEAT_COLS = 90
HIDDEN = (64, 128, 256)
LATENT = 16
NATIVE_ROWS = 64
CHANNELS = (128, 64, 32, 1)


def _init_dense(key, nin, nout, dtype):
    lim = jnp.sqrt(jnp.asarray(6.0 / (nin + nout), dtype=dtype))
    w = jax.random.uniform(key, (nin, nout), dtype=dtype, minval=-lim, maxval=lim)
    return w, jnp.zeros((nout,), dtype=dtype)


def _init_conv(key, kh, kw, cin, cout, dtype):
    lim = jnp.sqrt(jnp.asarray(6.0 / (kh * kw * (cin + cout)), dtype=dtype))
    w = jax.random.uniform(key, (kh, kw, cin, cout), dtype=dtype, minval=-lim, maxval=lim)
    return w, jnp.zeros((cout,), dtype=dtype)


def make_weights(dtype):
    keys = iter(jax.random.split(jax.random.PRNGKey(41073), 16))
    dense = []
    nin = 3
    for nout in HIDDEN:
        dense.append(_init_dense(next(keys), nin, nout, dtype))
        nin = nout
    dense.append(_init_dense(next(keys), nin, LATENT * (NATIVE_ROWS // 8) * 16, dtype))
    conv = []
    cin = LATENT
    for cout in CHANNELS[:-1]:
        conv.append(_init_conv(next(keys), 4, 4, cin, cout, dtype))
        cin = cout
    conv.append(_init_conv(next(keys), 3, 3, cin, CHANNELS[-1], dtype))
    return tuple(dense), tuple(conv)


def circular_conv_transpose(x, wb, strides):
    w, b = wb
    pad_rows = max(0, (w.shape[0] - strides[0] + 1) // 2)
    if pad_rows:
        x = jnp.concatenate((x[:, -pad_rows:], x, x[:, :pad_rows]), axis=1)
    y = jax.lax.conv_transpose(
        x,
        w,
        strides=strides,
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        transpose_kernel=False,
    ) + b
    if pad_rows:
        crop = pad_rows * strides[0]
        y = y[:, crop:-crop]
    return y


def periodic_resize_mu1(x):
    input_rows = x.shape[1]
    numerators = jnp.arange(MU1_ROWS, dtype=jnp.int32) * input_rows
    lo = numerators // MU1_ROWS
    hi = (lo + 1) % input_rows
    frac = (numerators % MU1_ROWS).astype(x.dtype) / MU1_ROWS
    return (
        jnp.take(x, lo, axis=1) * (1 - frac[None, :, None, None])
        + jnp.take(x, hi, axis=1) * frac[None, :, None, None]
    )


def network_apply(weights, params):
    dense, conv = weights
    h = params
    for w, b in dense[:-1]:
        h = jax.nn.relu(h @ w + b)
    w, b = dense[-1]
    h = jax.nn.relu(h @ w + b)
    h = h.reshape((-1, NATIVE_ROWS // 8, 16, LATENT))
    for wb in conv[:-1]:
        h = jax.nn.relu(circular_conv_transpose(h, wb, (2, 2)))
    h = circular_conv_transpose(h, conv[-1], (1, 1))
    h = periodic_resize_mu1(h)
    h = jax.image.resize(h, (h.shape[0], h.shape[1], FEAT_COLS, 1), method="linear")
    logits = h[..., 0]
    return logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)


def density_asymmetry(log_surface):
    p = jnp.exp(log_surface)
    mu = jnp.arange(-180.0, 180.0, 2.0, dtype=p.dtype)
    pos = (mu > 0) & (jnp.abs(mu) < 180)
    neg = (mu < 0) & (jnp.abs(mu) < 180)
    curve = (
        jnp.sum(jnp.where(pos[None, :, None], p, 0.0), axis=1)
        - jnp.sum(jnp.where(neg[None, :, None], p, 0.0), axis=1)
    )
    x = jnp.arange(-20, 21, dtype=p.dtype)
    kernel = jnp.exp(-0.5 * (x / jnp.asarray(10.0, p.dtype)) ** 2)
    kernel = kernel / jnp.sum(kernel)
    return jax.lax.conv_general_dilated(
        curve[..., None],
        kernel[:, None, None],
        (1,),
        "SAME",
        dimension_numbers=("NWC", "WIO", "NWC"),
    )[..., 0]


def ccc_loss(pred, target):
    pm = jnp.mean(pred, axis=-1)
    tm = jnp.mean(target, axis=-1)
    pc = pred - pm[..., None]
    tc = target - tm[..., None]
    covariance = jnp.mean(pc * tc, axis=-1)
    denominator = (
        jnp.mean(pc * pc, axis=-1)
        + jnp.mean(tc * tc, axis=-1)
        + (pm - tm) ** 2
    )
    denominator = jnp.where(denominator < 1e-10, 1.0, denominator)
    return 1.0 - 2.0 * covariance / denominator


def build_problem(dtype_name="float32", n_conditions=4):
    dtype = jnp.float32 if dtype_name == "float32" else jnp.float64
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    weights = make_weights(dtype)
    true_pair = jnp.asarray(
        [[38.0, 52.0], [46.0, 61.0], [57.0, 43.0], [67.0, 79.0]],
        dtype=dtype,
    )[:n_conditions]
    true_log = jnp.log(
        jnp.concatenate([true_pair.reshape(-1), jnp.asarray([71.0], dtype=dtype)])
    )

    def condition_params(logp):
        vals = jnp.exp(logp)
        pairs = vals[:-1].reshape(n_conditions, 2)
        shared = jnp.broadcast_to(vals[-1], (n_conditions, 1))
        return jnp.concatenate([pairs, shared], axis=1)

    target = jax.lax.stop_gradient(
        density_asymmetry(network_apply(weights, condition_params(true_log)))
    )
    feat = jnp.linspace(2.0, 180.0, FEAT_COLS, dtype=dtype)
    target = target + jnp.asarray(0.002, dtype) * jnp.sin(feat / 17.0)[None, :]

    def loss(logp):
        curves = density_asymmetry(network_apply(weights, condition_params(logp)))
        return jnp.mean(ccc_loss(curves, target))

    vg = jax.jit(jax.value_and_grad(loss))

    def scipy_fg(x):
        f, g = vg(jnp.asarray(x, dtype=dtype))
        f.block_until_ready()
        out_dtype = np.float32 if dtype_name == "float32" else np.float64
        return float(f), np.asarray(g, dtype=out_dtype)

    out_dtype = np.float32 if dtype_name == "float32" else np.float64
    lo = np.full(2 * n_conditions + 1, np.log(5.0), dtype=out_dtype)
    hi = np.full(2 * n_conditions + 1, np.log(200.0), dtype=out_dtype)
    return loss, scipy_fg, lo, hi


def median_time(fn, repeats):
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append(time.perf_counter() - t0)
    return float(np.median(values)), values


def run(dtype_name="float32", counts=COUNTS, repeats=REPEATS, maxiter=120):
    loss, scipy_fg, lo, hi = build_problem(dtype_name)
    rng = np.random.default_rng(SEED)
    starts_all = rng.uniform(lo, hi, size=(max(counts), len(lo))).astype(lo.dtype)
    bounds = list(zip(lo, hi))
    scipy_fg(starts_all[0])
    rows = []

    for n_starts in counts:
        starts = starts_all[:n_starts]
        solver = BatchedLbfgsb(
            loss,
            lo,
            hi,
            maxcor=10,
            gtol=1e-5 if dtype_name == "float32" else 1e-7,
            ftol=1e-8,
            maxiter=maxiter,
            maxfun=maxiter * 10,
            maxls=20,
        )
        t0 = time.perf_counter()
        solver.compile(jnp.asarray(starts))
        compile_s = time.perf_counter() - t0
        warm = solver.run(jnp.asarray(starts))
        jax.block_until_ready(warm.fun)

        def jax_run():
            result = solver.run(jnp.asarray(starts))
            jax.block_until_ready(result.fun)
            return result

        jax_s, jax_times = median_time(jax_run, repeats)
        jax_result = jax_run()

        def scipy_run():
            output = []
            for x0 in starts:
                output.append(
                    minimize(
                        lambda x: scipy_fg(x),
                        x0,
                        jac=True,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={
                            "maxcor": 10,
                            "gtol": 1e-5 if dtype_name == "float32" else 1e-7,
                            "ftol": 1e-8,
                            "maxiter": maxiter,
                            "maxfun": maxiter * 10,
                            "maxls": 20,
                        },
                    )
                )
            return output

        scipy_times = []
        scipy_result = None
        for i in range(repeats):
            t0 = time.perf_counter()
            current = scipy_run()
            scipy_times.append(time.perf_counter() - t0)
            if i == 0:
                scipy_result = current
        scipy_s = float(np.median(scipy_times))

        scorer = jax.jit(jax.vmap(loss))
        jscore = np.asarray(scorer(jax_result.x))
        sx = jnp.asarray(np.stack([r.x for r in scipy_result]), dtype=jax_result.x.dtype)
        sscore = np.asarray(scorer(sx))
        delta = np.abs(jscore - sscore)

        row = {
            "dtype": dtype_name,
            "n_starts": n_starts,
            "compile_s": compile_s,
            "jax_s": jax_s,
            "scipy_s": scipy_s,
            "speedup": scipy_s / jax_s,
            "within_1e-5": float(np.mean(delta <= 1e-5)),
            "max_abs_loss_diff": float(np.max(delta)),
            "jax_eval_mean": float(np.mean(np.asarray(jax_result.evaluations))),
            "scipy_eval_mean": float(np.mean([r.nfev for r in scipy_result])),
            "jax_times_s": jax_times,
            "scipy_times_s": scipy_times,
        }
        rows.append(row)
        print(
            dtype_name,
            n_starts,
            f"JAX={jax_s:.4f}s SciPy={scipy_s:.4f}s ",
            f"speedup={row['speedup']:.2f}x agree={row['within_1e-5']:.3f}",
            flush=True,
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument(
        "--output", type=Path, default=Path("dm_public_architecture_scaling_results.json")
    )
    args = parser.parse_args()
    counts = tuple(int(x) for x in args.counts.split(",") if x)
    rows = run(args.dtype, counts=counts, repeats=args.repeats, maxiter=args.maxiter)
    payload = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "devices": [str(d) for d in jax.devices()],
        "source_model": (
            "public demixing_model MirrorAwareMu1Predictor architecture; "
            "deterministic random weights (checkpoint binary unavailable)"
        ),
        "source_files": [
            "neural_network_optimization/mirror_aware_model.py",
            "shared/config.py",
            "shared/mu1_axis.py",
            "shared/utils.py",
            "model_fit_to_data/density_objective.py",
        ],
        "surface_shape": [MU1_ROWS, FEAT_COLS],
        "conditions": 4,
        "parameters": 9,
        "repeats": args.repeats,
        "maxiter": args.maxiter,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print("WROTE", args.output)
