from __future__ import annotations

"""Scaling benchmark using the public demixing_model surface-network architecture.

By default this uses deterministic random weights; ``--checkpoint`` loads a
trusted MirrorAwareMu1Predictor checkpoint. It mirrors the current predictor
shape, periodic 180x90 output, density-asymmetry collapse, and 1-CCC objective.
This remains a synthetic workload, not an empirical DM recovery test.
"""

import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from benchmark_utils import (
    configure_jax,
    file_sha256,
    median_time,
    run_jax_chunks,
    set_platform_from_argv,
    solver_source_fingerprint,
)

set_platform_from_argv("cpu", deterministic_gpu_default=True)

import jax
import jax.numpy as jnp
from jax_lbfgsb import BatchedLbfgsb

COUNTS = (1, 4, 8)
REPEATS = 3
SEED = 20260909
MU1_ROWS = 180
FEAT_COLS = 90
HIDDEN = (64, 128, 256)
LATENT = 16
NATIVE_ROWS = 128
CHANNELS = (128, 64, 32, 1)


def _init_dense(key, nin, nout, dtype):
    lim = jnp.sqrt(jnp.asarray(6.0 / (nin + nout), dtype=dtype))
    w = jax.random.uniform(key, (nin, nout), dtype=dtype, minval=-lim, maxval=lim)
    return w, jnp.zeros((nout,), dtype=dtype)


def _init_conv(key, kh, kw, cin, cout, dtype):
    lim = jnp.sqrt(jnp.asarray(6.0 / (kh * kw * (cin + cout)), dtype=dtype))
    w = jax.random.uniform(key, (kh, kw, cin, cout), dtype=dtype, minval=-lim, maxval=lim)
    return w, jnp.zeros((cout,), dtype=dtype)


def make_weights(dtype, native_rows=NATIVE_ROWS):
    keys = iter(jax.random.split(jax.random.PRNGKey(41073), 16))
    dense = []
    nin = 3
    for nout in HIDDEN:
        dense.append(_init_dense(next(keys), nin, nout, dtype))
        nin = nout
    dense.append(_init_dense(next(keys), nin, LATENT * (native_rows // 8) * 16, dtype))
    conv = []
    cin = LATENT
    for cout in CHANNELS[:-1]:
        conv.append(_init_conv(next(keys), 4, 4, cin, cout, dtype))
        cin = cout
    conv.append(_init_conv(next(keys), 3, 3, cin, CHANNELS[-1], dtype))
    return tuple(dense), tuple(conv)


def load_checkpoint_weights(path, dtype):
    path = Path(path).resolve()
    dm_root = path.parent.parent
    sys.path[:0] = [str(dm_root), str(dm_root / "neural_network_optimization")]
    with path.open("rb") as stream:
        checkpoint = pickle.load(stream)
    params = checkpoint["params"]["params"]
    dense = tuple(
        (
            jnp.asarray(params[f"Dense_{i}"]["kernel"], dtype=dtype),
            jnp.asarray(params[f"Dense_{i}"]["bias"], dtype=dtype),
        )
        for i in range(4)
    )
    conv = tuple(
        (
            jnp.asarray(params[f"ConvTranspose_{i}"]["kernel"], dtype=dtype),
            jnp.asarray(params[f"ConvTranspose_{i}"]["bias"], dtype=dtype),
        )
        for i in range(4)
    )
    native_rows = 8 * dense[-1][0].shape[1] // (LATENT * 16)
    info = {
        "source_model": "demixing_model trained checkpoint",
        "checkpoint": path.name,
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_training_loss": checkpoint.get("loss"),
        "native_mu1_rows": native_rows,
    }
    return (dense, conv), info


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
    initial_rows = w.shape[1] // (LATENT * 16)
    h = h.reshape((-1, initial_rows, 16, LATENT))
    for wb in conv[:-1]:
        h = jax.nn.relu(circular_conv_transpose(h, wb, (2, 2)))
    h = circular_conv_transpose(h, conv[-1], (1, 1))
    h = periodic_resize_mu1(h)
    h = jax.image.resize(h, (h.shape[0], h.shape[1], FEAT_COLS, 1), method="linear")
    logits = h[..., 0]
    return logits - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)


def density_asymmetry(log_surface):
    # network_apply returns discrete mass (sum=1), while DM returns density
    # (sum*2=1) and integrates with dx=2.  Those factors cancel here.
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
    pad = kernel.shape[0] // 2
    curve = jnp.pad(curve, ((0, 0), (pad, pad)), mode="edge")
    return jax.lax.conv_general_dilated(
        curve[..., None],
        kernel[:, None, None],
        (1,),
        "VALID",
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


def build_problem(dtype_name="float32", n_conditions=4, checkpoint=None):
    dtype = jnp.float32 if dtype_name == "float32" else jnp.float64
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    if checkpoint is None:
        weights = make_weights(dtype)
        model_info = {
            "source_model": "current public demixing_model architecture; deterministic random weights",
            "native_mu1_rows": NATIVE_ROWS,
        }
    else:
        weights, model_info = load_checkpoint_weights(checkpoint, dtype)
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
    return loss, scipy_fg, lo, hi, model_info


def run(
    dtype_name="float32",
    counts=COUNTS,
    repeats=REPEATS,
    maxiter=120,
    ftol=1e-8,
    maxls=20,
    checkpoint=None,
    jax_batch_size=None,
):
    loss, scipy_fg, lo, hi, model_info = build_problem(dtype_name, checkpoint=checkpoint)
    rng = np.random.default_rng(SEED)
    starts_all = rng.uniform(lo, hi, size=(max(counts), len(lo))).astype(lo.dtype)
    bounds = list(zip(lo, hi))
    scipy_fg(starts_all[0])
    rows = []

    for n_starts in counts:
        starts = starts_all[:n_starts]
        batch_size = min(jax_batch_size or n_starts, n_starts)
        solver = BatchedLbfgsb(
            loss,
            lo,
            hi,
            maxcor=10,
            gtol=1e-5 if dtype_name == "float32" else 1e-7,
            ftol=ftol,
            maxiter=maxiter,
            maxfun=maxiter * 10,
            maxls=maxls,
        )
        t0 = time.perf_counter()
        solver.compile(jnp.asarray(starts[:batch_size]))
        compile_s = time.perf_counter() - t0
        warm = run_jax_chunks(solver, starts, batch_size)
        jax.block_until_ready(warm.fun)

        def jax_run():
            result = run_jax_chunks(solver, starts, batch_size)
            jax.block_until_ready(result.fun)
            return result

        jax_s, jax_times, jax_result = median_time(jax_run, repeats)

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
                            "ftol": ftol,
                            "maxiter": maxiter,
                            "maxfun": maxiter * 10,
                            "maxls": maxls,
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
        initial_score = np.asarray(scorer(jnp.asarray(starts)))
        jscore = np.asarray(scorer(jax_result.x))
        sx = jnp.asarray(np.stack([r.x for r in scipy_result]), dtype=jax_result.x.dtype)
        sscore = np.asarray(scorer(sx))
        signed_delta = jscore - sscore
        delta = np.abs(signed_delta)
        jax_status = np.asarray(jax_result.status)
        jax_messages = jax_result.messages()
        jax_converged = np.isin(jax_status, (0, 1))
        scipy_converged = np.asarray([r.success for r in scipy_result])

        per_start = []
        for i, scipy_item in enumerate(scipy_result):
            per_start.append({
                "start_index": i,
                "initial_x": starts[i].tolist(),
                "initial_loss": float(initial_score[i]),
                "jax_x": np.asarray(jax_result.x[i]).tolist(),
                "jax_loss": float(jscore[i]),
                "jax_status": int(jax_status[i]),
                "jax_message": jax_messages[i],
                "jax_iterations": int(np.asarray(jax_result.iterations)[i]),
                "jax_evaluations": int(np.asarray(jax_result.evaluations)[i]),
                "jax_projected_grad_norm": float(np.asarray(jax_result.projected_grad_norm)[i]),
                "scipy_x": scipy_item.x.tolist(),
                "scipy_loss": float(sscore[i]),
                "scipy_status": int(scipy_item.status),
                "scipy_success": bool(scipy_item.success),
                "scipy_message": str(scipy_item.message),
                "scipy_iterations": int(scipy_item.nit),
                "scipy_evaluations": int(scipy_item.nfev),
                "signed_loss_diff": float(signed_delta[i]),
                "abs_loss_diff": float(delta[i]),
            })

        jax_codes, jax_counts = np.unique(jax_status, return_counts=True)
        scipy_codes, scipy_counts = np.unique(
            np.asarray([r.status for r in scipy_result]), return_counts=True
        )

        row = {
            "dtype": dtype_name,
            "n_starts": n_starts,
            "jax_batch_size": batch_size,
            "compile_s": compile_s,
            "jax_s": jax_s,
            "scipy_s": scipy_s,
            "speedup": scipy_s / jax_s,
            "within_1e-5": float(np.mean(delta <= 1e-5)),
            "jax_no_worse_1e-5": float(np.mean(signed_delta <= 1e-5)),
            "scipy_no_worse_1e-5": float(np.mean(signed_delta >= -1e-5)),
            "median_signed_loss_diff": float(np.median(signed_delta)),
            "max_abs_loss_diff": float(np.max(delta)),
            "jax_converged_fraction": float(np.mean(jax_converged)),
            "scipy_converged_fraction": float(np.mean(scipy_converged)),
            "both_converged_fraction": float(np.mean(jax_converged & scipy_converged)),
            "jax_status_counts": {
                str(int(code)): int(count) for code, count in zip(jax_codes, jax_counts)
            },
            "scipy_status_counts": {
                str(int(code)): int(count) for code, count in zip(scipy_codes, scipy_counts)
            },
            "jax_eval_mean": float(np.mean(np.asarray(jax_result.evaluations))),
            "scipy_eval_mean": float(np.mean([r.nfev for r in scipy_result])),
            "jax_times_s": jax_times,
            "scipy_times_s": scipy_times,
            "per_start": per_start,
        }
        rows.append(row)
        print(
            dtype_name,
            n_starts,
            f"JAX={jax_s:.4f}s SciPy={scipy_s:.4f}s ",
            f"speedup={row['speedup']:.2f}x |loss gap|<=1e-5={row['within_1e-5']:.3f} ",
            f"converged JAX/SciPy={row['jax_converged_fraction']:.3f}/{row['scipy_converged_fraction']:.3f}",
            flush=True,
        )
    return rows, model_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--ftol", type=float, default=1e-8)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument(
        "--jax-batch-size",
        type=int,
        help="process starts in reusable chunks instead of one monolithic batch",
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="trusted demixing_model MirrorAwareMu1Predictor checkpoint",
    )
    parser.add_argument(
        "--deterministic-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use deterministic XLA GPU convolutions; disable only for throughput tests",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dm_public_architecture_scaling_results.json")
    )
    args = parser.parse_args()
    devices = configure_jax(args.platform, args.dtype)
    counts = tuple(int(x) for x in args.counts.split(",") if x)
    rows, model_info = run(
        args.dtype,
        counts=counts,
        repeats=args.repeats,
        maxiter=args.maxiter,
        ftol=args.ftol,
        maxls=args.maxls,
        checkpoint=args.checkpoint,
        jax_batch_size=args.jax_batch_size,
    )
    solver_sha256, solver_sources = solver_source_fingerprint()
    payload = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "requested_platform": args.platform,
        "deterministic_gpu": args.deterministic_gpu,
        "devices": devices,
        "solver_sha256": solver_sha256,
        "solver_sources": solver_sources,
        "benchmark_sha256": file_sha256(Path(__file__)),
        "benchmark_utils_sha256": file_sha256(Path(__file__).with_name("benchmark_utils.py")),
        **model_info,
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
        "ftol": args.ftol,
        "maxls": args.maxls,
        "requested_jax_batch_size": args.jax_batch_size,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print("WROTE", args.output)
