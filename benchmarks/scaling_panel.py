"""Steady-state CPU scaling benchmark vs SciPy L-BFGS-B.

Compilation is excluded.  Each timing is the median of repeated full solves from
identical starts.  The panel is intentionally small enough to run on a clean CPU
machine while exposing the batching crossover.
"""
from __future__ import annotations

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
SEED = 20260909


def _median_time(fn, repeats):
    times = []
    value = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), times, value


def _coupled_quadratic(dtype):
    n = 8
    rng = np.random.default_rng(123)
    M = rng.normal(size=(n, n)).astype(dtype)
    A = (M.T @ M + np.diag(np.geomspace(0.2, 20.0, n))).astype(dtype)
    target = np.linspace(-1.2, 1.1, n, dtype=dtype)
    lo = np.full(n, -0.75, dtype=dtype)
    hi = np.full(n, 0.75, dtype=dtype)
    Aj, tj = jnp.asarray(A), jnp.asarray(target)

    def jfun(x):
        d = x - tj
        return 0.5 * d @ Aj @ d

    def nfun(x):
        d = np.asarray(x, dtype=dtype) - target
        return float(np.asarray(0.5 * d @ A @ d, dtype=dtype))

    def ngrad(x):
        return np.asarray(A @ (np.asarray(x, dtype=dtype) - target), dtype=np.float64)

    return "coupled_quad_8", jfun, nfun, ngrad, lo, hi


def _rosenbrock(dtype):
    n = 6
    lo = np.full(n, -1.5, dtype=dtype)
    hi = np.full(n, 1.5, dtype=dtype)

    def jfun(x):
        return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)

    def nfun(x):
        x = np.asarray(x, dtype=dtype)
        return float(np.asarray(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2), dtype=dtype))

    def ngrad(x):
        x = np.asarray(x, dtype=dtype)
        g = np.zeros_like(x)
        t = x[1:] - x[:-1] ** 2
        g[:-1] += -400.0 * x[:-1] * t - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * t
        return np.asarray(g, dtype=np.float64)

    return "rosenbrock_6", jfun, nfun, ngrad, lo, hi


def _run_problem(problem, dtype, repeats):
    name, jfun, nfun, ngrad, lo, hi = problem
    rng = np.random.default_rng(SEED + (0 if name.startswith("coupled") else 1))
    all_starts = rng.uniform(lo, hi, size=(max(COUNTS), lo.size)).astype(dtype)
    rows = []

    for n_starts in COUNTS:
        starts = all_starts[:n_starts].copy()
        solver = BatchedLbfgsb(jfun, lo, hi, maxiter=1000, maxfun=5000)

        t0 = time.perf_counter()
        solver.compile(jnp.asarray(starts))
        compile_s = time.perf_counter() - t0

        warm = solver.run(jnp.asarray(starts))
        jax.block_until_ready(warm.fun)

        def jrun():
            result = solver.run(jnp.asarray(starts))
            jax.block_until_ready(result.fun)
            return result

        jax_s, jax_times, jr = _median_time(jrun, repeats)

        bounds = list(zip(lo, hi))
        def srun():
            return [
                minimize(nfun, x0, jac=ngrad, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 1000, "maxfun": 5000})
                for x0 in starts
            ]

        scipy_s, scipy_times, sr = _median_time(srun, repeats)
        jscore = np.asarray([nfun(x) for x in np.asarray(jr.x)])
        sscore = np.asarray([nfun(r.x) for r in sr])
        diff = jscore - sscore

        rows.append({
            "problem": name,
            "n_starts": n_starts,
            "compile_s": compile_s,
            "jax_s": jax_s,
            "scipy_s": scipy_s,
            "speedup": scipy_s / jax_s,
            "jax_us_per_start": 1e6 * jax_s / n_starts,
            "scipy_us_per_start": 1e6 * scipy_s / n_starts,
            "within_1e-5": float(np.mean(np.abs(diff) <= 1e-5)),
            "max_abs_loss_diff": float(np.max(np.abs(diff))),
            "jax_eval_mean": float(np.mean(np.asarray(jr.evaluations))),
            "scipy_eval_mean": float(np.mean([r.nfev for r in sr])),
            "jax_times_s": jax_times,
            "scipy_times_s": scipy_times,
        })
        print(name, n_starts, f"JAX={1e3*jax_s:.2f} ms", f"SciPy={1e3*scipy_s:.2f} ms", f"speedup={scipy_s/jax_s:.2f}x")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="validation/scaling_cpu.json")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = np.float64 if args.dtype == "float64" else np.float32
    rows = []
    for problem in (_coupled_quadratic(dtype), _rosenbrock(dtype)):
        rows.extend(_run_problem(problem, dtype, args.repeats))

    payload = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "device": [str(d) for d in jax.devices()],
        "dtype": args.dtype,
        "repeats": args.repeats,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
