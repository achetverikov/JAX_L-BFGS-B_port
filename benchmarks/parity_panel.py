"""Portable per-start SciPy parity panel for the JAX L-BFGS-B port."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from benchmark_utils import (
    configure_jax,
    file_sha256,
    set_platform_from_argv,
    solver_source_fingerprint,
)

set_platform_from_argv("cpu")

import jax
import jax.numpy as jnp
from jax_lbfgsb import BatchedLbfgsb


def qproblem(name, A, target, lo, hi, dtype):
    A = np.asarray(A, dtype=dtype)
    target = np.asarray(target, dtype=dtype)
    lo = np.asarray(lo, dtype=dtype)
    hi = np.asarray(hi, dtype=dtype)
    Aj, tj = jnp.asarray(A), jnp.asarray(target)

    def jfun(x):
        d = x - tj
        return 0.5 * d @ Aj @ d

    def nfun(x):
        x = np.asarray(x, dtype=dtype)
        d = x - target
        return float(np.asarray(0.5 * d @ A @ d, dtype=dtype))

    def ngrad(x):
        x = np.asarray(x, dtype=dtype)
        return np.asarray(A @ (x - target), dtype=np.float64)

    return name, jfun, nfun, ngrad, lo, hi


def problems(dtype):
    out = []
    for n in (2, 5, 9):
        target = np.linspace(-0.5, 0.7, n)
        A = np.diag(np.geomspace(1.0, 100.0, n))
        out.append(
            qproblem(
                f"scaled_quad_{n}",
                A,
                target,
                np.full(n, -1),
                np.full(n, 1),
                dtype,
            )
        )

    A = np.array(
        [
            [5.0, 1.2, 0.4, 0.0, 0.0],
            [1.2, 4.0, 0.8, 0.2, 0.0],
            [0.4, 0.8, 3.0, 0.7, 0.1],
            [0.0, 0.2, 0.7, 2.5, 0.5],
            [0.0, 0.0, 0.1, 0.5, 2.0],
        ]
    )
    out.append(
        qproblem(
            "coupled_5",
            A,
            [0.8, -0.6, 0.3, 0.9, -0.2],
            np.full(5, -0.7),
            np.full(5, 0.7),
            dtype,
        )
    )
    out.append(
        qproblem(
            "face_edge_corner_4",
            np.diag([1.0, 2.0, 3.0, 4.0]),
            [-2.0, 2.0, -3.0, 0.25],
            np.full(4, -1),
            np.full(4, 1),
            dtype,
        )
    )
    out.append(
        qproblem(
            "flat_direction_5",
            np.diag([1e-4, 1e-2, 1.0, 10.0, 100.0]),
            [0.2, -0.3, 0.4, -0.2, 0.1],
            np.full(5, -1),
            np.full(5, 1),
            dtype,
        )
    )
    out.append(
        qproblem(
            "hierarchical_7",
            np.diag([0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 6.0]),
            [0.4, 0.8, 0.6, 1.1, 0.9, 0.5, 0.7],
            np.full(7, 0.1),
            np.full(7, 1.5),
            dtype,
        )
    )

    def jros(x):
        return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def nros(x):
        x = np.asarray(x, dtype=dtype)
        return float(
            np.asarray(
                np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2),
                dtype=dtype,
            )
        )

    def ngros(x):
        x = np.asarray(x, dtype=dtype)
        g = np.zeros_like(x)
        t = x[1:] - x[:-1] ** 2
        g[:-1] += -400 * x[:-1] * t - 2 * (1 - x[:-1])
        g[1:] += 200 * t
        return np.asarray(g, dtype=np.float64)

    out.append(
        (
            "rosenbrock_2",
            jros,
            nros,
            ngros,
            np.full(2, -1.5, dtype=dtype),
            np.full(2, 1.5, dtype=dtype),
        )
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation_output/cpu64")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--starts", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="cpu")
    ap.add_argument("--ftol", type=float, default=2.220446049250313e-9)
    ap.add_argument("--maxls", type=int, default=20)
    args = ap.parse_args()

    devices = configure_jax(args.platform, args.dtype)
    dtype = np.float64 if args.dtype == "float64" else np.float32
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260909)
    solver_sha256, solver_sources = solver_source_fingerprint()
    metadata = {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "dtype": args.dtype,
        "requested_platform": args.platform,
        "devices": devices,
        "seed": 20260909,
        "solver_sha256": solver_sha256,
        "solver_sources": solver_sources,
        "benchmark_sha256": file_sha256(Path(__file__)),
        "benchmark_utils_sha256": file_sha256(Path(__file__).with_name("benchmark_utils.py")),
        "ftol": args.ftol,
        "maxls": args.maxls,
        "cuda_visible": any(d.platform == "gpu" for d in jax.devices()),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    summary = []
    jsonl_path = outdir / "per_start.jsonl"
    jsonl_path.write_text("")

    for name, jfun, nfun, ngrad, lo, hi in problems(dtype):
        starts = rng.uniform(lo, hi, size=(args.starts, lo.size)).astype(dtype)
        solver = BatchedLbfgsb(jfun, lo, hi, ftol=args.ftol, maxls=args.maxls)
        t0 = time.perf_counter()
        solver.compile(jnp.asarray(starts))
        compile_s = time.perf_counter() - t0

        jr = solver.run(jnp.asarray(starts))
        jax.block_until_ready(jr.fun)
        jtimes = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            jr = solver.run(jnp.asarray(starts))
            jax.block_until_ready(jr.fun)
            jtimes.append(time.perf_counter() - t0)
        jax_s = float(np.median(jtimes))

        rows = []
        stimes = []
        for rep in range(args.repeats):
            rep_rows = []
            t0 = time.perf_counter()
            for x0 in starts:
                r = minimize(
                    nfun,
                    x0,
                    jac=ngrad,
                    method="L-BFGS-B",
                    bounds=list(zip(lo, hi)),
                    options={"ftol": args.ftol, "maxls": args.maxls},
                )
                rep_rows.append((r.x, r.fun, r.status, r.nit, r.nfev, str(r.message)))
            stimes.append(time.perf_counter() - t0)
            if rep == 0:
                rows = rep_rows
        scipy_s = float(np.median(stimes))

        sx = np.stack([r[0] for r in rows])
        sf = np.array([r[1] for r in rows])
        jx = np.asarray(jr.x)
        jf = np.asarray(jr.fun)
        jscore = np.array([nfun(np.asarray(x, dtype=dtype)) for x in jx])
        sscore = np.array([nfun(np.asarray(x, dtype=dtype)) for x in sx])
        diff = jscore - sscore

        np.savez(
            outdir / f"{name}.npz",
            starts=starts,
            jax_x=jx,
            scipy_x=sx,
            jax_fun=jf,
            scipy_fun=sf,
            jax_rescore=jscore,
            scipy_rescore=sscore,
            jax_status=np.asarray(jr.status),
            jax_nit=np.asarray(jr.iterations),
            jax_nfev=np.asarray(jr.evaluations),
            scipy_status=np.array([r[2] for r in rows]),
            scipy_nit=np.array([r[3] for r in rows]),
            scipy_nfev=np.array([r[4] for r in rows]),
            scipy_message=np.array([r[5] for r in rows]),
        )
        bad = np.flatnonzero(np.abs(diff) > 1e-5)
        with jsonl_path.open("a") as fh:
            for i in range(args.starts):
                row = {
                    "problem": name,
                    "start_index": i,
                    "start": starts[i].tolist(),
                    "jax_x": jx[i].tolist(),
                    "scipy_x": sx[i].tolist(),
                    "jax_rescore": float(jscore[i]),
                    "scipy_rescore": float(sscore[i]),
                    "loss_diff": float(diff[i]),
                    "jax_status": int(np.asarray(jr.status)[i]),
                    "jax_nit": int(np.asarray(jr.iterations)[i]),
                    "jax_nfev": int(np.asarray(jr.evaluations)[i]),
                    "scipy_status": int(rows[i][2]),
                    "scipy_nit": int(rows[i][3]),
                    "scipy_nfev": int(rows[i][4]),
                    "scipy_message": rows[i][5],
                }
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")

        item = {
            "problem": name,
            "compile_s": compile_s,
            "jax_s": jax_s,
            "scipy_s": scipy_s,
            "speedup": scipy_s / jax_s,
            "within_1e-5": float(np.mean(np.abs(diff) <= 1e-5)),
            "max_abs_loss_diff": float(np.max(np.abs(diff))),
            "miss_indices": bad.tolist(),
        }
        summary.append(item)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
