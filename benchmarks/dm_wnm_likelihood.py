from __future__ import annotations

"""Compare batched JAX and SciPy L-BFGS-B on a frozen DM WNM likelihood fit."""

import argparse
import csv
import json
import os
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


set_platform_from_argv("cpu")
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_lbfgsb_wnm_cache"

_early = argparse.ArgumentParser(add_help=False)
_early.add_argument(
    "--dm-root",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "demixing_model",
)
_early.add_argument("--dtype", choices=("float32", "float64"), default="float32")
_known, _ = _early.parse_known_args()
if _known.dtype == "float64":
    os.environ["JAX_ENABLE_X64"] = "1"
DM_ROOT = _known.dm_root.resolve()
sys.path.insert(0, str(DM_ROOT))

import jax
import jax.numpy as jnp

from jax_lbfgsb import BatchedLbfgsb
from model_fit_to_data.continuous_optimizer import dispersed_starts
from model_fit_to_data.likelihood_search import LikelihoodEvaluator
from model_fit_to_data.wnm_scoring import trial_log_density
from shared import surrogate
from shared.prediction import WrappedMixturePredictor, predictor_from_surrogate


DEFAULT_SUBJECT = "ordinary_1_seed0_n450"
DEFAULT_COUNTS = (32,)
DEFAULT_REPEATS = 3
SOURCE_PATHS = (
    "continuous_density/wrapped_mixture_model.py",
    "model_fit_to_data/continuous_optimizer.py",
    "model_fit_to_data/likelihood_search.py",
    "model_fit_to_data/wnm_scoring.py",
    "shared/prediction.py",
    "shared/surrogate.py",
)


def load_subject(path: Path, subject: str):
    trials = []
    metadata = None
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["subject"] != subject:
                continue
            trials.append((float(row["abs_td_dist"]), float(row["bias_to_distr_corr"])))
            current = {
                "subject": subject,
                "case": row["case"],
                "regime": row["regime"],
                "split": row["split"],
                "response_seed": int(row["response_seed"]),
                "n_trials": int(row["n_trials"]),
                "truth": [
                    float(row["true_sd_feat1"]),
                    float(row["true_sd_feat2"]),
                    float(row["true_sd_spat"]),
                ],
            }
            if metadata is not None and current != metadata:
                raise ValueError(f"inconsistent metadata for {subject!r} in {path}")
            metadata = current
    if not trials:
        raise ValueError(f"subject {subject!r} not found in {path}")
    if len(trials) != metadata["n_trials"]:
        raise ValueError(
            f"subject {subject!r} has {len(trials)} rows, expected {metadata['n_trials']}"
        )
    return np.asarray(trials, dtype=np.float32), metadata


def scipy_minimize_each(evaluator, payload, starts, bounds, args):
    feature_difference, bias = payload

    def objective(log_parameters):
        value, gradient = evaluator.value_and_grad(
            jnp.asarray(log_parameters, args.jax_dtype), feature_difference, bias
        )
        value.block_until_ready()
        return float(value), np.asarray(gradient, dtype=args.np_dtype)

    return [
        minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxcor": 10,
                "ftol": args.ftol,
                "gtol": args.gtol,
                "maxiter": args.maxiter,
                "maxfun": args.maxfun,
                "maxls": args.maxls,
            },
        )
        for start in starts
    ]


def add_float64_rescore(loaded, trials, rows):
    """Apply the recovery diagnostic's promoted scorer to all stored endpoints."""
    jax.config.update("jax_enable_x64", True)
    variables = jax.tree.map(
        lambda leaf: jnp.asarray(leaf, jnp.float64), loaded.payload["variables"]
    )
    predictor = WrappedMixturePredictor(
        model=loaded.payload["model"],
        variables=variables,
        n_samples=loaded.n_samples,
        artifact=loaded.path.name,
        meta=loaded.meta,
        sd_motor=0.0,
    )
    feature_difference = jnp.asarray(trials[:, 0], jnp.float64)
    bias = jnp.asarray(trials[:, 1], jnp.float64)

    def loss(parameters):
        return -jnp.sum(
            trial_log_density(
                predictor,
                parameters[0],
                parameters[1],
                parameters[2],
                feature_difference,
                bias,
            )
        )

    scorer = jax.jit(jax.vmap(loss))
    scored_dtype = None
    for row in rows:
        jax_parameters = jnp.asarray(
            [item["jax_parameters"] for item in row["per_start"]], jnp.float64
        )
        scipy_parameters = jnp.asarray(
            [item["scipy_parameters"] for item in row["per_start"]], jnp.float64
        )
        jax_values = scorer(jax_parameters)
        scipy_values = scorer(scipy_parameters)
        scored_dtype = str(jax_values.dtype)
        jax_loss = np.asarray(jax_values)
        scipy_loss = np.asarray(scipy_values)
        difference = jax_loss - scipy_loss
        absolute = np.abs(difference)
        jax_winner = int(np.argmin(jax_loss))
        scipy_winner = int(np.argmin(scipy_loss))
        row["float64_rescore"] = {
            "within_1e-5": float(np.mean(absolute <= 1e-5)),
            "within_1e-3": float(np.mean(absolute <= 1e-3)),
            "jax_better_beyond_1e-3": int(np.sum(difference < -1e-3)),
            "close_within_1e-3": int(np.sum(absolute <= 1e-3)),
            "scipy_better_beyond_1e-3": int(np.sum(difference > 1e-3)),
            "median_signed_loss_difference": float(np.median(difference)),
            "max_absolute_loss_difference": float(np.max(absolute)),
            "jax_winner_index": jax_winner,
            "scipy_winner_index": scipy_winner,
            "jax_best_loss": float(jax_loss[jax_winner]),
            "scipy_best_loss": float(scipy_loss[scipy_winner]),
            "best_loss_difference": float(
                jax_loss[jax_winner] - scipy_loss[scipy_winner]
            ),
        }
        for index, item in enumerate(row["per_start"]):
            item["jax_loss_float64"] = float(jax_loss[index])
            item["scipy_loss_float64"] = float(scipy_loss[index])
            item["signed_loss_difference_float64"] = float(difference[index])
        print(
            f"float64 rescore {row['n_starts']}: "
            f"<=1e-3={row['float64_rescore']['within_1e-3']:.3f} "
            f"best_gap={row['float64_rescore']['best_loss_difference']:.6g}",
            flush=True,
        )
    if scored_dtype != "float64":
        raise RuntimeError(f"float64 rescore produced {scored_dtype}")
    return {"enabled": True, "dtype": scored_dtype}


def run(args):
    loaded = surrogate.load_surrogate(checkpoint_path=args.checkpoint)
    if args.dtype == "float32":
        predictor = predictor_from_surrogate(loaded)
    else:
        variables = jax.tree.map(
            lambda leaf: jnp.asarray(leaf, jnp.float64), loaded.payload["variables"]
        )
        predictor = WrappedMixturePredictor(
            model=loaded.payload["model"],
            variables=variables,
            n_samples=loaded.n_samples,
            artifact=loaded.path.name,
            meta=loaded.meta,
            sd_motor=0.0,
        )
    evaluator = LikelihoodEvaluator(predictor)
    trials, subject_info = load_subject(args.input, args.subject)
    payload = evaluator.trials(trials)
    domain = surrogate.search_bounds(predictor.domain)
    natural_bounds = np.asarray(
        [domain["sd_feat"], domain["sd_feat"], domain["sd_spat"]],
        dtype=np.float64,
    )
    lower, upper = np.log(natural_bounds).T.astype(args.np_dtype)
    bounds = list(zip(lower, upper))
    scorer = jax.jit(jax.vmap(evaluator.loss_fn, in_axes=(0, None, None)))
    rows = []

    for n_starts in args.counts:
        starts = np.log(dispersed_starts(natural_bounds, n_starts, args.seed)).astype(
            args.np_dtype
        )
        batch_size = min(args.jax_batch_size or n_starts, n_starts)
        solver = BatchedLbfgsb(
            evaluator.loss_fn,
            lower,
            upper,
            maxcor=10,
            ftol=args.ftol,
            gtol=args.gtol,
            maxiter=args.maxiter,
            maxfun=args.maxfun,
            maxls=args.maxls,
        )
        compile_started = time.perf_counter()
        solver.compile(jnp.asarray(starts[:batch_size]), *payload)
        compile_seconds = time.perf_counter() - compile_started

        warm = run_jax_chunks(solver, starts, batch_size, *payload)
        jax.block_until_ready(warm.fun)

        def jax_run():
            result = run_jax_chunks(solver, starts, batch_size, *payload)
            jax.block_until_ready(result.fun)
            return result

        jax_seconds, jax_times, jax_result = median_time(jax_run, args.repeats)

        scipy_times = []
        scipy_result = None
        for repeat in range(args.repeats):
            started = time.perf_counter()
            current = scipy_minimize_each(evaluator, payload, starts, bounds, args)
            scipy_times.append(time.perf_counter() - started)
            if repeat == 0:
                scipy_result = current
        scipy_seconds = float(np.median(scipy_times))

        initial_loss = np.asarray(scorer(jnp.asarray(starts), *payload))
        jax_loss = np.asarray(scorer(jax_result.x, *payload))
        scipy_x = np.stack([item.x for item in scipy_result]).astype(args.np_dtype)
        scipy_loss = np.asarray(scorer(jnp.asarray(scipy_x), *payload))
        signed_difference = jax_loss - scipy_loss
        absolute_difference = np.abs(signed_difference)
        jax_status = np.asarray(jax_result.status)
        jax_converged = np.isin(jax_status, (0, 1))
        scipy_converged = np.asarray([item.success for item in scipy_result])
        jax_winner = int(np.argmin(jax_loss))
        scipy_winner = int(np.argmin(scipy_loss))

        messages = jax_result.messages()
        per_start = []
        for index, item in enumerate(scipy_result):
            per_start.append(
                {
                    "start_index": index,
                    "initial_parameters": np.exp(starts[index]).tolist(),
                    "initial_loss": float(initial_loss[index]),
                    "jax_parameters": np.exp(np.asarray(jax_result.x[index])).tolist(),
                    "jax_loss": float(jax_loss[index]),
                    "jax_status": int(jax_status[index]),
                    "jax_message": messages[index],
                    "jax_iterations": int(np.asarray(jax_result.iterations)[index]),
                    "jax_evaluations": int(np.asarray(jax_result.evaluations)[index]),
                    "scipy_parameters": np.exp(item.x).tolist(),
                    "scipy_loss": float(scipy_loss[index]),
                    "scipy_status": int(item.status),
                    "scipy_success": bool(item.success),
                    "scipy_message": str(item.message),
                    "scipy_iterations": int(item.nit),
                    "scipy_evaluations": int(item.nfev),
                    "signed_loss_difference": float(signed_difference[index]),
                    "absolute_loss_difference": float(absolute_difference[index]),
                }
            )

        row = {
            "dtype": args.dtype,
            "realized_loss_dtype": str(jax_result.fun.dtype),
            "n_starts": n_starts,
            "jax_batch_size": batch_size,
            "compile_seconds": compile_seconds,
            "jax_seconds": jax_seconds,
            "scipy_seconds": scipy_seconds,
            "speedup": scipy_seconds / jax_seconds,
            "within_1e-5": float(np.mean(absolute_difference <= 1e-5)),
            "within_1e-3": float(np.mean(absolute_difference <= 1e-3)),
            "jax_no_worse_1e-3": float(np.mean(signed_difference <= 1e-3)),
            "scipy_no_worse_1e-3": float(np.mean(signed_difference >= -1e-3)),
            "median_signed_loss_difference": float(np.median(signed_difference)),
            "max_absolute_loss_difference": float(np.max(absolute_difference)),
            "jax_converged_fraction": float(np.mean(jax_converged)),
            "scipy_converged_fraction": float(np.mean(scipy_converged)),
            "both_converged_fraction": float(np.mean(jax_converged & scipy_converged)),
            "jax_evaluations": int(np.sum(np.asarray(jax_result.evaluations))),
            "scipy_evaluations": int(sum(item.nfev for item in scipy_result)),
            "jax_winner_index": jax_winner,
            "scipy_winner_index": scipy_winner,
            "same_winner": jax_winner == scipy_winner,
            "jax_best_loss": float(jax_loss[jax_winner]),
            "scipy_best_loss": float(scipy_loss[scipy_winner]),
            "best_loss_difference": float(
                jax_loss[jax_winner] - scipy_loss[scipy_winner]
            ),
            "jax_best_parameters": np.exp(np.asarray(jax_result.x[jax_winner])).tolist(),
            "scipy_best_parameters": np.exp(scipy_result[scipy_winner].x).tolist(),
            "jax_times_seconds": jax_times,
            "scipy_times_seconds": scipy_times,
            "per_start": per_start,
        }
        rows.append(row)
        print(
            f"{args.dtype} {n_starts} JAX={jax_seconds:.4f}s "
            f"SciPy={scipy_seconds:.4f}s "
            f"speedup={row['speedup']:.2f}x <=1e-3={row['within_1e-3']:.3f} "
            f"best_gap={row['best_loss_difference']:.6g} "
            f"converged={row['jax_converged_fraction']:.3f}/"
            f"{row['scipy_converged_fraction']:.3f}",
            flush=True,
        )

    if args.dtype == "float64":
        rescore = {"enabled": False, "reason": "search already uses promoted float64 WNM"}
    elif args.float64_rescore:
        rescore = add_float64_rescore(loaded, trials, rows)
    else:
        rescore = {"enabled": False}
    return rows, subject_info, natural_bounds, rescore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dm-root", type=Path, default=DM_ROOT)
    parser.add_argument(
        "--input",
        type=Path,
        default=DM_ROOT.parent
        / "results/continuous_density_4.1q/recovery/single_condition_n100/"
        "surface_baseline_input.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DM_ROOT / "pretrained/current_wnm_k12_100samples.pkl",
    )
    parser.add_argument("--subject", default=DEFAULT_SUBJECT)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--counts", default=",".join(map(str, DEFAULT_COUNTS)))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jax-batch-size", type=int)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--maxfun", type=int, default=15000)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument("--ftol", type=float, default=1e-9)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument(
        "--float64-rescore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rescore float32 search endpoints with the recovery diagnostic's promoted WNM",
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("validation/dm_wnm_likelihood.json")
    )
    args = parser.parse_args(argv)
    args.dm_root = args.dm_root.resolve()
    args.input = args.input.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.dm_root != DM_ROOT:
        parser.error("--dm-root must be fixed before Python imports; pass it on the CLI")
    if args.dtype != _known.dtype:
        parser.error("--dtype must be fixed before Python imports; pass it on the CLI")
    args.counts = tuple(int(value) for value in args.counts.split(",") if value)
    args.jax_dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    args.np_dtype = np.float32 if args.dtype == "float32" else np.float64
    devices = configure_jax(args.platform, args.dtype)
    rows, subject_info, natural_bounds, rescore = run(args)
    solver_sha256, solver_sources = solver_source_fingerprint()
    source_hashes = {
        relative: file_sha256(args.dm_root / relative) for relative in SOURCE_PATHS
    }
    payload = {
        "protocol": "dm_wnm_likelihood_lbfgsb_parity_v2",
        "python": platform.python_version(),
        "jax": jax.__version__,
        "scipy": scipy.__version__,
        "dtype": args.dtype,
        "float64_rescore": rescore,
        "requested_platform": args.platform,
        "devices": devices,
        "objective": "WNM continuous point negative log-likelihood",
        "search": "log-space L-BFGS-B with deterministic Latin-hypercube starts",
        "subject": subject_info,
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "natural_bounds": natural_bounds.tolist(),
        "seed": args.seed,
        "repeats": args.repeats,
        "maxiter": args.maxiter,
        "maxfun": args.maxfun,
        "maxls": args.maxls,
        "ftol": args.ftol,
        "gtol": args.gtol,
        "requested_jax_batch_size": args.jax_batch_size,
        "solver_sha256": solver_sha256,
        "solver_sources": solver_sources,
        "benchmark_sha256": file_sha256(Path(__file__)),
        "benchmark_utils_sha256": file_sha256(Path(__file__).with_name("benchmark_utils.py")),
        "dm_source_sha256": source_hashes,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
