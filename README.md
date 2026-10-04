# JAX L-BFGS-B port

Standalone, dimension-agnostic JAX implementation of **L-BFGS-B**, designed to optimize many independent starts as one `vmap`/`jit` accelerator batch. It does not replace L-BFGS-B with clipped L-BFGS, a parameter transform, or a barrier objective.

Use it for differentiable scalar objectives with box-constrained parameters,
especially multi-start searches where the same objective and data are reused.
Gradients come from JAX autodiff, and each start has its own optimizer history
and stopping status. The core package requires only JAX and NumPy; SciPy is
used for validation. It runs on the devices supported by your JAX installation.

Start with the [quick start](#quick-start), then see the
[API reference](docs/API.md) for input contracts, options, result fields and
termination codes. [VALIDATION.md](VALIDATION.md) records the measured parity
and known limits.

## Algorithm implemented

Each start has independent state and follows the L-BFGS-B structure:

1. projected-gradient convergence test;
2. generalized Cauchy point along the piecewise projected steepest-descent path;
3. active/free identification from that breakpoint walk;
4. free-subspace Newton minimization using the limited-memory compact representation and a small `2m x 2m` solve;
5. bound-aware More-Thuente line search;
6. curvature-gated limited-memory updates;
7. projected-gradient, relative-function-reduction, iteration/evaluation, non-finite, and line-search termination statuses.

The generalized Cauchy implementation recomputes compact Hessian-vector products at breakpoints instead of using the Fortran incremental recurrence. This is shape-static for JAX and may differ in floating-point ordering, so differences are exposed to parity testing rather than hidden.

## Installation

Requires Python 3.11 or newer. From this repository's root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

If you already have a working JAX/CUDA environment, activate it and use the
editable install command directly. The package does not select a CUDA version
or install CUDA extras; use your environment's compatible JAX accelerator
installation.

The solver, quick start and test suite need nothing else. Some benchmarks refer
to **DM**, the [Demixing Model](https://github.com/achetverikov/demixing_model):
a model of attraction and repulsion biases between two remembered items, whose
behavioral fits were the motivating workload for this port. Only the
[DM WNM likelihood benchmark](#dm-wnm-likelihood) requires a DM checkout, which
it expects next to this repository (`../demixing_model`).

## Quick start

```python
import jax.numpy as jnp
from jax_lbfgsb import BatchedLbfgsb


def loss_fn(parameters, target):
    return jnp.sum((parameters - target) ** 2)


search = BatchedLbfgsb(loss_fn, lower=[-2., -2.], upper=[2., 2.])
starts = jnp.array([[-1., 1.], [1.5, -1.5]])
target = jnp.array([0.2, 0.7])

search.compile(starts, target)  # optional: compile for these shapes first
result = search.run(starts, target)

best = result.winner_index()
print(result.x[best])        # approximately [0.2, 0.7]
print(result.fun[best])      # approximately zero
print(result.messages())     # one termination message per start
```

`result` preserves start order and contains initial/final points and losses, status, iteration/evaluation counts, projected-gradient norm, final bound activity, and boundary-hit diagnostics. `result.winner_index()` chooses the finite minimum loss, breaking exact ties by start order.

Check the selected start's status before accepting it: winner selection includes
finite losses from unsuccessful runs too. See the [result reference](docs/API.md#results).

A complete example with an optimum on a bound is available in
[examples/quadratic.py](examples/quadratic.py):

```bash
python examples/quadratic.py
```

It should find approximately `[0.2, 2.0]`, with loss `1.0` and the second upper
bound active, for both starts.

## Batching, precision and performance

- Pass floating-point starts shaped `(n_starts, n_parameters)`. Bounds are
  vectors of length `n_parameters`; out-of-bounds starts are clipped.
- Write a scalar objective with JAX operations. All starts receive the same
  payload arguments; payload arrays are not mapped along their first axis.
- Float32 is JAX's usual default. Enable `jax_enable_x64` before creating arrays
  and supply float64 inputs for higher precision. Choose tolerances appropriate
  to the dtype and objective scale.
- Compilation is specific to shapes and dtypes. Reuse a solver and fixed batch
  shapes to amortize it. Synchronize with `result.fun.block_until_ready()` when
  timing device execution.
- Large batches of expensive objectives can exhaust accelerator memory. Run
  starts in smaller sequential chunks when needed; the last chunk may require
  another compilation if its shape differs. Batched execution is not always
  faster than sequential SciPy, and float64 GPU performance depends on hardware.

The [API reference](docs/API.md) explains stopping thresholds, compilation and
the shared-payload contract in detail.

## Repository map and development

| Path | Contents |
| --- | --- |
| `src/jax_lbfgsb/solver.py` | Public solver and batched optimization loop. |
| `src/jax_lbfgsb/core.py` | Result records, projected gradients, Cauchy point and subspace calculations. |
| `src/jax_lbfgsb/line_search.py` | More-Thuente line-search implementation. |
| `examples/` | Small standalone usage example. |
| `tests/` | Solver, Optim.jl-inspired and regression tests. |
| `benchmarks/` | SciPy parity, scaling and DM integration scripts. |
| `validation/` | Recorded benchmark reports. |

Install test dependencies and run the suite on CPU:

```bash
python -m pip install -e '.[test]'
JAX_PLATFORMS=cpu pytest -q
```

The tests need only this package, pytest and SciPy; they do not import DM.
On a configured CUDA machine, run `JAX_PLATFORMS=cuda pytest -q` separately.
The test configuration enables float64. Validation includes the native test
suite plus cases adapted from Optim.jl's L-BFGS-B tests. Run GPU checks one at
a time, and put temporary benchmark outputs and caches under `/tmp`.

## Hierarchical objectives

The solver is agnostic to parameter structure. DM fits, for example, use a
`2*C + 1` vector for `C` experimental conditions, with two per-condition noise
parameters and one shared one:

```text
[sd_feat1_1, sd_feat2_1, ..., sd_feat1_C, sd_feat2_C, sd_spat_shared]
```

Put condition logic and masks in the objective payload and use a condition-mean loss; autodiff then accumulates all condition contributions into the shared parameter. The test suite includes a synthetic hierarchical recovery case.

The core package does not depend on DM. A separate benchmark reimplements DM's
public surface-network architecture and density objective as a workload proxy,
so it runs without a DM checkout. It can optionally load a trusted
`MirrorAwareMu1Predictor` checkpoint (a pickle, so load only files you trust).
Empirical fitted-model parity still needs to be run in the DM environment.

## Validation and benchmarks

### Parity with SciPy L-BFGS-B

Each benchmark runs every start through both this solver and
`scipy.optimize.minimize(method="L-BFGS-B")` with the same starting point,
bounds and tolerances, then compares the final losses start by start.

| Benchmark | Device | Search dtype | Starts within `1e-5` of SciPy | Largest per-start loss difference | Best loss, JAX − SciPy | Record |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Portable panel: 8 problems × 32 starts | CPU | float64 | 256/256 | `1.7e-13` | n/a | `validation_output/cpu64/` |
| Portable panel: 8 problems × 32 starts | CPU | float32 | 254/256 | `6.1e-05` | n/a | `validation_output/cpu32/` |
| DM WNM likelihood: 1 subject × 32 starts | CPU | float64 | 32/32 | `5e-09` | `0` | `validation/dm_wnm_likelihood_cpu64.json` |
| DM WNM likelihood: 1 subject × 32 starts | GPU | float64 | 32/32 | `2.1e-08` | `1.7e-12` | `validation/dm_wnm_likelihood_gpu64.json` |
| DM WNM likelihood: 1 subject × 32 starts | CPU | float32 | 28/32 | `7.7e-05` | `-9.5e-10` | `validation/dm_wnm_likelihood_cpu32.json` |
| DM WNM likelihood: 1 subject × 32 starts | GPU | float32 | 1/32 | `80` | `-8.1e-05` | `validation/dm_wnm_likelihood_gpu32.json` |

- Loss differences are absolute. Float32 WNM endpoints are compared after
  rescoring in float64; WNM losses are about 937, so `1e-5` is roughly
  `1e-8` relative.
- Both float32 portable-panel misses are on the deliberately near-flat
  quadratic: JAX is `1.6e-05` worse on one start and `6.1e-05` better on the
  other. Per-problem results are in [VALIDATION.md](VALIDATION.md#audited-cpu-results).
- On GPU in float32, individual starts follow different paths than SciPy
  (float32 reduction order differs). JAX ended more than `1e-3` lower on 28
  pairs and SciPy on 1. The best losses still agree to `8.1e-05`, with JAX lower.
  In float64 the paths match.
- The GPU portable panel has no current record: the previously recorded
  `validation_output/gpu32/` was produced by the solver that the 2026-09-09
  audit replaced, and needs a rerun.

### Per-start SciPy parity

```bash
PYTHONPATH=src python benchmarks/parity_panel.py \
  --platform cpu --dtype float64 --out validation_output/cpu64
PYTHONPATH=src python benchmarks/parity_panel.py \
  --platform cpu --dtype float32 --out validation_output/cpu32

# CUDA machine, with a normal JAX CUDA installation:
PYTHONPATH=src python benchmarks/parity_panel.py \
  --platform gpu --dtype float32 --maxls 40 --out validation_output/gpu32
```

The parity panel records every start, final parameters, canonical rescored losses, statuses, iteration/evaluation counts, compile time, and steady-state runtime. Any loss discrepancy above `1e-5` should be inspected individually rather than summarized away.

### CPU multi-start scaling

```bash
PYTHONPATH=src python benchmarks/scaling_panel.py \
  --platform cpu --dtype float64 --out validation/scaling_cpu64.json
PYTHONPATH=src python benchmarks/scaling_panel.py \
  --platform cpu --dtype float32 --out validation/scaling_cpu32.json
```

This measures steady-state JAX vs sequential SciPy at `1, 4, 8, 16, 32, 64, 128` starts on coupled quadratic and Rosenbrock objectives. See `validation/SCALING_CPU.md` for the current CPU measurements.

### Public DM architecture workload

```bash
PYTHONPATH=src python benchmarks/dm_public_architecture_scaling.py \
  --platform gpu \
  --dtype float32 \
  --checkpoint ../demixing_model/pretrained/surface_legacy_epoch1425_10ktrain_20samples.pkl \
  --counts 32 --jax-batch-size 1 \
  --repeats 3 \
  --maxiter 120 --maxls 40 \
  --output dm_public_architecture_scaling_results.json
```

This benchmark mirrors the current `demixing_model` predictor architecture, the
periodic `180 x 90` surface shape, density-asymmetry collapse, hierarchical
9-parameter layout for four conditions, and `1 - CCC` loss. Without
`--checkpoint` it uses deterministic random weights. Even with a trained
checkpoint, its target is synthetic, so this is a **compute/scaling benchmark,
not an empirical recovery or fitted-model parity test**. See
`validation/DM_PUBLIC_ARCHITECTURE.md`.

GPU convolutions are deterministic by default in this benchmark so repeated
validation runs use the same objective. Pass `--no-deterministic-gpu` only when
measuring unconstrained throughput. A short run such as `--counts 1 --repeats 1
--maxiter 20` is a smoke test: its final-loss gap compares two truncated paths
through a nonconvex float32 objective and is not a solver-parity verdict.
For the current DM network, avoid a monolithic 32-start accelerator batch; use
`--jax-batch-size 1` (or benchmark other small chunk sizes) to reuse the compiled
solver without the large batched-gradient temporary allocation.

### DM WNM likelihood

This integration benchmark requires a DM checkout and imports its code. DM's
fitted predictor is a conditional wrapped-normal mixture (WNM): a neural network
that maps noise parameters to a 12-component (K12) mixture density over
response errors. The benchmark uses the packaged K12 WNM, a fixed
parameter-recovery dataset simulated from DM, DM's WNM point-likelihood
evaluator and parameter bounds, and the 32 deterministic log-space
Latin-hypercube starts used for those recovery fits:

```bash
PYTHONPATH=src python benchmarks/dm_wnm_likelihood.py \
  --platform gpu --counts 32 --repeats 3 \
  --output validation/dm_wnm_likelihood_gpu32.json
```

It searches in float32, matching the recovery runs, then promotes the checkpoint
and endpoints for the recovery project's float64 diagnostic rescore. This is
important on GPU: scalar and batched float32 reductions can send the same start
down different paths even when their selected winners are equivalent under the
stable rescore. The input CSV is an external recovery artifact; override
`--input`, `--checkpoint`, or `--dm-root` when the sibling DM checkout is laid
out differently. See `validation/DM_WNM_LIKELIHOOD.md`.

Pass `--dtype float64` to search with the promoted WNM arithmetic as well as
score with it. On the current RTX 5080, the 32-start JAX solve is 9.76x slower
in float64 (1.687 s versus 0.173 s), but all 32 JAX/SciPy pairs converge and the
best losses agree to `1.71e-12`. This is a precision/performance diagnostic;
the selected DM recovery search remains float32.

## Current validation boundary

CPU and GPU tests, portable SciPy parity panels, and one real WNM likelihood
integration case are available here. The WNM check does not yet cover the full
recovery panel or the other objective-specific searches, so it is not by itself
a production-readiness claim.

See `VALIDATION.md` for the current validation summary.

## License

MIT. See [LICENSE](LICENSE).
