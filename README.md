# JAX L-BFGS-B port

Standalone, dimension-agnostic JAX implementation of **L-BFGS-B**, designed to optimize many independent starts as one `vmap`/`jit` accelerator batch. It does not replace L-BFGS-B with clipped L-BFGS, a parameter transform, or a barrier objective.

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

## Install and CPU tests

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e '.[test]'
pytest -q
```

Validation includes the native test suite plus cases adapted from Optim.jl's
L-BFGS-B tests on both CPU and GPU.

## API

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
```

`result` preserves start order and contains initial/final points and losses, status, iteration/evaluation counts, projected-gradient norm, final bound activity, and boundary-hit diagnostics. `result.winner_index()` chooses the finite minimum loss, breaking exact ties by start order.

## Hierarchical objectives / DM shape

The solver is agnostic to a `2*C + 1` vector such as

```text
[sd_feat1_1, sd_feat2_1, ..., sd_feat1_C, sd_feat2_C, sd_spat_shared]
```

Put condition logic and masks in the objective payload and use a condition-mean loss; autodiff then accumulates all condition contributions into the shared parameter. The test suite includes a synthetic hierarchical recovery case.

The core package does not depend on `demixing_model`. A separate benchmark
reproduces the public DM surface-network architecture and density objective as a
workload proxy. It can optionally load a trusted `MirrorAwareMu1Predictor`
checkpoint. Empirical fitted-model parity still needs to be run in the DM
environment.

## Validation and benchmarks

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

### Actual DM WNM likelihood

The integration benchmark uses the packaged K12 WNM, a frozen actual-DM
recovery dataset, the WNM point-likelihood evaluator, its artifact bounds, and
the selected 32 deterministic log-space Latin-hypercube starts:

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
