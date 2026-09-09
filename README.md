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

CPU validation includes the native test suite plus cases adapted from Optim.jl's L-BFGS-B tests. GPU validation remains pending.

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

The core package does not depend on `demixing_model`. A separate benchmark reproduces the public DM/WNM network architecture and density objective as a workload proxy. Real fitted-checkpoint / empirical WNM parity still needs to be run in the DM environment.

## Validation and benchmarks

### Per-start SciPy parity

```bash
PYTHONPATH=src python benchmarks/parity_panel.py --dtype float64 --out validation_output/cpu64
PYTHONPATH=src python benchmarks/parity_panel.py --dtype float32 --out validation_output/cpu32

# CUDA machine, with a normal JAX CUDA installation:
PYTHONPATH=src python benchmarks/parity_panel.py --dtype float32 --out validation_output/gpu32
```

The parity panel records every start, final parameters, canonical rescored losses, statuses, iteration/evaluation counts, compile time, and steady-state runtime. Any loss discrepancy above `1e-5` should be inspected individually rather than summarized away.

### CPU multi-start scaling

```bash
python benchmarks/scaling_panel.py --dtype float64 --out validation/scaling_cpu64.json
python benchmarks/scaling_panel.py --dtype float32 --out validation/scaling_cpu32.json
```

This measures steady-state JAX vs sequential SciPy at `1, 4, 8, 16, 32, 64, 128` starts on coupled quadratic and Rosenbrock objectives. See `validation/SCALING_CPU.md` for the current CPU measurements.

### Public DM architecture workload

```bash
python benchmarks/dm_public_architecture_scaling.py \
  --dtype float32 \
  --counts 1,4,8,16,32,64,128 \
  --repeats 3 \
  --maxiter 120 \
  --output dm_public_architecture_scaling_results.json
```

This benchmark mirrors the public `demixing_model` predictor architecture, the periodic `180 x 90` surface shape, density-asymmetry collapse, hierarchical 9-parameter layout for four conditions, and `1 - CCC` loss. It uses deterministic random network weights because the trained checkpoint binary was not materialized in the development runtime; therefore it is a **compute/scaling benchmark, not a WNM recovery or fitted-model parity test**. See `validation/DM_PUBLIC_ARCHITECTURE.md`.

## Current validation boundary

CPU tests, SciPy parity panels, and CPU scaling benchmarks are available here. GPU execution and real WNM checkpoint/data recovery remain the outstanding validation steps. Passing toy/public workloads is not a production-readiness claim.

See `VALIDATION.md` for the current validation summary.
