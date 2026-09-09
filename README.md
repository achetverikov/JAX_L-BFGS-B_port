# JAX L-BFGS-B port

Standalone, dimension-agnostic JAX implementation of **L-BFGS-B**, designed to optimize many independent starts as one `vmap`/`jit` accelerator batch. It does not replace L-BFGS-B with clipped L-BFGS, a parameter transform, or a barrier objective.

## Algorithm implemented

Each start has independent state and follows the L-BFGS-B structure: projected-gradient convergence; generalized Cauchy point along the piecewise projected steepest-descent path; active/free identification; free-subspace Newton minimization using the limited-memory compact representation; bound-aware More-Thuente line search; curvature-gated memory updates; and per-start termination/diagnostics.

The generalized Cauchy implementation recomputes compact Hessian-vector products at breakpoints instead of using the Fortran incremental recurrence. This is shape-static for JAX and may differ in floating-point ordering, so differences are exposed to parity testing rather than hidden.

## Install and CPU tests

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e '.[test]'
pytest -q
```

CPU validation used Python 3.13.5, JAX/JAXlib 0.9.0.1, SciPy 1.17.0, NumPy 2.3.5. GPU validation remains pending.

## API

```python
import jax.numpy as jnp
from jax_lbfgsb import BatchedLbfgsb

def loss_fn(parameters, target):
    return jnp.sum((parameters - target) ** 2)

search = BatchedLbfgsb(loss_fn, lower=[-2., -2.], upper=[2., 2.])
starts = jnp.array([[-1., 1.], [1.5, -1.5]])
target = jnp.array([0.2, 0.7])
search.compile(starts, target)
result = search.run(starts, target)
```

`result` preserves start order and contains initial/final points and losses, status, iteration/evaluation counts, projected-gradient norm, final bound activity, and boundary-hit diagnostics. `result.winner_index()` chooses the finite minimum loss, breaking exact ties by start order.

## Hierarchical objective and DM handoff

The solver is agnostic to a `2*C + 1` vector such as `[sd_feat1_1, sd_feat2_1, ..., sd_feat1_C, sd_feat2_C, sd_spat_shared]`. Put condition logic/masks in the objective payload and use a condition-mean loss; autodiff accumulates all condition contributions into the shared parameter.

No `demixing_model` code is imported. A later adapter can pass `loss_fn(log_parameters, *payload)`, log starts, and log bounds directly. WNM artifacts were unavailable here and must be validated later on the DM machine.

## Portable parity panel

```bash
python benchmarks/parity_panel.py --dtype float64 --out validation_output/cpu64
python benchmarks/parity_panel.py --dtype float32 --out validation_output/cpu32
# CUDA machine:
python benchmarks/parity_panel.py --dtype float32 --out validation_output/gpu32
```

See `VALIDATION.md` for current CPU results and limitations. Passing public/toy objectives is not a production-readiness claim.