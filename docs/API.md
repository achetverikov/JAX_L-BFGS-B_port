# API reference

Import the public API from `jax_lbfgsb`. The implementation modules are internal.

## Objective and inputs

`BatchedLbfgsb(loss_fn, lower, upper, **options)` minimizes a differentiable
scalar objective under box constraints. The objective signature is
`loss_fn(parameters, *payload)`, with `parameters` shaped `(n_parameters,)`.
Use JAX operations so `jax.value_and_grad` can differentiate the objective.
It must return a scalar array, rather than a vector of per-observation losses.

`lower` and `upper` are one-dimensional arrays of identical length, with
`lower <= upper`. Use `-jnp.inf` or `jnp.inf` for unbounded coordinates;
equal bounds fix a coordinate. All starts share the same bounds and objective.

`run(starts, *payload)` accepts floating-point starts shaped
`(n_starts, n_parameters)`, with at least one start. Initial points outside
the box are clipped before evaluation. Every payload argument is shared across
starts (`vmap` uses `in_axes=None`); a leading payload dimension does not assign
different data to individual starts. Array pytrees such as dictionaries can
carry shared data and masks. Reduce data or condition axes inside the objective.

The solver computes gradients internally. There is no custom `jac`, callback,
SciPy-style `OptimizeResult`, or differentiation API through the optimization
trajectory. The solver's dynamic `lax.while_loop` does not support reverse-mode
autodiff through the solve.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `maxcor` | `10` | Maximum stored curvature pairs per start. |
| `ftol` | `2.220446049250313e-9` | Relative objective reduction threshold. |
| `gtol` | `1e-5` | Projected-gradient infinity-norm threshold. |
| `maxiter` | `15000` | Maximum accepted iterations per start. |
| `maxfun` | `15000` | Maximum value/gradient evaluations per start, including the initial evaluation. |
| `maxls` | `20` | Maximum trials in each line-search attempt. |

The function reduction test is
`(f_old - f_new) / max(abs(f_old), abs(f_new), 1) <= ftol`, provided the
reduction is nonnegative. `ftol` is the direct threshold used by SciPy's
`minimize` interface, rather than its legacy `factr` parameter.
Projected-gradient convergence is evaluated before function-reduction and
budget tests after an accepted step. A failed line search can clear nonempty
curvature history and retry; these trials still consume the evaluation budget.

Configure a new solver when changing options or bounds: these are captured
when its compiled function is built.

## Compilation and execution

`compile(starts, *payload)` optionally compiles without running optimization
and returns the solver. `run` otherwise compiles on first use. Reusing shapes,
dtypes and payload structure reuses compilation; changing the number of starts
or parameters can trigger another compilation. Payload values can change
without recompiling when their shapes and dtypes stay the same.

The start dtype controls solver arithmetic. JAX normally uses float32;
to use float64, call `jax.config.update("jax_enable_x64", True)` before
constructing arrays and pass float64 starts and payload arrays. Very tight
tolerances may be ineffective in float32. Precision can change optimization
paths and termination, especially for nearly flat objectives.

JAX dispatch is asynchronous. For timing, compile first, run a warm-up, and call
`result.fun.block_until_ready()` before stopping the timer. This separates
compilation and device execution time.

## Results

`run` returns a `BatchedLbfgsbResult` named tuple of JAX arrays. Start order is
preserved. Below, `B` is the number of starts and `N` the number of parameters.

| Field | Shape | Meaning |
| --- | --- | --- |
| `initial_x` | `(B, N)` | Initial points after clipping to bounds. |
| `x` | `(B, N)` | Final accepted points. |
| `initial_fun`, `fun` | `(B,)` | Initial and final objective values. |
| `status` | `(B,)` | Termination codes listed below. |
| `iterations` | `(B,)` | Accepted iterations. |
| `evaluations` | `(B,)` | Value/gradient evaluations. |
| `projected_grad_norm` | `(B,)` | Final projected-gradient infinity norm. |
| `active_lower`, `active_upper` | `(B, N)` | Exact equality to finite bounds at the final point. |
| `ever_hit_lower`, `ever_hit_upper` | `(B, N)` | Bound activity at the clipped initial point or any accepted iterate. |

`messages()` returns a host list of termination messages. `winner_index()`
returns the index of the smallest finite final loss, breaking exact ties by
start order. It does **not** filter by convergence status. If every loss is
nonfinite it returns `0`; check finiteness before accepting a winner.

| Code | Exported constant | Interpretation |
| --- | --- | --- |
| `0` | `CONVERGED_PGTOL` | Projected-gradient tolerance met. |
| `1` | `CONVERGED_FTOL` | Function-reduction tolerance met. |
| `2` | `ITERATION_LIMIT` | Iteration budget exhausted. |
| `3` | `EVALUATION_LIMIT` | Evaluation budget exhausted. |
| `4` | `LINE_SEARCH_FAILED` | Line search failed after any history restart. |
| `5` | `NONFINITE` | Nonfinite objective, gradient, or compact-model computation. |

Inspect status and projected gradient as well as loss. A finite endpoint can
still have terminated at a budget limit or failed line search. For nonconvex
objectives, successful termination establishes a local stopping condition,
not a global optimum.

`STATUS_MESSAGES` maps integer codes to messages. The public helpers
`projected_gradient(x, g, lower, upper)` and
`projected_gradient_norm(x, g, lower, upper)` operate on a single point;
use `jax.vmap` for batched diagnostics.
