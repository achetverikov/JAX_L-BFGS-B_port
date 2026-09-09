# Validation status

The task requires genuine L-BFGS-B structure, per-start SciPy comparison, hierarchical support, and explicit separation of CPU/GPU/DM validation. GPU and WNM/DM validation are not claimed here.

## Repository parity audit (2026-09-09)

An external validation run exposed a repository-state error: the solver committed to `main` was a compressed earlier implementation, while the CPU parity numbers previously reported in this document came from a fuller local development implementation. The stale committed solver used a simplified strong-Wolfe bracket instead of the audited More-Thuente `dcsrch/dcstep` state machine. It consequently produced widespread `ABNORMAL TERMINATION IN LINE SEARCH` failures, often before the first accepted iteration, on the portable parity panel.

The audit replaced that stale implementation and checked the solver against SciPy's current L-BFGS-B control flow. In addition to restoring the proper More-Thuente line search and first-iteration `stpmx/stp` rules, three semantic defects in the fuller development implementation were corrected:

- More-Thuente trial steps are clipped to the global `[stpmin, stpmax]` limits, not to the evolving internal bracket.
- A line-search failure with non-empty L-BFGS memory restores the old point, clears curvature history, and retries the iteration, matching reference L-BFGS-B behavior; only failure with empty history is terminal (subject to evaluation/nonfinite limits).
- The curvature-update gate uses `y^T s > eps * (-g_old^T s)`, rather than the incorrect `eps * y^T y` scale.

Two exact starts from the external validation archive that failed at iteration 0 under the stale solver are now permanent regression tests: one bounded Rosenbrock start and one scaled five-dimensional quadratic start.

## Audited CPU results

Local audit environment: Python 3.13.5, JAX/JAXlib 0.9.0.1, SciPy 1.17.0, NumPy 2.3.5. No CUDA device was available.

The original CPU unit suite plus the two new external-validation regressions passed:

```text
13 passed
```

The portable 32-start x 8-problem parity panel on the audited implementation produced:

- float64: **256/256 starts within `1e-5`** canonical rescored loss of SciPy; largest absolute loss difference `2.84e-13`.
- float32: **255/256 starts within `1e-5`**. The only miss is the intentionally near-flat quadratic where JAX reaches loss `1.397e-13` while SciPy stops at `6.093e-05` on its relative-function-reduction criterion; this is not a worse JAX solution.

Additional audit stress tests covered 144 random coupled positive-definite quadratics with mixtures of two-sided, one-sided, unbounded, and fixed coordinates: **0 failures at `1e-7` loss parity**. Bounded Rosenbrock tests at dimensions 2, 4, 8, and 12 had maximum JAX-vs-SciPy loss differences of approximately `2.6e-20`, `4.4e-16`, `6.8e-13`, and `4.3e-10`, respectively.

## Optim.jl native L-BFGS-B regression panel

Relevant cases from the native L-BFGS-B test suite in JuliaNLSolvers/Optim.jl were adapted to this package. The earlier CPU run passed **13/13** adapted cases. They cover active-bound solutions, bounded Rosenbrock, one-sided and fixed bounds, high-dimensional active sets with strict feasibility, starts directly on a boundary, memory lengths `m=1,2,4,20`, float32 active sets, iteration-limit semantics, the three-problem Fortran-reference cross-check panel, and Optim.jl's dedicated regression for subspace backtracking measured from the generalized Cauchy point.

During the audit, the most sensitive Optim.jl-derived cases were rerun after the solver corrections: Rosenbrock converged for `m=1,2,4,20`, and the subspace-backtracking regression reached loss `0.12234569775490674` in 32 evaluations (well inside the `<=55` guard).

The Optim.jl cases that are interface-specific rather than algorithmic (callbacks/tracing), alternate line-search implementations not exposed by this package, and BigFloat support were not ported.

## CPU scaling

The earlier lightweight CPU scaling results remain useful as batching-overhead measurements, but they should be rerun after this solver correction before being treated as final performance numbers. See `validation/SCALING_CPU.md` and `benchmarks/scaling_panel.py`.

The public-DM-architecture benchmark is a compute-shape proxy only. It uses the public network architecture with deterministic random weights because the trained checkpoint binary was not materialized in the clean audit runtime. See `validation/DM_PUBLIC_ARCHITECTURE.md`.

## Still required

Run float32 GPU tests and the portable parity panel on the corrected `main`; inspect every discrepancy. Then, on the DM machine, run identical 32-start WNM comparisons for `current_kde_ccc`, `binned_sign_ccc`, `kernel_sign_ccc`, `matched_kde_ccc`, and `sign_likelihood`, rescoring both methods on matched dtype/device and stratifying by target, trial count, generating regime, and feature dissimilarity. Do not claim WNM parity or production readiness before those checks.
