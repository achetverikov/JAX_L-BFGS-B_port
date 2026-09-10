# Validation status

The task requires genuine L-BFGS-B structure, per-start SciPy comparison,
hierarchical support, and explicit separation of CPU/GPU/DM validation. Portable
CPU/GPU validation is covered here, along with one real subject-target WNM
likelihood integration case.

## Repository parity audit (2026-09-09)

An external validation run exposed a repository-state error: the solver committed to `main` was a compressed earlier implementation, while the CPU parity numbers previously reported in this document came from a fuller local development implementation. The stale committed solver used a simplified strong-Wolfe bracket instead of the audited More-Thuente `dcsrch/dcstep` state machine. It consequently produced widespread `ABNORMAL TERMINATION IN LINE SEARCH` failures, often before the first accepted iteration, on the portable parity panel.

The audit replaced that stale implementation and checked the solver against SciPy's current L-BFGS-B control flow. In addition to restoring the proper More-Thuente line search and first-iteration `stpmx/stp` rules, three semantic defects in the fuller development implementation were corrected:

- More-Thuente trial steps are clipped to the global `[stpmin, stpmax]` limits, not to the evolving internal bracket.
- A line-search failure with non-empty L-BFGS memory restores the old point, clears curvature history, and retries the iteration, matching reference L-BFGS-B behavior; only failure with empty history is terminal (subject to evaluation/nonfinite limits).
- The curvature-update gate uses `y^T s > eps * (-g_old^T s)`, rather than the incorrect `eps * y^T y` scale.

Two exact starts from the external validation archive that failed at iteration 0 under the stale solver are now permanent regression tests: one bounded Rosenbrock start and one scaled five-dimensional quadratic start.

## Audited CPU results

Local audit environment: Python 3.12.3, JAX/JAXlib 0.10.1.dev20260909,
SciPy 1.17.1, NumPy 2.4.6, with CPU and CUDA devices.

The unit and regression suite, including a non-finite compact-representation
termination check, passed on both CPU and GPU:

```text
22 passed (CPU)
22 passed (GPU)
```

The portable 32-start x 8-problem parity panel on the audited implementation produced:

- float64: **256/256 starts within `1e-5`** canonical rescored loss of
  SciPy; largest absolute loss difference `1.74e-13`.
- float32: **254/256 starts within `1e-5`**. Both misses are on the
  intentionally near-flat quadratic. JAX is worse by `1.64e-5` on one and
  better by `6.09e-5` on the other; neither implementation failed.

Additional audit stress tests covered 144 random coupled positive-definite quadratics with mixtures of two-sided, one-sided, unbounded, and fixed coordinates: **0 failures at `1e-7` loss parity**. Bounded Rosenbrock tests at dimensions 2, 4, 8, and 12 had maximum JAX-vs-SciPy loss differences of approximately `2.6e-20`, `4.4e-16`, `6.8e-13`, and `4.3e-10`, respectively.

## Optim.jl native L-BFGS-B regression panel

Relevant cases from the native L-BFGS-B test suite in JuliaNLSolvers/Optim.jl were adapted to this package. The earlier CPU run passed **13/13** adapted cases. They cover active-bound solutions, bounded Rosenbrock, one-sided and fixed bounds, high-dimensional active sets with strict feasibility, starts directly on a boundary, memory lengths `m=1,2,4,20`, float32 active sets, iteration-limit semantics, the three-problem Fortran-reference cross-check panel, and Optim.jl's dedicated regression for subspace backtracking measured from the generalized Cauchy point.

During the audit, the most sensitive Optim.jl-derived cases were rerun after the solver corrections: Rosenbrock converged for `m=1,2,4,20`, and the subspace-backtracking regression reached loss `0.12234569775490674` in 32 evaluations (well inside the `<=55` guard).

The Optim.jl cases that are interface-specific rather than algorithmic (callbacks/tracing), alternate line-search implementations not exposed by this package, and BigFloat support were not ported.

## Audited GPU results

The same 32-start x 8-problem float32 panel produced **249/256 starts within
`1e-5`** with `maxls=20`. All seven endpoint misses were on the intentionally
near-flat quadratic; the largest absolute gap was `6.11e-5`. One coupled
quadratic start ended with line-search status 4, but its canonical loss differed
from SciPy by only `7.45e-9`. Running the panel with `maxls=40` removed that
terminal status without changing the seven flat-direction comparisons.

These float32 misses are relative-function-reduction decisions in a direction
whose curvature is deliberately tiny. They are retained in the panel because
they expose the numerical boundary rather than hiding it in an aggregate.

## CPU scaling

The earlier lightweight CPU scaling results remain useful as batching-overhead measurements, but they should be rerun after this solver correction before being treated as final performance numbers. See `validation/SCALING_CPU.md` and `benchmarks/scaling_panel.py`.

The public-DM-architecture benchmark is a synthetic-target compute proxy. It can
use either deterministic random weights or a trusted trained surface-network
checkpoint. Its checkpoint forward pass and density collapse were checked
against DM's implementations, including edge-padded Gaussian smoothing. With
the production 20-observation checkpoint, a one-start float64 GPU run matched
SciPy to `2.97e-13` loss and 46 iterations (104 versus 101 evaluations).
Float32 can follow a different path through this nonconvex objective, so its
endpoint gap is not used as the portable solver-parity criterion. See
`validation/DM_PUBLIC_ARCHITECTURE.md`.

## Actual-DM WNM likelihood integration

`benchmarks/dm_wnm_likelihood.py` uses the packaged 100-observation K12 WNM and
the frozen `ordinary_1_seed0_n450` actual-DM recovery dataset. It reproduces the
recovery selection: WNM continuous point NLL, artifact bounds, float32 search,
seed-0 deterministic log-space Latin-hypercube starts, 32 starts, 500
iterations, `ftol=1e-9`, and `gtol=1e-6`. Failed-status finite points remain in
the diagnostics, as in the recovery protocol.

On CPU, float32 batched JAX took 2.759 s and serial SciPy 3.223 s (1.17x). All 32 paired
endpoints were within 0.001 NLL. After the recovery diagnostic's promoted
float64 rescore, their best losses differed by `9.54e-10` and all 32 pairs were
within 0.001.

On GPU, a monolithic 32-start float32 batch took 0.173 s after compilation. This is
39.3x faster than serial SciPy driving the same GPU objective, and 18.6x faster
than the selected CPU-SciPy baseline measured above. Individual GPU float32
paths are not parity-stable: only 3 of 32 endpoint pairs were within 0.001 after
float64 rescoring. Nevertheless, the selected JAX-GPU winner's float64 loss was
`937.4738208`, only `7.33e-06` above the CPU-SciPy winner's `937.4738135`; the
fitted vectors were respectively `[10.1883, 30.2419, 57.2422]` and
`[10.1872, 30.2480, 57.2430]` degrees.

Thus this case supports winner-level recovery and useful accelerator throughput,
not start-by-start GPU trajectory equivalence. Compile time is excluded from all
steady-state numbers. See `validation/DM_WNM_LIKELIHOOD.md` for commands and
interpretation.

The matched float64-search diagnostic removes the instability completely: all
32 starts converged for both implementations and all paired endpoints were
within `1e-5` on CPU and GPU. CPU JAX took 4.930 s versus SciPy's 2.232 s;
GPU JAX took 1.687 s versus SciPy's 2.635 s. Relative to float32, JAX float64
was 1.79x slower on CPU and 9.76x slower on the RTX 5080 GPU. The GPU float64
compile was also 11.844 s versus 1.018 s for float32. SciPy became faster in
float64 because stable arithmetic reduced its evaluations from 1,437 to 587 on
CPU and from 1,579 to 585 on GPU. These measurements characterize a diagnostic
search; they do not change the recovery project's selected float32 search.

## Still required

Extend the likelihood check across the frozen development cases, trial counts,
and generating regimes before claiming panel-level WNM equivalence. Validate
the other objectives against their selected searches rather than imposing one
universal 32-start comparison: bias-weighted CRPS uses 32-start serial L-BFGS-B,
density uses the reusable 80-by-64 log-curve cache plus one polish, and smoothed
expectation uses that cache with multiple distinct polishes and a conditional
hierarchy. Curve comparisons must remain stratified by feature dissimilarity.
