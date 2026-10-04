# Validation status

The task requires genuine L-BFGS-B structure, per-start SciPy comparison,
hierarchical support, and explicit separation of CPU/GPU/DM validation. Portable
CPU/GPU validation is covered here, along with one integration case that fits a
single simulated subject with DM's WNM likelihood.

Terms used below:

- **DM** is the [Demixing Model](https://github.com/achetverikov/demixing_model),
  a model of attraction and repulsion biases between two remembered items. Its
  behavioral fits are the workload that motivated this port. The solver and
  test suite do not depend on it.
- **WNM** is DM's fitted predictor: a neural network that maps noise parameters
  to a conditional wrapped-normal mixture density over response errors. **K12**
  means 12 mixture components.
- **Actual-DM** benchmarks import DM's own code and checkpoints from a sibling
  checkout. The **public-DM-architecture** benchmark instead reimplements DM's
  network and objective inside this repository, so it runs without DM.
- A **recovery dataset** is data simulated from DM with known parameters, used
  to check that a fit recovers them.

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

The recorded panel outputs in `validation_output/cpu64/` and
`validation_output/cpu32/` were regenerated on 2026-10-04 on CPU with the
current solver (source fingerprint `d016017f…`, JAX 0.10.1.dev20261004) and
reproduce these figures. Per problem:

| Problem | float64 within `1e-5` | float64 largest difference | float32 within `1e-5` | float32 largest difference | float32 same stopping reason |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scaled_quad_2` | 32/32 | `6.9e-22` | 32/32 | `3.5e-13` | 32/32 |
| `scaled_quad_5` | 32/32 | `4.8e-19` | 32/32 | `3.2e-11` | 30/32 |
| `scaled_quad_9` | 32/32 | `1.2e-17` | 32/32 | `3.2e-09` | 32/32 |
| `coupled_5` | 32/32 | `1.4e-17` | 32/32 | `7.5e-09` | 18/32 |
| `face_edge_corner_4` | 32/32 | `0` | 32/32 | `0` | 31/32 |
| `flat_direction_5` | 32/32 | `1.7e-13` | 30/32 | `6.1e-05` | 28/32 |
| `hierarchical_7` | 32/32 | `7.4e-22` | 32/32 | `1.3e-12` | 32/32 |
| `rosenbrock_2` | 32/32 | `5.9e-19` | 32/32 | `6.3e-12` | 19/32 |

"Same stopping reason" counts starts where both solvers stopped on the same
test, projected gradient or relative function reduction. In float64 all 256
starts converged in both solvers, for the same reason and after the same number
of evaluations. In float32, every JAX start converged, and SciPy converged on
255. The exception is `coupled_5` start 5, which ended in SciPy's
`ABNORMAL` line-search status while its loss still matched to within `1e-5`.
Differing stopping reasons are expected in float32, where the two tests trigger
at nearly the same step.

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

The per-start output behind these GPU figures is not in the repository. The
`validation_output/gpu32/` directory was produced by the stale pre-audit solver
(its `solver.py` hash matches commit `6da679e`), so it does not record this
result. The GPU panel needs a rerun before these figures have a recorded source.

These float32 misses are relative-function-reduction decisions in a direction
whose curvature is deliberately tiny. They are retained in the panel because
they expose the numerical boundary rather than hiding it in an aggregate.

## CPU scaling

The earlier lightweight CPU scaling results remain useful as batching-overhead measurements, but they should be rerun after this solver correction before being treated as final performance numbers. See `validation/SCALING_CPU.md` and `benchmarks/scaling_panel.py`.

The public-DM-architecture benchmark is a synthetic-target compute proxy. It can
use either deterministic random weights or a trusted trained surface-network
checkpoint. Its checkpoint forward pass and density collapse were checked
against DM's implementations, including edge-padded Gaussian smoothing. With
the production checkpoint trained on 20 internal evidence samples per simulated
trial, a one-start float64 GPU run matched
SciPy to `2.97e-13` loss and 46 iterations (104 versus 101 evaluations).
Float32 can follow a different path through this nonconvex objective, so its
endpoint gap is not used as the portable solver-parity criterion. See
`validation/DM_PUBLIC_ARCHITECTURE.md`.

## Actual-DM WNM likelihood integration

`benchmarks/dm_wnm_likelihood.py` requires a DM checkout. It uses DM's packaged
K12 WNM checkpoint for 100 internal evidence samples per simulated trial
(`current_wnm_k12_100samples.pkl`) and the frozen simulated subject
`ordinary_1_seed0_n450` from DM's recovery dataset. It reproduces the search
settings selected for DM's recovery fits: WNM continuous point negative
log-likelihood (NLL), DM's parameter bounds, float32 search,
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
DM's other fitting objectives against their selected searches rather than imposing one
universal 32-start comparison: bias-weighted CRPS uses 32-start serial L-BFGS-B,
density uses the reusable 80-by-64 log-curve cache plus one polish, and smoothed
expectation uses that cache with multiple distinct polishes and a conditional
hierarchy. Curve comparisons must remain stratified by feature dissimilarity.
