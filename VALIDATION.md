# Validation status

The attached task requires genuine L-BFGS-B structure, per-start SciPy comparison, hierarchical support, and explicit separation of CPU/GPU/DM validation. GPU and WNM/DM validation are not claimed here.

## Clean-machine CPU development run

Local development environment: Python 3.13.5, JAX/JAXlib 0.9.0.1, SciPy 1.17.0, NumPy 2.3.5, pytest 9.0.2. No CUDA device was available.

The original implementation test suite passed 11 CPU tests with one GPU test skipped. It covered interior/boundary quadratics, simultaneous bounds, coupled active sets, bounded Rosenbrock, independent batched termination, failure/limit reporting, arbitrary dimensions, and synthetic `2*C+1` hierarchical recovery.

## Optim.jl native L-BFGS-B regression panel

After merging the initial implementation to `main`, relevant cases from the native L-BFGS-B test suite in JuliaNLSolvers/Optim.jl were adapted to this package. The CPU run passed **13/13** adapted cases. They cover active-bound solutions, bounded Rosenbrock, one-sided and fixed bounds, high-dimensional active sets with strict feasibility, starts directly on a boundary, memory lengths `m=1,2,4,20`, float32 active sets, iteration-limit semantics, the three-problem Fortran-reference cross-check panel, and Optim.jl's dedicated regression for subspace backtracking measured from the generalized Cauchy point. The latter also satisfied Optim.jl's `<=55` objective-evaluation guard.

The Optim.jl cases that are interface-specific rather than algorithmic (callbacks/tracing), alternate line-search implementations not exposed by this package, and BigFloat support were not ported.

Local commands/results:

```text
PYTHONPATH=src pytest -q tests/test_optim_jl_cases.py
13 passed in 31.81s

PYTHONPATH=src pytest -q tests/test_solver.py tests/test_gpu.py
11 passed, 1 skipped in 25.70s
```

The skipped test is the explicit CUDA test because no GPU is available on this machine.

## Portable SciPy parity

A 32-start x 8-problem parity panel gave 256/256 float64 CPU starts within `1e-5` canonical rescored loss of SciPy. Float32 gave 255/256 within `1e-5`; the single discrepancy was an advantage for JAX on a deliberately nearly-flat direction: JAX loss `1.397e-13` versus SciPy `6.093e-05`, with SciPy stopping on relative function reduction. Median repeated aggregate CPU speedup was about 1.82x float64 and 2.04x float32. These CPU timings do not resolve the required GPU performance gate.

## Important repository/CI check

CPU CI exercises the committed implementation after upload. Treat CI as authoritative for the repository state; the local numbers above describe the clean-machine development implementation and parity artifacts.

## Still required

Run float32 GPU tests/parity; inspect every GPU discrepancy; then on the DM machine run identical 32-start WNM comparisons for `current_kde_ccc`, `binned_sign_ccc`, `kernel_sign_ccc`, `matched_kde_ccc`, and `sign_likelihood`, rescoring both methods on matched dtype/device and stratifying by target, trial count, generating regime, and feature dissimilarity. Do not claim WNM parity or production readiness before those checks.
