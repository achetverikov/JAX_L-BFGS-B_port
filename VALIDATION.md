# Validation status

The attached task requires genuine L-BFGS-B structure, per-start SciPy comparison, hierarchical support, and explicit separation of CPU/GPU/DM validation. GPU and WNM/DM validation are not claimed here.

## Clean-machine CPU development run

Local development environment: Python 3.13.5, JAX/JAXlib 0.9.0.1, SciPy 1.17.0, NumPy 2.3.5, pytest 9.0.2. No CUDA device was available.

The full local implementation test suite passed 11 CPU tests with one GPU test skipped. It covered interior/boundary quadratics, simultaneous bounds, coupled active sets, bounded Rosenbrock, independent batched termination, failure/limit reporting, arbitrary dimensions, and synthetic `2*C+1` hierarchical recovery.

A 32-start x 8-problem parity panel gave 256/256 float64 CPU starts within `1e-5` canonical rescored loss of SciPy. Float32 gave 255/256 within `1e-5`; the single discrepancy was an advantage for JAX on a deliberately nearly-flat direction: JAX loss `1.397e-13` versus SciPy `6.093e-05`, with SciPy stopping on relative function reduction. Median repeated aggregate CPU speedup was about 1.82x float64 and 2.04x float32. These CPU timings do not resolve the required GPU performance gate.

## Important repository/CI check

The branch includes CPU CI so the exact committed implementation is independently exercised after upload. Treat CI as authoritative for the committed branch; the local numbers above describe the clean-machine development implementation and parity artifacts.

## Still required

Run float32 GPU tests/parity; inspect every GPU discrepancy; then on the DM machine run identical 32-start WNM comparisons for `current_kde_ccc`, `binned_sign_ccc`, `kernel_sign_ccc`, `matched_kde_ccc`, and `sign_likelihood`, rescoring both methods on matched dtype/device and stratifying by target, trial count, generating regime, and feature dissimilarity. Do not claim WNM parity or production readiness before those checks.
