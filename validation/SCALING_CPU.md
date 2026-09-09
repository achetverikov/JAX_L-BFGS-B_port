# CPU scaling with number of starts

Steady-state timings exclude compilation and use the median of 3 repeated full solves from identical starts. SciPy runs one L-BFGS-B solve per start; JAX runs the same start matrix as one compiled batch. The development machine exposed CPU only.

## Float64

### Coupled 8D quadratic

| starts | JAX ms | SciPy ms | speedup |
|---:|---:|---:|---:|
| 1 | 1.47 | 0.65 | 0.44x |
| 4 | 2.10 | 2.14 | 1.02x |
| 8 | 3.36 | 3.80 | 1.13x |
| 16 | 5.49 | 7.45 | 1.36x |
| 32 | 9.31 | 15.12 | 1.62x |
| 64 | 14.79 | 27.13 | 1.83x |
| 128 | 30.22 | 76.24 | 2.52x |

### 6D bounded Rosenbrock

| starts | JAX ms | SciPy ms | speedup |
|---:|---:|---:|---:|
| 1 | 3.79 | 3.12 | 0.82x |
| 4 | 9.66 | 10.53 | 1.09x |
| 8 | 14.71 | 19.17 | 1.30x |
| 16 | 21.89 | 54.69 | 2.50x |
| 32 | 36.60 | 98.37 | 2.69x |
| 64 | 72.89 | 172.29 | 2.36x |
| 128 | 138.39 | 363.30 | 2.63x |

## Float32

### Coupled 8D quadratic

| starts | JAX ms | SciPy ms | speedup |
|---:|---:|---:|---:|
| 1 | 1.20 | 0.81 | 0.67x |
| 4 | 2.68 | 2.89 | 1.08x |
| 8 | 3.61 | 3.15 | 0.87x |
| 16 | 5.07 | 6.74 | 1.33x |
| 32 | 9.64 | 15.29 | 1.59x |
| 64 | 16.59 | 48.35 | 2.91x |
| 128 | 35.63 | 81.64 | 2.29x |

### 6D bounded Rosenbrock

| starts | JAX ms | SciPy ms | speedup |
|---:|---:|---:|---:|
| 1 | 4.90 | 3.69 | 0.75x |
| 4 | 10.23 | 9.58 | 0.94x |
| 8 | 14.39 | 18.18 | 1.26x |
| 16 | 25.04 | 33.48 | 1.34x |
| 32 | 52.90 | 83.33 | 1.58x |
| 64 | 88.35 | 171.51 | 1.94x |
| 128 | 165.63 | 414.06 | 2.50x |

All 28 benchmark cells had 100% of starts within `1e-5` canonical rescored loss of SciPy. Maximum observed absolute loss differences were approximately `9e-12` in float64 and `8.1e-7` in float32.

## Interpretation

A single JAX start is slower because dispatch and compiled-control-flow overhead dominate. The crossover is around 4--16 starts on these cheap CPU objectives. At 32 starts the gain is already meaningful, and by 128 starts both problems are about 2.3--2.6x faster in the more stable measurements. The exact float32 crossover is noisier because the absolute timings are very small.

This is still a conservative CPU benchmark. It does not establish GPU speed. The intended DM workload has a substantially more expensive objective and should benefit more from accelerator batching if the objective itself remains JAX-native and vectorizes across starts.
