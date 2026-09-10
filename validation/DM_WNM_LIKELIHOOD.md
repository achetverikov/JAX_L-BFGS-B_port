# Actual-DM WNM likelihood benchmark

`benchmarks/dm_wnm_likelihood.py` is the real-model counterpart to the synthetic
surface-network workload. It imports the current DM implementation instead of
copying the WNM equations. The benchmark uses:

- `pretrained/wnm_k12_100samples.pkl`;
- the frozen actual-DM `ordinary_1_seed0_n450` recovery dataset;
- continuous WNM point negative log likelihood;
- artifact bounds of 2.5--200 degrees for both feature SDs and 5--200 degrees
  for spatial SD;
- 32 seed-0 Latin-hypercube starts dispersed in log space; and
- the recovery settings of 500 iterations, `ftol=1e-9`, and `gtol=1e-6`.

These choices follow the current parameter-recovery decision. They should not
be transferred to the WNM density objective, whose selected search is an
80-by-64 reusable log-curve cache followed by one continuous polish.

## Commands

From `JAX_L-BFGS-B_port` in the three-repository workspace:

```bash
PYTHONPATH=src python benchmarks/dm_wnm_likelihood.py \
  --platform cpu --counts 32 --repeats 3 \
  --output validation/dm_wnm_likelihood_cpu32.json

PYTHONPATH=src python benchmarks/dm_wnm_likelihood.py \
  --platform gpu --counts 32 --repeats 3 \
  --output validation/dm_wnm_likelihood_gpu32.json

PYTHONPATH=src python benchmarks/dm_wnm_likelihood.py \
  --platform cpu --dtype float64 --counts 32 --repeats 3 \
  --output validation/dm_wnm_likelihood_cpu64.json

PYTHONPATH=src python benchmarks/dm_wnm_likelihood.py \
  --platform gpu --dtype float64 --counts 32 --repeats 3 \
  --output validation/dm_wnm_likelihood_gpu64.json
```

The benchmark records checkpoint, input, DM-source, benchmark, and complete
solver-source hashes. Compilation is separate from steady-state timing. Every
start includes its initial and final parameters, canonical loss, status,
iterations, and evaluations for both implementations.

The default search uses DM recovery's float32 arithmetic. By default, the benchmark
then follows the recovery diagnostic: it promotes the checkpoint variables and
endpoints and records a float64 rescore. `--no-float64-rescore` disables that
additional diagnostic. `--dtype float64` instead searches with that promoted
arithmetic; it is a diagnostic because parameter recovery selected float32 and
did not adopt an x64 search.

## Current result

| Device | Precision | JAX steady state | SciPy steady state | JAX/SciPy speedup | JAX compile |
|---|---|---:|---:|---:|---:|
| CPU | float32 | 2.759 s | 3.223 s | 1.17x | 1.650 s |
| CPU | float64 | 4.930 s | 2.232 s | 0.45x | 1.618 s |
| RTX 5080 GPU | float32 | 0.173 s | 6.793 s | 39.28x | 1.018 s |
| RTX 5080 GPU | float64 | 1.687 s | 2.635 s | 1.56x | 11.844 s |

At float32, GPU JAX is 18.6x faster than the CPU-SciPy implementation selected
by parameter recovery. At float64 that cross-device advantage falls to 1.32x.
These ratios are derived from separate device runs, not a same-process timing.

For JAX itself, float64 is 1.79x slower than float32 on CPU and 9.76x slower on
this GPU. The GPU result is hardware-specific: the RTX 5080 has much lower
double-precision than single-precision throughput. The float64 GPU compilation
cost is also 11.6x larger in this run.

CPU endpoint parity is strong: all 32 paired endpoints are within 0.001 NLL,
and the promoted rescore puts the best JAX and SciPy losses within `9.54e-10`.

GPU start-by-start parity is not strong. Only 3 of 32 pairs are within 0.001
after the promoted rescore; 28 JAX endpoints are better by more than 0.001, one
SciPy endpoint is better, and one pair differs by 79.8 because the scalar SciPy
path remains in a worse basin. This is consistent with the recovery finding
that float32 likelihood reductions are path-sensitive on GPU. It must not be
reported as 9.4% solver agreement without that context.

The selected solutions do agree. The promoted losses and parameters are:

| Search | Best NLL | sd_feat1 | sd_feat2 | sd_spat |
|---|---:|---:|---:|---:|
| CPU SciPy | 937.4738135 | 10.1872 | 30.2480 | 57.2430 |
| CPU JAX | 937.4738135 | 10.1872 | 30.2479 | 57.2427 |
| GPU JAX | 937.4738208 | 10.1883 | 30.2419 | 57.2422 |

The GPU-JAX winner is `7.33e-06` NLL above the CPU-SciPy winner under the stable
rescore. This validates the selected winner on one representative recovery
dataset; it does not replace a panel stratified by regime and trial count.

## Float64 search accuracy and work

Float64 removes the path instability rather than merely hiding it in a rescore.
On both CPU and GPU, all 32 starts converged for JAX and SciPy and every paired
endpoint was within `1e-5`. The CPU best losses were identical at the stored
precision; the GPU best-loss difference was `1.71e-12`.

The higher precision also reduced optimizer work:

| Device | Precision | JAX evaluations | SciPy evaluations |
|---|---|---:|---:|
| CPU | float32 | 880 | 1,437 |
| CPU | float64 | 588 | 587 |
| GPU | float32 | 1,539 | 1,579 |
| GPU | float64 | 586 | 585 |

This explains why SciPy is faster in float64 despite more expensive arithmetic:
its stable run uses roughly 60% fewer evaluations. JAX also uses fewer
evaluations, but batched float64 arithmetic dominates the saving, especially on
the consumer GPU.
