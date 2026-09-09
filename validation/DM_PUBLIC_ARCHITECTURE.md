# Public DM architecture benchmark

`benchmarks/dm_public_architecture_scaling.py` is a workload benchmark derived from the public `achetverikov/demixing_model` implementation. It mirrors the public `MirrorAwareMu1Predictor` compute shape (3 -> 64 -> 128 -> 256 dense stack, latent reshape, 128/64/32/1 transposed-convolution decoder), the periodic 180-row `mu1_bias` grid, 90 feature-difference columns, signed density-asymmetry collapse, and the `1 - CCC` density objective.

The public pretrained checkpoint is present on GitHub, but its 9-15 MB pickle could not be materialized through the connector in this clean runtime. Therefore this benchmark uses deterministic random weights with the same architecture. It is a compute/memory scaling proxy, **not a WNM recovery/parity test and not a claim about the trained checkpoint's landscape**.

## CPU smoke

Environment: Python 3.13.5, JAX/JAXlib 0.9.0.1, SciPy 1.17.0, CPU only.

Command:

```bash
PYTHONPATH=src python benchmarks/dm_public_architecture_scaling.py \
  --counts 1 --repeats 1 --maxiter 20 \
  --output validation/dm_public_architecture_cpu_smoke.json
```

Observed steady-state optimizer runtime after compilation:

- one-start JAX: 3.9543 s
- one-start SciPy: 1.1918 s
- speed ratio: 0.30x (JAX slower)

At the forced 20-iteration cap the final rescored losses did not agree within 1e-5, so this smoke result is **throughput only**, not a parity result. A one-start `maxiter=120` run exceeded this environment's 90 s execution window. Four-start compilation likewise exceeded the shorter CPU execution window. Those facts make the benchmark more useful on the intended GPU machine than on this CPU runner.

## Accelerator command

```bash
PYTHONPATH=src python benchmarks/dm_public_architecture_scaling.py \
  --dtype float32 --counts 1,4,8,16,32,64,128 --repeats 3 --maxiter 120 \
  --output validation/dm_public_architecture_gpu.json
```

Compilation is reported separately. The benchmark compares identical starts and rescored final parameters, but checkpoint-level WNM parity still belongs on the DM machine with the real checkpoint and empirical/recovery artifacts.
