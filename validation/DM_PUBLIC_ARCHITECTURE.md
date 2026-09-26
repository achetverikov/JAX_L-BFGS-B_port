# Public DM architecture benchmark

`benchmarks/dm_public_architecture_scaling.py` is a workload benchmark derived from the public `achetverikov/demixing_model` implementation. It mirrors the public `MirrorAwareMu1Predictor` compute shape (3 -> 64 -> 128 -> 256 dense stack, latent reshape, 128/64/32/1 transposed-convolution decoder), the periodic 180-row `mu1_bias` grid, 90 feature-difference columns, signed density-asymmetry collapse, and the `1 - CCC` density objective.

By default the benchmark uses deterministic random weights with the current
128-row native architecture. `--checkpoint` instead loads a trusted
`MirrorAwareMu1Predictor` checkpoint; the extracted production-checkpoint
forward pass was checked against DM's Flax call and matched exactly. The target
is still synthetic, so this is a compute/memory scaling proxy, **not an
empirical recovery/parity test**.

The density collapse now matches DM's model-side definition, including the
41-point, sigma-10 Gaussian kernel and edge padding. GPU convolutions are made
deterministic by default; `--no-deterministic-gpu` is available for an
unconstrained throughput run. The JSON records the requested and actual device,
all solver source files in its fingerprint, per-start statuses, iteration and
evaluation counts, projected-gradient norm, final parameters, and signed loss
gap.

## CPU smoke

Command:

```bash
PYTHONPATH=src python benchmarks/dm_public_architecture_scaling.py \
  --platform cpu \
  --counts 1 --repeats 1 --maxiter 20 \
  --output validation/dm_public_architecture_cpu_smoke.json
```

A run capped at 20 iterations is **throughput only**, not a parity result. The
two methods may stop for different reasons or simply be at different points on
a nonconvex path when the cap is reached. Inspect `per_start`, especially the
statuses and signed loss gap, before interpreting the endpoint difference.

## Accelerator command

```bash
PYTHONPATH=src python benchmarks/dm_public_architecture_scaling.py \
  --platform gpu --dtype float32 --counts 32 --jax-batch-size 1 \
  --repeats 3 --maxiter 120 --maxls 40 \
  --checkpoint ../demixing_model/pretrained/surface_legacy_epoch1425_10ktrain_20samples.pkl \
  --output validation/dm_public_architecture_gpu.json
```

Compilation is reported separately. Increase the largest count only after
checking GPU memory: the batched network gradients create large temporary
allocations. `--jax-batch-size` processes all requested starts in reusable
chunks and pads only the final incomplete chunk, keeping compilation outside the
reported solve time.

In the 2026-09-09 audit, a deterministic one-start float64 run with the
production 20-observation checkpoint (`maxiter=120`, `maxls=40`) matched SciPy
to `2.97e-13` loss. Both took 46 iterations; JAX used 104 evaluations and SciPy
101. In float32 both methods reported function-reduction convergence, but their
losses differed by `5.28e-4`. Small line-search differences can select different
paths on this nonconvex objective, while float32's discrete resolution can stop
progress with a non-small projected gradient. The portable convex panel is the
appropriate solver-parity test; real subject-target recovery still belongs in
the DM pipeline.
