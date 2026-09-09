# JAX L-BFGS-B port

Standalone, dimension-agnostic JAX implementation of **L-BFGS-B**, designed to optimize many independent starts as one `vmap`/`jit` accelerator batch. It does not replace L-BFGS-B with clipped L-BFGS, a parameter transform, or a barrier objective.
