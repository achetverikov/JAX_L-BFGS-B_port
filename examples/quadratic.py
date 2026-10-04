"""Run with `python examples/quadratic.py` after installing the package."""

import jax.numpy as jnp
import numpy as np

from jax_lbfgsb import BatchedLbfgsb


def loss_fn(parameters, target):
    return jnp.sum((parameters - target) ** 2)


def main():
    # The second target coordinate lies outside the box: its optimum is 2.
    target = jnp.array([0.2, 3.0], dtype=jnp.float32)
    starts = jnp.array([[-1.0, 1.0], [1.5, -1.5]], dtype=jnp.float32)
    search = BatchedLbfgsb(loss_fn, lower=[-2.0, -2.0], upper=[2.0, 2.0])
    result = search.run(starts, target)
    winner = result.winner_index()

    print("Solutions:", np.asarray(result.x))
    print("Losses:", np.asarray(result.fun))
    print("Statuses:", result.messages())
    print("Best start:", winner)
    print("Best parameters:", np.asarray(result.x[winner]))
    print("Active upper bounds:", np.asarray(result.active_upper[winner]))


if __name__ == "__main__":
    main()
