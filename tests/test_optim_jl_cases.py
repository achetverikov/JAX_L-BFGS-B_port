import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from jax_lbfgsb import BatchedLbfgsb


def run1(fun, lower, upper, x0, **kw):
    opt = BatchedLbfgsb(fun, np.asarray(lower), np.asarray(upper), **kw)
    return opt.run(np.asarray(x0)[None, :])


def rosen(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def test_optimjl_active_bound_solution():
    f = lambda x: (x[0] - 3.0) ** 2 + (x[1] + 4.0) ** 2
    r = run1(f, [-2.0, -2.0], [2.0, 2.0], [0.0, 0.0])
    x = np.asarray(r.x[0])
    assert np.all((x >= [-2, -2]) & (x <= [2, 2]))
    np.testing.assert_allclose(x, [2, -2], atol=1e-6)


def test_optimjl_rosenbrock():
    r = run1(rosen, [-2.0, -2.0], [2.0, 2.0], [-1.2, 1.0], gtol=1e-8, maxiter=2000)
    np.testing.assert_allclose(np.asarray(r.x[0]), [1.0, 1.0], atol=1e-4)


def test_optimjl_one_sided_and_fixed():
    f = lambda x: (x[0] - 3.0) ** 2 + (x[1] + 4.0) ** 2
    r = run1(f, [1.0, -np.inf], [np.inf, np.inf], [3.0, 0.0], gtol=1e-8)
    np.testing.assert_allclose(np.asarray(r.x[0]), [3.0, -4.0], atol=1e-6)
    r = run1(f, [1.0, -10.0], [1.0, 10.0], [1.0, 0.0], gtol=1e-8)
    np.testing.assert_allclose(np.asarray(r.x[0]), [1.0, -4.0], atol=1e-6)


def test_optimjl_high_dimensional_active_set_and_strict_feasibility():
    n = 30
    c = np.linspace(-3.0, 3.0, n)
    lo = np.full(n, -1.0)
    hi = np.full(n, 1.0)
    xstar = np.clip(c, lo, hi)
    f = lambda x: jnp.sum((x - jnp.asarray(c)) ** 2)
    r = run1(f, lo, hi, np.zeros(n), gtol=1e-8)
    x = np.asarray(r.x[0])
    assert np.all(x >= lo) and np.all(x <= hi)
    np.testing.assert_allclose(x, xstar, atol=1e-6)
    assert np.any(x == -1.0)
    assert np.any(x == 1.0)


def test_optimjl_boundary_start():
    n = 10
    c = np.linspace(-3.0, 3.0, n)
    lo = np.full(n, -1.0)
    hi = np.full(n, 1.0)
    f = lambda x: jnp.sum((x - jnp.asarray(c)) ** 2)
    r = run1(f, lo, hi, np.full(n, -1.0), gtol=1e-8)
    np.testing.assert_allclose(np.asarray(r.x[0]), np.clip(c, lo, hi), atol=1e-6)


@pytest.mark.parametrize("m", [1, 2, 4, 20])
def test_optimjl_memory_lengths(m):
    r = run1(
        rosen,
        [-2.0, -2.0],
        [2.0, 2.0],
        [-1.2, 1.0],
        maxcor=m,
        gtol=1e-8,
        maxiter=2000,
        maxfun=10000,
    )
    np.testing.assert_allclose(np.asarray(r.x[0]), [1.0, 1.0], atol=1e-4)


def test_optimjl_float32_active_set():
    n = 12
    c = np.linspace(-3.0, 3.0, n, dtype=np.float32)
    lo = np.full(n, -1.0, np.float32)
    hi = np.full(n, 1.0, np.float32)
    f = lambda x: jnp.sum((x - jnp.asarray(c)) ** 2)
    opt = BatchedLbfgsb(f, lo, hi, gtol=1e-5)
    r = opt.run(np.zeros((1, n), np.float32))
    x = np.asarray(r.x[0])
    assert x.dtype == np.float32
    assert np.all(x >= lo) and np.all(x <= hi)
    np.testing.assert_allclose(x, np.clip(c, lo, hi), atol=1e-4)


def test_optimjl_iteration_limit():
    r = run1(rosen, [-2.0, -2.0], [2.0, 2.0], [-1.2, 1.0], maxiter=3)
    assert int(r.iterations[0]) == 3
    assert int(r.status[0]) == 2


def test_optimjl_reference_three_cases():
    # Mirrors Optim.jl's cross-check panel but compares against analytic targets.
    r = run1(rosen, [-2.0, -2.0], [2.0, 2.0], [-1.2, 1.0], gtol=1e-9, maxiter=10000, maxfun=10000)
    assert float(r.fun[0]) < 1e-8

    f2 = lambda x: (x[0] - 3.0) ** 2 + (x[1] + 4.0) ** 2
    r2 = run1(f2, [-2.0, -2.0], [2.0, 2.0], [0.0, 0.0], gtol=1e-9)
    np.testing.assert_allclose(np.asarray(r2.x[0]), [2.0, -2.0], atol=1e-4)

    c = np.linspace(-3.0, 3.0, 25)
    lo = np.full(25, -1.0)
    hi = np.full(25, 1.0)
    f3 = lambda x: jnp.sum((x - jnp.asarray(c)) ** 2)
    r3 = run1(f3, lo, hi, np.zeros(25), gtol=1e-9)
    np.testing.assert_allclose(np.asarray(r3.x[0]), np.clip(c, lo, hi), atol=1e-4)


def test_optimjl_subspace_backtracking_regression():
    # Adapted from Optim.jl's native L-BFGS-B regression: when a clipped
    # subspace step hits a bound, backtracking must measure from the unclamped
    # generalized Cauchy point rather than from already-clamped coordinates.
    lo = np.array([-0.6246061824346689, -1.3848789693740542])
    hi = np.array([1.2952187059881008, 0.42144488961717363])
    x0 = np.array([-0.6246061824346689, 0.42144488961717363])
    r = run1(rosen, lo, hi, x0, maxcor=5, gtol=1e-5, maxiter=15000, maxfun=15000)
    assert int(r.status[0]) in (0, 1)
    x = np.asarray(r.x[0])
    assert np.all(x >= lo) and np.all(x <= hi)
    np.testing.assert_allclose(float(r.fun[0]), 0.12234569775490675, atol=1e-8)
    assert int(r.evaluations[0]) <= 55
