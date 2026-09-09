import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jax_lbfgsb import BatchedLbfgsb


def test_user_validation_rosenbrock_start_no_line_search_failure():
    x0 = np.array([[-0.8337393502380092, 0.7875388841985229]])

    def f(x):
        return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

    r = BatchedLbfgsb(f, [-1.5, -1.5], [1.5, 1.5], gtol=1e-8, maxiter=2000).run(x0)
    assert int(r.status[0]) in (0, 1)
    assert float(r.fun[0]) < 1e-12
    np.testing.assert_allclose(np.asarray(r.x[0]), [1.0, 1.0], atol=1e-5)


def test_user_validation_scaled_quad_start_no_line_search_failure():
    x0 = np.array([[
        0.09891638096515898,
        -0.08504366751960113,
        -0.8135679227343415,
        -0.2679090634647938,
        0.7925077829026115,
    ]])
    target = np.linspace(-0.5, 0.7, 5)
    A = np.diag(np.geomspace(1.0, 100.0, 5))
    Aj = jnp.asarray(A)
    tj = jnp.asarray(target)

    def f(x):
        d = x - tj
        return 0.5 * d @ Aj @ d

    r = BatchedLbfgsb(f, np.full(5, -1.0), np.full(5, 1.0)).run(x0)
    assert int(r.status[0]) in (0, 1)
    assert float(r.fun[0]) < 1e-10
    np.testing.assert_allclose(np.asarray(r.x[0]), target, atol=1e-5)
