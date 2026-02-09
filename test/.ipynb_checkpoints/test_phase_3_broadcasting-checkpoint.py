import math
import numpy as np

from learning_dynamics.core.populationNode import PopulationNode
from learning_dynamics.core.ops import add, sub, mul, tanh, sum_pop


def _almost_equal_list(a, b, tol=1e-6):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) < tol


def test_add_forward_backward():
    x = PopulationNode([1.0, 2.0], requires_grad=True)
    y = PopulationNode([3.0, 4.0], requires_grad=True)

    out = add(x, y)
    loss = sum_pop(out)  # scalar

    loss.zero_grad_graph()
    loss.backprop()

    assert out.data == [4.0, 6.0]
    _almost_equal_list(x.grad, [1.0, 1.0])
    _almost_equal_list(y.grad, [1.0, 1.0])


def test_sub_forward_backward():
    x = PopulationNode([1.0, 2.0], requires_grad=True)
    y = PopulationNode([3.0, 4.0], requires_grad=True)

    out = sub(x, y)
    loss = sum_pop(out)

    loss.zero_grad_graph()
    loss.backprop()

    assert out.data == [-2.0, -2.0]
    _almost_equal_list(x.grad, [1.0, 1.0])
    _almost_equal_list(y.grad, [-1.0, -1.0])


def test_mul_forward_backward():
    x = PopulationNode([2.0, 3.0], requires_grad=True)
    y = PopulationNode([5.0, 7.0], requires_grad=True)

    out = mul(x, y)
    loss = sum_pop(out)

    loss.zero_grad_graph()
    loss.backprop()

    assert out.data == [10.0, 21.0]
    # d/dx sum(x*y) = y
    _almost_equal_list(x.grad, [5.0, 7.0])
    # d/dy sum(x*y) = x
    _almost_equal_list(y.grad, [2.0, 3.0])


def test_broadcast_scalar_backward():
    # a is scalar, b is vector => broadcast a
    a = PopulationNode(2.0, requires_grad=True)         # shape (1,)
    b = PopulationNode([1.0, 2.0, 3.0], requires_grad=True)

    out = mul(a, b)         # out = [2,4,6]
    loss = sum_pop(out)     # loss = 12

    loss.zero_grad_graph()
    loss.backprop()

    # dloss/da = sum(b) = 6
    _almost_equal_list(a.grad, [6.0])
    # dloss/db = a = 2 for each
    _almost_equal_list(b.grad, [2.0, 2.0, 2.0])


def test_tanh_grad_finite_difference():
    # Compare analytic gradient with finite difference
    eps = 1e-6
    x0 = np.array([0.1, -0.2, 0.3], dtype=float)

    x = PopulationNode(x0.tolist(), requires_grad=True)
    y = tanh(x)
    loss = sum_pop(y)

    loss.zero_grad_graph()
    loss.backprop()
    analytic = np.array(x.grad, dtype=float)

    # Finite difference
    fd = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps

        lp = sum(math.tanh(v) for v in xp)
        lm = sum(math.tanh(v) for v in xm)
        fd[i] = (lp - lm) / (2 * eps)

    assert np.max(np.abs(analytic - fd)) < 1e-4
