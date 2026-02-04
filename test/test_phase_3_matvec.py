import numpy as np

from learning_dynamics.core.populationNode import PopulationNode
from learning_dynamics.core.ops import matvec, sum_pop


def _almost_equal_list(a, b, tol=1e-6):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) < tol


def test_matvec_forward_backward_diagonal():
    A = [[2.0, 0.0],
         [0.0, 3.0]]

    x = PopulationNode([1.0, 1.0], requires_grad=True)
    y = matvec(A, x)          # [2,3]
    loss = sum_pop(y)         # 5

    loss.zero_grad_graph()
    loss.backprop()

    assert y.data == [2.0, 3.0]
    # dL/dx = A^T @ [1,1] = [2,3]
    _almost_equal_list(x.grad, [2.0, 3.0])


def test_matvec_forward_backward_random():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 2))
    x0 = rng.normal(size=(2,))

    x = PopulationNode(x0.tolist(), requires_grad=True)
    y = matvec(A, x)
    loss = sum_pop(y)

    loss.zero_grad_graph()
    loss.backprop()

    # analytic: A^T @ ones(3)
    expected = A.T @ np.ones(3)
    got = np.array(x.grad)
    assert np.max(np.abs(got - expected)) < 1e-6
