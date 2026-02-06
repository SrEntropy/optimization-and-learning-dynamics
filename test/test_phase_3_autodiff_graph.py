from core.populationNode import PopulationNode
from core.ops import sum_pop


def test_grad_accumulates_then_resets():
    x = PopulationNode([1.0, 2.0], requires_grad=True)
    loss = sum_pop(x)  # loss = x1 + x2

    loss.zero_grad_graph()
    loss.backprop()
    g1 = list(x.grad)
    assert g1 == [1.0, 1.0]

    # backprop again without clearing => accumulates
    loss.backprop()
    g2 = list(x.grad)
    assert g2 == [2.0, 2.0]

    # clear entire graph
    loss.zero_grad_graph()
    assert x.grad == [0.0, 0.0]
