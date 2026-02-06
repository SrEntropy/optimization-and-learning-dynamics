from core.populationNode import PopulationNode

#from core.ops import _as_node

from typing import Tuple, Any, List

def tanh(x: Any) -> PopulationNode:
    x = _as_node(x)

    out_data = [math.tanh(v) for v in x.data]
    out = PopulationNode(
        out_data,
        (x,),
        op="tanh",
        requires_grad=x.requires_grad,
    )

    def _backward():
        if not x.requires_grad:
            return
        for i in range(len(out.grad)):
            # d/dx tanh(x) = 1 - tanh(x)^2
            x.grad[i] += (1.0 - out_data[i] ** 2) * out.grad[i]

    out._backward = _backward
    return out
