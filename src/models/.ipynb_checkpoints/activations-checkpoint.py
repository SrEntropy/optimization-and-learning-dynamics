import math
from core.populationNode import PopulationNode

from core.ops import _as_node

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


def sigmoid(x: Any) -> PopulationNode:
    x = _as_node(x)

    out_data = [ 1.0 / (1.0 + (math.exp(-z) for z in x.data))]
    out = PopulationNode(
        out_data,
        (x,),
        op="sigmoid",
        requires_grad=x.requires_grad,
    )

    def _backward():
        if not x.requires_grad:
            return
        for i in range(len(out.grad)):
            # d/dx sigmoid(x) = @(x)(1-@(x))
            x.grad[i] += out_data[i] *( 1.0 - out_data[i])*out.grad[i]

    out._backward = _backward
    return out


def relu(x: Any) -> PopulationNode:
    x = _as_node(x)

    out_data = [v if v > 0 else 0.0 for v in x.data]
    out = PopulationNode(
        out_data,
        (x,),
        op="relu",
        requires_grad=x.requires_grad,
    )

    def _backward():
        if not x.requires_grad:
            return
        for i in range(len(out.grad)):
            out_data = (1.0 if x.data[i] > 0 else 0.0)* out.grad[i]
            x.grad[i] += (1.0 - out_data[i] ** 2) * out.grad[i]

    out._backward = _backward
    return out


def softmax(x: Any) -> PopulationNode:
    x = _as_node(x)
    max_val = max(x.data)
    numerator = [math.exp(v - max_val) for v in x.data]
    denominator = sum(numerator)
    out_data = [n/denominator for n in numerator]
    out = PopulationNode(
        out_data,
        (x,),
        op="softmax",
        requires_grad=x.requires_grad,
    )

    def _backward():
        if not x.requires_grad:
            return
        for j in range(len(x.data)): 
            grad_j = 0.0 
            for i in range(len(x.data)): 
                s_i = out_data[i] 
                s_j = out_data[j] 
                ds_i_dx_j = s_i * ((1.0 if i == j else 0.0) - s_j) 
                grad_j += out.grad[i] * ds_i_dx_j 
            x.grad[j] += grad_j

    out._backward = _backward
    return out

