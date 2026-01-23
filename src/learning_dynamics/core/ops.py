
"""
ops.py

Defines all differentiable operations on PopulationNodes.
Each op:

- computes forward population values
- builds the computation graph
- defines a local backward rule for reverse-mode autodiff

This file contains *math*, not learning rules.
"""

#TODO:  sub, matmul, etc.

import math
from learning_dynamics.core.populationNode import PopulationNode


# -------------------------
# Elementwise ops
# -------------------------

def add(a, b):
    b = b if isinstance(b, PopulationNode) else PopulationNode(b)
    a._enforce_shape(b)

    out_data = [x + y for x, y in zip(a.data, b.data)]
    out = PopulationNode(out_data, (a, b), op="+")

    def _backward():
        for i in range(len(out.grad)):
            a.grad[i] += out.grad[i]
            b.grad[i] += out.grad[i]

    out._backward = _backward
    return out


def mul(a, b):
    b = b if isinstance(b, PopulationNode) else PopulationNode(b)
    a._enforce_shape(b)

    out_data = [x * y for x, y in zip(a.data, b.data)]
    out = PopulationNode(out_data, (a, b), op="*")

    def _backward():
        for i in range(len(out.grad)):
            a.grad[i] += b.data[i] * out.grad[i]
            b.grad[i] += a.data[i] * out.grad[i]

    out._backward = _backward
    return out


def tanh(x):
    out_data = [math.tanh(v) for v in x.data]
    out = PopulationNode(out_data, (x,), op="tanh")

    def _backward():
        for i in range(len(out.grad)):
            x.grad[i] += (1.0 - out_data[i] ** 2) * out.grad[i]

    out._backward = _backward
    return out


# -------------------------
# Reduction
# -------------------------

def sum_pop(x):
    """
    Sum population into a scalar node
    """
    out = PopulationNode(sum(x.data), (x,), op="sum")

    def _backward():
        for i in range(len(x.grad)):
            x.grad[i] += out.grad[0]

    out._backward = _backward
    return out
