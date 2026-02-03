"""
ops.py

Defines all differentiable operations on PopulationNodes.
Each op:

- computes forward population values
- builds the computation graph
- defines a local backward rule for reverse-mode autodiff

This file contains *math*, not learning rules.
"""

# TODO: sub, matmul, etc.

import math
from learning_dynamics.core.populationNode import PopulationNode


# -------------------------
# Elementwise ops
# -------------------------

def _as_node(x):
    # Constants should NOT require grad
    return x if isinstance(x, PopulationNode) else PopulationNode(x, requires_grad=False)


def add(a, b):
    a = _as_node(a)
    b = _as_node(b)
    a._enforce_shape(b)

    out_data = [x + y for x, y in zip(a.data, b.data)]
    out = PopulationNode(out_data, (a, b), op="+")

    def _backward():
        for i in range(len(out.grad)):
            if a.requires_grad:
                a.grad[i] += out.grad[i]
            if b.requires_grad:
                b.grad[i] += out.grad[i]

    out._backward = _backward
    return out


def sub(a, b):
    a = _as_node(a)
    b = _as_node(b)
    a._enforce_shape(b)

    out_data = [x - y for x, y in zip(a.data, b.data)]
    out = PopulationNode(out_data, (a, b), op="-")

    def _backward():
        for i in range(len(out.grad)):
            if a.requires_grad:
                a.grad[i] += out.grad[i]
            if b.requires_grad:
                b.grad[i] -= out.grad[i]

    out._backward = _backward
    return out


def mul(a, b):
    a = _as_node(a)
    b = _as_node(b)
    a._enforce_shape(b)

    # Capture forward values for safety
    a_data = list(a.data)
    b_data = list(b.data)

    out_data = [x * y for x, y in zip(a_data, b_data)]
    out = PopulationNode(out_data, (a, b), op="*")

    def _backward():
        for i in range(len(out.grad)):
            if a.requires_grad:
                a.grad[i] += b_data[i] * out.grad[i]
            if b.requires_grad:
                b.grad[i] += a_data[i] * out.grad[i]

    out._backward = _backward
    return out


def tanh(x):
    x = _as_node(x)

    out_data = [math.tanh(v) for v in x.data]
    out = PopulationNode(out_data, (x,), op="tanh")

    def _backward():
        if not x.requires_grad:
            return
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
    x = _as_node(x)

    out = PopulationNode(sum(x.data), (x,), op="sum")

    def _backward():
        if not x.requires_grad:
            return
        for i in range(len(x.grad)):
            x.grad[i] += out.grad[0]

    out._backward = _backward
    return out
