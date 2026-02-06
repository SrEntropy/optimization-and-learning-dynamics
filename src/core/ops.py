# Learning_dynamics/core/ops.py

"""
Defines differentiable operations on PopulationNode.

Each op:
  - computes forward population values
  - builds the computation graph
  - defines a local backward rule for reverse-mode autodiff

This file contains *math ops*, not learning rules or optimizers.
"""

import math
from typing import Tuple, Any, List
from core.populationNode import PopulationNode


# -------------------------
# Helpers
# -------------------------

def _as_node(x: Any) -> PopulationNode:
    """
    Convert Python scalars / iterables into PopulationNode constants.

    Constants should NOT require grad by default.
    """
    return x if isinstance(x, PopulationNode) else PopulationNode(x, requires_grad=False)


def _broadcast_to_match(a: PopulationNode, b: PopulationNode) -> Tuple[PopulationNode, PopulationNode]:
    """
    Minimal broadcasting:
      - allow scalar (len==1) to broadcast to vector length
      - backward sums broadcast grads back into the scalar parent

    Returns potentially new nodes (broadcasted versions).
    """
    la, lb = len(a.data), len(b.data)

    if la == lb:
        return a, b

    # a is scalar, b is vector => broadcast a
    if la == 1 and lb > 1:
        parent = a
        out = PopulationNode(
            [parent.data[0]] * lb,
            (parent,),
            op="broadcast_scalar",
            requires_grad=parent.requires_grad,
        )

        def _backward():
            if not parent.requires_grad:
                return
            # Sum all gradient contributions back into the scalar
            parent.grad[0] += sum(out.grad)

        out._backward = _backward
        return out, b

    # b is scalar, a is vector => broadcast b
    if lb == 1 and la > 1:
        parent = b
        out = PopulationNode(
            [parent.data[0]] * la,
            (parent,),
            op="broadcast_scalar",
            requires_grad=parent.requires_grad,
        )

        def _backward():
            if not parent.requires_grad:
                return
            parent.grad[0] += sum(out.grad)

        out._backward = _backward
        return a, out

    raise ValueError(f"Cannot broadcast shapes: {la} vs {lb}")


# -------------------------
# Elementwise ops
# -------------------------

def add(a: Any, b: Any) -> PopulationNode:
    a = _as_node(a)
    b = _as_node(b)
    a, b = _broadcast_to_match(a, b)
    a._enforce_shape(b)

    out_data = [x + y for x, y in zip(a.data, b.data)]
    out = PopulationNode(
        out_data,
        (a, b),
        op="+",
        requires_grad=(a.requires_grad or b.requires_grad),
    )

    def _backward():
        for i in range(len(out.grad)):
            if a.requires_grad:
                a.grad[i] += out.grad[i]
            if b.requires_grad:
                b.grad[i] += out.grad[i]

    out._backward = _backward
    return out


def sub(a: Any, b: Any) -> PopulationNode:
    a = _as_node(a)
    b = _as_node(b)
    a, b = _broadcast_to_match(a, b)
    a._enforce_shape(b)

    out_data = [x - y for x, y in zip(a.data, b.data)]
    out = PopulationNode(
        out_data,
        (a, b),
        op="-",
        requires_grad=(a.requires_grad or b.requires_grad),
    )

    def _backward():
        for i in range(len(out.grad)):
            if a.requires_grad:
                a.grad[i] += out.grad[i]
            if b.requires_grad:
                # d/d(b) (a - b) = -1
                b.grad[i] -= out.grad[i]

    out._backward = _backward
    return out


def mul(a: Any, b: Any) -> PopulationNode:
    a = _as_node(a)
    b = _as_node(b)
    a, b = _broadcast_to_match(a, b)
    a._enforce_shape(b)

    # Capture forward values for safety (avoid mutation surprises)
    a_data = list(a.data)
    b_data = list(b.data)

    out_data = [x * y for x, y in zip(a_data, b_data)]
    out = PopulationNode(
        out_data,
        (a, b),
        op="*",
        requires_grad=(a.requires_grad or b.requires_grad),
    )

    def _backward():
        for i in range(len(out.grad)):
            if a.requires_grad:
                a.grad[i] += b_data[i] * out.grad[i]
            if b.requires_grad:
                b.grad[i] += a_data[i] * out.grad[i]

    out._backward = _backward
    return out


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


# -------------------------
# Reduction
# -------------------------

def sum_pop(x: Any) -> PopulationNode:
    """
    Sum population vector into a scalar node.

    If x is length-N, output is length-1:
      out = sum_i x_i
      d(out)/d(x_i) = 1
    """
    x = _as_node(x)

    out = PopulationNode(
        sum(x.data),
        (x,),
        op="sum",
        requires_grad=x.requires_grad,
    )

    def _backward():
        if not x.requires_grad:
            return
        # out is scalar => out.grad[0] broadcasts to each x_i
        for i in range(len(x.grad)):
            x.grad[i] += out.grad[0]

    out._backward = _backward
    return out


# -------------------------
# Matrix-vector multiply
# -------------------------

def matvec(A: Any, x: Any) -> PopulationNode:
    """
    Matrix-vector multiply: y = A @ x

    Inputs:
      - A: constant matrix (list-of-lists or numpy array)
      - x: PopulationNode (vector)

    Backprop:
      dL/dx = A^T @ dL/dy
    """
    x = _as_node(x)

    # Convert numpy arrays to list-of-lists if needed
    if hasattr(A, "tolist"):
        A = A.tolist()

    if not isinstance(A, (list, tuple)) or len(A) == 0:
        raise ValueError("A must be a non-empty matrix (list-of-lists or numpy array).")
    if not isinstance(A[0], (list, tuple)):
        raise ValueError("A must be a 2D matrix (list-of-lists).")

    m = len(A)         # output size
    n = len(A[0])      # input size

    if len(x.data) != n:
        raise ValueError(f"matvec shape mismatch: A is {m}x{n}, x is length {len(x.data)}")
    for r in A:
        if len(r) != n:
            raise ValueError("All rows of A must have the same length.")

    # Forward: y = A x
    x_data = list(x.data)
    out_data: List[float] = []
    for i in range(m):
        s = 0.0
        row = A[i]
        for j in range(n):
            s += float(row[j]) * x_data[j]
        out_data.append(s)

    out = PopulationNode(
        out_data,
        (x,),
        op="matvec",
        requires_grad=x.requires_grad,
    )

    # Precompute A^T for backward
    AT = [[float(A[i][j]) for i in range(m)] for j in range(n)]

    def _backward():
        if not x.requires_grad:
            return
        # x.grad += A^T @ out.grad
        for j in range(n):
            s = 0.0
            for i in range(m):
                s += AT[j][i] * out.grad[i]
            x.grad[j] += s

    out._backward = _backward
    return out
