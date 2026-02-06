# learning_dynamics/core/populationNode.py

from __future__ import annotations
from typing import Iterable, Callable, Tuple, List, Optional, Any


class PopulationNode:
    """
    PopulationNode = state variable in a computation graph.

    Conceptually:
      - Think "population activity" (neuronal state), not synaptic weights.
      - .data is the forward-pass value (vector of floats).
      - .grad accumulates d(output)/d(this node), same shape as .data,
        valid only after .backprop().

    Notes:
      - Gradients ACCUMULATE by design (+=). Call .zero_grad_graph()
        before a fresh backward pass if you don't want accumulation.
      - This class is intentionally lightweight; it is not a full tensor library.
    """

    def __init__(
        self,
        data: Any,
        _parents: Tuple["PopulationNode", ...] = (),
        op: str = "leaf",
        requires_grad: bool = True,
    ):
        # Normalize data to a list[float]
        if isinstance(data, (int, float)):
            self.data: List[float] = [float(data)]
        else:
            # Force float conversion for numerical hygiene and consistent behavior
            self.data = [float(x) for x in data]

        # Gradient vector (same shape as data)
        self.grad: List[float] = [0.0 for _ in self.data]

        # Graph structure
        self._parents: Tuple["PopulationNode", ...] = tuple(_parents)
        self.op: str = op
        self.requires_grad: bool = bool(requires_grad)

        # Local backward function (set by ops)
        self._backward: Callable[[], None] = lambda: None

    # -------------------------
    # Utility
    # -------------------------

    def zero_grad(self) -> None:
        """Reset *this node's* grad buffer to zero."""
        if self.requires_grad:
            self.grad = [0.0 for _ in self.grad]

    def _enforce_shape(self, other: "PopulationNode") -> None:
        """Strict shape check for elementwise ops."""
        if len(self.data) != len(other.data):
            raise ValueError(
                f"Population size mismatch: {len(self.data)} vs {len(other.data)}"
            )

    def _topological_order(self) -> List["PopulationNode"]:
        """
        Return nodes reachable from self in topological order.
        (parents come before children)
        """
        topo: List[PopulationNode] = []
        visited = set()

        def visit(node: PopulationNode):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    visit(parent)
                topo.append(node)

        visit(self)
        return topo

    # -------------------------
    # Autodiff
    # -------------------------

    def backprop(self, debug: bool = False, seed_grad: Optional[List[float]] = None) -> None:
        """
        Reverse-mode autodiff from this node.

        seed_grad:
          - None (default):
              - if output is scalar: seed grad = [1.0]
              - if output is vector: seed grad = ones (interpretable as upstream grad of ones)
          - list[float]:
              - must match output shape exactly

        IMPORTANT:
          - Gradients accumulate (+=). Call zero_grad_graph() beforehand
            if you want a clean backward pass.
        """
        topo = self._topological_order()

        # Seed gradient for final output node
        if seed_grad is None:
            if len(self.grad) == 1:
                self.grad = [1.0]
            else:
                # For vector output, default is upstream ones (sum-of-components objective)
                self.grad = [1.0 for _ in self.grad]
        else:
            if len(seed_grad) != len(self.grad):
                raise ValueError(
                    f"seed_grad shape mismatch: expected {len(self.grad)} got {len(seed_grad)}"
                )
            self.grad = [float(g) for g in seed_grad]

        # Reverse traversal: apply each node's local backward rule
        for node in reversed(topo):
            node._backward()

            if debug:
                print(
                    f"[NODE] op={node.op}, value={node.data}, grad={node.grad} | "
                    f"<-- Parents={[p.data for p in node._parents]}"
                )

    def zero_grad_graph(self) -> None:
        """
        Zero grads for all nodes reachable from this node (including intermediates).
        Use before running a fresh backward pass to avoid accumulation.
        """
        for node in self._topological_order():
            node.zero_grad()

    # -------------------------
    # Operator overloading
    # -------------------------

    def __add__(self, other):
        from core.ops import add
        return add(self, other)

    def __sub__(self, other):
        from core.ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from core.ops import sub
        return sub(other, self)

    def __mul__(self, other):
        from core.ops import mul
        return mul(self, other)

    __radd__ = __add__
    __rmul__ = __mul__

    def matvec(self, A):
        from core.ops import matvec
        return matvec(A, self)

    def tanh(self):
        from core.ops import tanh
        return tanh(self)

    # -------------------------
    # Representation
    # -------------------------

    def __repr__(self) -> str:
        return f"PopulationNode( data={self.data}, grad={self.grad}, op='{self.op}')"
