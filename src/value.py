# TODO: Minimal Value class
# Roadmap:
# 1. Scalar autodiff (DONE)
# 2. Nonlinearities (DONE)
# 3. Clean-up + comment (THIS STEP)

import math

class Value:
    """
    A minimal scalar reverse-mode autodiff node.
    Each Value represents a single scalar value in a computation graph.
    It stores:
      • data: the scalar value from the forward pass
      • grad: the accumulated gradient d(output)/d(this node)
      • _children: the nodes that produced this node (graph edges)
      • _chain_rule: the local derivative used during backprop
    """

    def __init__(self, data, _children=(), required_grad=True, op="leaf"):
        # Forward-pass value (always a scalar float)
        self.data = float(data)

        # Gradient accumulator (starts at 0, filled during backprop)
        self.grad = 0.0

        # Whether this node participates in gradient computation
        self.required_grad = required_grad

        # Operation label (for debugging / graph tracing)
        self.op = op

        # Parents in the computation graph
        self._children = tuple(_children)

        # Local derivative function (set by operations like +, *, tanh)
        self._chain_rule = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op='{self.op}')"
    
    def __add__(self, val):
        # --- Forward Pass ---
        # Promote Python numbers to Value
        val = val if isinstance(val, Value) else Value(val)

        # Compute the forward value
        # z = x + y
        y = Value(self.data + val.data, (self, val), op="+")

        # --- Backward Pass ---
        # For z = x + y:
        #   ∂z/∂x = 1
        #   ∂z/∂y = 1
        #
        # Multivariate chain rule:
        #   x.grad += (∂z/∂x) * z.grad
        #   y.grad += (∂z/∂y) * z.grad
        def _chain_rule():
            self.grad += 1.0 * y.grad     # derivative of addition wrt left input
            val.grad += 1.0 * y.grad      # derivative wrt right input

        y._chain_rule = _chain_rule
        return y

    __radd__ = __add__

    def __mul__(self, val):
        # --- Forward Pass ---
        val = val if isinstance(val, Value) else Value(val)

        # z = x * y
        y = Value(self.data * val.data, (self, val), op="*")

        # --- Backward Pass ---
        # For z = x * y:
        #   ∂z/∂x = y
        #   ∂z/∂y = x
        #
        # Chain rule:
        #   x.grad += y * z.grad
        #   y.grad += x * z.grad
        def _chain_rule():
            self.grad += val.data * y.grad
            val.grad += self.data * y.grad

        y._chain_rule = _chain_rule
        return y

    __rmul__ = __mul__

    def tanh(self):
        # --- Forward Pass ---
        # Compute tanh(x) manually (avoids numerical issues)
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)

        y = Value(t, (self,), op="tanh")

        # --- Backward Pass ---
        # d/dx tanh(x) = 1 - tanh(x)^2
        #
        # Chain rule:
        #   x.grad += (1 - t^2) * y.grad
        def _chain_rule():
            self.grad += (1 - t**2) * y.grad

        y._chain_rule = _chain_rule
        return y

    def backprop(self):
        """
        Performs reverse-mode autodiff.

        Steps:
        1. Build a topological ordering of the computation graph.
           (We must visit children before parents.)
        2. Seed the output node's gradient with 1.0.
        3. Traverse nodes in reverse topological order,
           calling each node's local chain rule.
        """

        # --- Build topological order ---
        top_nodes = []
        visited = set()

        def visit(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    visit(child)
                top_nodes.append(v)

        visit(self)

        # --- Seed gradient at the output ---
        # d(output)/d(output) = 1
        self.grad = 1.0

        # --- Backward Pass ---
        # Walk graph in reverse topological order
        for node in reversed(top_nodes):
            node._chain_rule()  # apply local derivative

            # Debug print (optional)
            print(
                f"[NODE] op={node.op}, value={node.data}, grad={node.grad} | "
                f"<-- children={[child.data for child in node._children]}"
            )



