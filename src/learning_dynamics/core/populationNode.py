
class PopulationNode:

    """
    Population Tensor Node

    - Stores population-valued data (vector)
    - Stores population-valued gradients
    - Tracks computational graph
    - Executes reverse-mode autodiff

    Math operations are defined externally in ops.py
    """

    def __init__(self, data, _parents=(), op="leaf", requires_grad=True):
        # Normalize data to vector
        if isinstance(data, (int, float)):
            self.data = [float(data)]
        else:
            self.data = list(data)

        self.grad = [0.0 for _ in self.data]
        self._parents = tuple(_parents)
        self.op = op
        self.requires_grad = requires_grad
        self._backward = lambda: None

    # -------------------------
    # Utility
    # -------------------------

    def zero_grad(self):
        self.grad = [0.0 for _ in self.grad]

    def _enforce_shape(self, other):
        if len(self.data) != len(other.data):
            raise ValueError(
                f"Population size mismatch: {len(self.data)} vs {len(other.data)}"
            )

    # -------------------------
    # Autodiff
    # -------------------------

    def backprop(self, debug=False):
        """
        Reverse-mode autodiff
        """
        topo = []
        visited = set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    visit(parent)
                topo.append(node)

        visit(self)
        # Seed gradient
        self.grad = [1.0 for _ in self.grad]

        # Reverse traversal
        for node in reversed(topo):
            node._backward()
             # Debug print (optional)
            if debug: 
                print(
                f"[NODE] op={node.op}, value={node.data}, grad={node.grad} | "
                f"<-- Parents={[child.data for child in node._parents]}"
            )

    # -------------------------
    # Optional operator overloading
    # -------------------------

    def __add__(self, other):
        from learning_dynamics.core.ops import add
        return add(self, other)

    def __mul__(self, other):
        from learning_dynamics.core.ops import mul
        return mul(self, other)

    __radd__ = __add__
    __rmul__ = __mul__

    # -------------------------
    # Representation
    # -------------------------

    def __repr__(self):
        return (
            f"PopulationNode(data={self.data}, "
            f"grad={self.grad}, op='{self.op}')"
        )
