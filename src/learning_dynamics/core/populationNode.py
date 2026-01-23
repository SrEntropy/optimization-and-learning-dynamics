class PopulationNode:
    """
    PopulationNode: This object represents a *state variable* in a computation graph.
    It is conceptually similar to a population of neurons with shared dynamics. 
     .data :
        A list of floats representing the activity/state of each unit
        in the population. This is the forward-pass value.

    .grad :
        A list of floats of the same size as .data.
        During backpropagation, .grad stores the accumulated partial derivatives d(output)/d(this node).
        It is only valid *after* calling .backprop().

    Think of this as:
        neuronal activity, membrane potentials, firing rates, not synaptic plasticity or weight updates.
   Think: neuronal activity, not synaptic weights. 
   """
    
    def __init__(self, data, _parents=(), op="leaf", requires_grad=True):
        # Normalize data to vector
        if isinstance(data, (int, float)):
            self.data = [float(data)]
        else:
            self.data = list(data)

        # Gradient vector (same size as data)
        self.grad = [0.0 for _ in self.data]

        # Graph structure
        self._parents = tuple(_parents)
        self.op = op
        self.requires_grad = requires_grad

        # Local backward function (set by ops)
        self._backward = lambda: None

    # -------------------------
    # Utility
    # -------------------------

    def zero_grad(self):
        # Reset gradient to zero (used before each backward pass)
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

        # Seed gradient for final output
        self.grad = [1.0 for _ in self.grad]

        # Reverse traversal
        for node in reversed(topo):
            node._backward()
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
