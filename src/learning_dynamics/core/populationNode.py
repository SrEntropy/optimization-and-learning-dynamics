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
        if self.requires_grad:
            self.grad = [0.0 for _ in self.grad]

    def _enforce_shape(self, other):
        if len(self.data) != len(other.data):
            raise ValueError(
                f"Population size mismatch: {len(self.data)} vs {len(other.data)}"
            )

    # -------------------------
    # Autodiff
    # -------------------------

    def backprop(self, debug=False, seed_grad=None):
        """
        Reverse-mode autodiff

        seed_grad:
          - None (default):
              - if output is scalar: seed grad = 1.0
              - if output is vector: seed grad = ones (interpretable as upstream grad of ones)
          - list[float]:
              - must match the output shape

        Note (warning!):
        - Gradients accumulate by design (+=). If backprop is called multiple times
        without clearing  grad, values will add up.  
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
        if seed_grad is None:
            if len(self.grad) == 1:
                self.grad = [1.0]
            else:
                # Keep your original behavior for vector outputs
                self.grad = [1.0 for _ in self.grad]
        else:
            if len(seed_grad) != len(self.grad):
                raise ValueError(
                    f"seed_grad shape mismatch: expected {len(self.grad)} got {len(seed_grad)}"
                )
            self.grad = [float(g) for g in seed_grad]

        # Reverse traversal
        for node in reversed(topo):
            node._backward()
            if debug:
                print(
                    f"[NODE] op={node.op}, value={node.data}, grad={node.grad} | "
                    f"<-- Parents={[child.data for child in node._parents]}"
                )

    # -------------------------
    # Optional helper for reusing graphs/state
    # -------------------------
    def zero_grad_graph(self):
        """Zero grads for all nodes reachable from this node
        (including intermediates)."""
        topo  = []
    
        visited = set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    visit(parent)
                topo.append(node)

        visit(self)
        for node in topo:
            node.zero_grad()


    # -------------------------
    # Optional operator overloading
    # -------------------------

    def __add__(self, other):
        from learning_dynamics.core.ops import add
        return add(self, other)
    
    def __sub__(self, other):
        from learning_dynamics.core.ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        from learning_dynamics.core.ops import sub
        return sub(other, self)


    def __mul__(self, other):
        from learning_dynamics.core.ops import mul
        return mul(self, other)

    __radd__ = __add__
    __rmul__ = __mul__

    
    def tanh(self):
        from learning_dynamics.core.ops import tanh
        return tanh(self)

    # -------------------------
    # Representation
    # -------------------------

    def __repr__(self):
        return (
            f"PopulationNode(data={self.data}, "
            f"grad={self.grad}, op='{self.op}')"
        )
