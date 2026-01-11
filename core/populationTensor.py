import math

class PopulationTensor:
    """
    Population Tensor:
    Represent a vector-valued nodecoresponingto a population of hoogeneous calar units sharing the same operation.
    - Scalars are 
    treated as population of size 1 (see value.py where classical scalar-only autodiff is implemented as a base-line.).
    Objective: This abstraction is intentionally chosen to study learning dynamic, population coding, and brain-inspired models, rather than classical scalar-only autodiff.
    """
    def __init__(self, data, _parents=(), required_grad=True, op="leaf"):
        # Normalize data: always a vector 
        if isinstance(data, (int, float)):
            self.data = [float(data)]
        else: 
            self.data = list(data)
        self.grad = [0.0 for _ in self.data]
        self.required_grad = required_grad
        self.op = op
        self._parents = tuple(_parents)
        self._backward = lambda: None
        
    def __repr__(self):
        return (
            f"PopulationTensor(data={self.data}, "
            f"grad={self.grad}, op='{self.op}')"
            )
    #-------------------------
    # Internal utilities
    #-------------------------

    def _enforce_shape(self, other):
        if len(self.data) != len(other.data):
            raise ValueError(
                 f" Population size mismatch:{len(self.data)} vs {len(other.data)}"
                 )
    def zero_grad(self):
        self.grad= [ 0.0 for _ in range(len(self.grad))]
    #-------------------------
    # Elementwise operations
    #-------------------------

    def __add__(self, other):
        other = other if isinstance(other, PopulationTensor) else PopulationTensor(other)
        self._enforce_shape(other)
        
        #Forward Pass
        out_data = [a + b for a, b in zip(self.data, other.data)]
        out =  PopulationTensor(out_data, (self, other), op = "+")

        # --- Backward Pass ---
        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[i]   
                other.grad[i] += out.grad[i]    

        out._backward = _backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        # --- Forward Pass ---
        other = other if isinstance(other, PopulationTensor) else PopulationTensor(other)
        self._enforce_shape(other)
        # z = x * y
        out_data = [a * b for a, b in zip(self.data, other.data)]
        out = PopulationTensor(out_data, (self, other), op="*")

        # --- Backward Pass ---
        def _backward():
            # For each i: z_i = x_i * y_i # dz_i/dx_i = y_i, dz_i/dy_i = x_i
            for i in range(len(self.data)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def tanh(self):
        # --- Forward Pass ---
        out_tanh = [math.tanh(x) for x in self.data]
        out = PopulationTensor(out_tanh, (self, ), op = "tanh")

        # --- Backward Pass ---
        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += (1 - out_tanh[i]**2) * out.grad[i]

        out._backward= _backward
        return out

    #-------------------------
    # Reduction
    #-------------------------
    def sum(self):
        out = PopulationTensor(sum(self.data), (self, ), op = "sum")
        def _backward():
            for i in range(len(self.data)):
                self.grad[i] += out.grad[0]   

        out._backward= _backward
        return out
    
    #-------------------------
    # Reduction
    #-------------------------
    
    def backprop(self):
        # Performs reverse-mode autodiff.
        # --- Build topological order ---
        topo_nodes = []
        visited = set()

        def visit(v):
            if v not in visited:
                visited.add(v)
                for child in v._parents:
                    visit(child)
                topo_nodes.append(v)

        visit(self)

        # --- Seed gradient (population of size 1) ---
        # d(output)/d(output) = 1
        self.grad = [1.0 for _ in self.grad]

        # --- Backward Pass ---
        # Walk graph in reverse topological order
        for node in reversed(topo_nodes):
            node._backward()  # apply local derivative

            # Debug print (optional)
            print(
                f"[NODE] op={node.op}, value={node.data}, grad={node.grad} | "
                f"<-- Parents={[child.data for child in node._parents]}"
            )
