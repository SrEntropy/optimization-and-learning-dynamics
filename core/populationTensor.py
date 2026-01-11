import math

class PopulationTensor:
    """
    Population Tensor:
    Represent a vector-valued node corresponding to a population of homogeneous scalar units sharing the same operation.
    - Scalars are treated as population of size 1 (see value.py where classical scalar-only autodiff is implemented as a base-line.).
    Objective: This abstraction is intentionally chosen to study learning dynamic, population coding, and brain-inspired models, rather than classical scalar-only autodiff.
    """
    def __init__(self, data, _children=(), required_grad=True, op="leaf"):
        # Update: Handle both list(vector) and floats(scalar) 
        if isinstance(data, list):
            self.data = [float(data)]
            self.grad = [0.0 * len(self.data)]
        else: 
            self.data = float(data)
            self.grad = 0.0

        self.required_grad = required_grad
        self.op = op
        self._children = tuple(_children)
        self._chain_rule = lambda: None
        
    #
    def __repr__(self):
        return f"Tensor(data={[self.data]}, grad={[self.grad]}, op='{self.op}')"

    def _enforce_shape(self, val):
        if len(self.data) != len(val.data):
            raise ValueError("Shape mismatch Tensor operation")
        
    def __add__(self, val):
        self._enforce_shape(val)
        val = val if isinstance(val, Tensor) else Tensor(val)

        #Forward Pass:
        #1. Vector addition + Broadcasting
        if isinstance(self.data, list) and isinstance(val.data, list):
            new_data = [a + b for a, b in zip(self.data, val.data)]
        #2. Vector + Scalar
        elif isinstance(self.data, list):
            new_data = [x + val.data for x in self.data]
        #3. Scalar + Vector
        elif isinstance(val.data, list):
            new_data = [self.data + x for x in val.data]
        else:
            new_data = self.data + val.data
        
        y = Tensor(new_data, (self, val), op = "+")
        #2. Backward Pass: Handle vector gradients
        def _chain_rule():
            # If self is a vector, distribute the upstream gradient
            if isinstance(self.data, list):
                #Gradient of addition is 1.0--> element wise if list-->return [1.0, 1.0, 1.0] else return---->[1.0]*3---->[1.0, 1.0, 1.0] 
                upstream = y.grad if isinstance(y.grad, list) else [y.grad]*len(self.data) 
                for i in range(len(self.grad)):
                  self.grad[i] += upstream[i]  
            else: 
                #If self is a scalar, sum all upstream gradients(Broadcasting rule)
                self.grad += sum(y.grad) if isinstance(y.grad, list) else y.grad

            if isinstance(self.val, list):
                upstream = y.grad if isinstance(y.grad, list) else [y.grad]*len(val.data) 
                for i in range(len(val.grad)):
                  val.grad[i] += upstream[i]  
            else: 
                #If self is a scalar, sum all upstream gradients(Broadcasting rule)
                val.grad += sum(y.grad) if isinstance(y.grad, list) else y.grad
                
        y._chain_rule = _chain_rule        
        return y
    
    __radd__ = __add__

    def __mul__(self, val):
        self._enforce_shape(val)
        val = val if isinstance(val, Tensor) else Tensor(val)
        y = Tensor(self.data * val.data, (self, val), op = "*")
        def _chain_rule():
            self.grad += val.data * y.grad 
            val.grad += self.data * y.grad
        y._chain_rule = _chain_rule
        return y

    __rmul__ = __mul__

    def tanh(self):
        x = self.data
        t = (math.exp(x*2) - 1)/(math.exp(x*2) + 1)
        y = Tensor(t, (self, ), op = "tanh")
        def _chain_rule():
            self.grad += (1 - t**2) * y.grad
        y._chain_rule = _chain_rule
        return y

    def backprop(self):
        top_nodes = []
        visited = set()
        def visit(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    visit(child)
                top_nodes.append(v)

        visit(self)
        self.grad = 1.0
        for node in top_nodes[::-1]:        
            node._chain_rule()
            print(
    f"[NODE] op={node.op}, value={node.data}, grad={node.grad} | "f"<-- children={[child.data for child in node._children]}"
)




