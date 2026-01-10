
#TODO: Minimal Tensor clas
# Roadmap:
# 1. Scalar autodiff (DONE)
# 2. Nonlinearities (DONE)
# 3. Clean-up + comment
# 4. Vector support
# 5. ops.py modularization
# 6. XOR experiment
# 7. Tensor semantics (shape, orientation, transformations)
import math

class Tensor:
    """
    A minimal scalar reverse autodiff node.
    Stores a value, gradient, children, and a local backward rule.
    """
    def __init__(self, data, _children=(), required_grad=True, op="leaf"):
        self.data = float(data)
        self.grad = 0.0
        self.required_grad = required_grad
        self.op = op
        self._children = tuple(_children)
        self._chain_rule = lambda: None
        
    #
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op='{self.op}')"
    
    @staticmethod
    def guard(fn, tensor):
        def wrapper():
            if not tensor.required_grad:
                return 
            fn()
        return wrapper

    def __add__(self, val):
        val = val if isinstance(val, Tensor) else Tensor(val)
        y = Tensor(self.data + val.data, (self, val), op = "+")
        def _chain_rule():
            self.grad += 1.0 * y.grad
            val.grad += 1.0 * y.grad
        y._chain_rule = Tensor.guard(_chain_rule, y)
        return y
    __radd__ = __add__

    def __mul__(self, val):
        val = val if isinstance(val, Tensor) else Tensor(val)
        y = Tensor(self.data * val.data, (self, val), op = "*")
        def _chain_rule():
            self.grad += val.data * y.grad 
            val.grad += self.data * y.grad
        y._chain_rule = Tensor.guard(_chain_rule, y)
        return y

    __rmul__ = __mul__

    def tanh(self):
        x = self.data
        t = (math.exp(x*2) - 1)/(math.exp(x*2) + 1)
        y = Tensor(t, (self, ), op = "tanh")
        def _chain_rule():
            self.grad += (1 - t**2) * y.grad
        y._chain_rule = Tensor.guard(_chain_rule, y)
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

"""
Once TODO1 is solid:
TODO 2:
- Add vector support
- Add ops.py
- Add XOR experiment
- Note before TODO 1

Tensors are container to store:
    shape: How many dimensions
    orientation: How axes relate
    transformation: How it behaves under operations
Tensors are the natural language of:
    Neula networks
    Physics
    Differential geometry
    Optimization
So, a tensor is a container for numbers witha structures

"""



