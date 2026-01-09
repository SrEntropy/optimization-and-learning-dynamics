
#TODO: Minimal Tensor clas

"""
aTODO 1:

- Scalar ops + backward
- Non-linearities + test
- clean-up + comments
"""
import math
import numpy as np

class Tensor:
    def __init__(self, data, _children=(), required_grad=True, op=""):
        self.data = float(data)
        self.grad = 0.0
        self._children = set(_children)
        self._chain_rule = lambda: None
        self.op = op

    def __repr__(self):
        return f"value = ({self.data})"

    def __add__(self, val):
        val = val if isinstance(val, Tensor) else Tensor(val)
        y = Tensor(self.data + val.data, (self, val), op = "+")
        def _chain_rule(self):
            self.grad += 1.0 * y.grad
            val.grad += 1.0 * y.grad
        y._chain_rule = _chain_rule
        return y
    __radd__ = __add__

    def __mul__(self, val):
        val = val if isinstance(val, Tensor) else Tensor(val)
        y = Tensor(self.data * val.data, (self, val), op = "*")
        def _chain_rule(self):
            self.grad += val.data * y.grad 
            val.grad += self.data * y.grad
        y._chain_rule =_chain_rule
        return y
    __rmul__ = __mul__

    def tanh(self):
        x = self.data
        t = (math.exp(x**2) - 1)/(math.exp(x**2) + 1)
        y = Tensor(t, (self, ), op = "tanh")
        def _chain_rule():
            self.grad = (1 - t**2)* y.grad
        y._chain_rule = _chain_rule
        return y

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



