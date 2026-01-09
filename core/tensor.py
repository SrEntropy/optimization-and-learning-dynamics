
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
        self.op = op
        self.required_grad = required_grad
        self._backward = lambda: None
        self._children = tuple(_children)
        self._parent = set()

    def __repr__(self):
        return f"Tensor(data={self.data})"


    def __add__(self, value):
        value = value if isinstance(value, Tensor) else Tensor(value)
        result = Tensor(self.data + value.data,(self, value), op= "+")
        def _backward():
            self.grad += 1.0 * result.grad
            value.grad += 1.0 * result.grad
        result._backward = _backward
        return result

    __radd__=__add__
    
    def __sub__(self, value):
        value = value if isinstance(value, Tensor) else Tensor(value)
        result = Tensor(self.data - value.data,(self, value), op = "-" )
        
        def _backward(): 
            self.grad += 1.0 * result.grad 
            value.grad += -1.0 * result.grad 
        result._backward = _backward
        return result
    
    def __rsub__(self, value): 
        return Tensor(value) - self
    

    def __mul__(self, value):
        value = value if isinstance(value, Tensor) else Tensor(value)
        result = Tensor(self.data * value.data,(self, value), op = "*")
        def _backward():
            self.grad += value.data * result.grad
            value.grad += self.data * result.grad
        result._backward = _backward
        return result
     
    __rmul__ = __mul__
    

    def __truediv__(self, value):
        value = value if isinstance(value, Tensor) else Tensor(value)
        result = Tensor(self.data / value.data, op = "/")

        def _backward(): 
            self.grad += (1/value.data) * result.grad 
            value.grad += (-self.data / (value.data**2)) * result.grad 
        result._backward = _backward
        return result
    
    def __rtruediv__(self, value): 
        return Tensor(value) / self


    def __pow1__(self, value):
        value = value if isinstance(value, Tensor) else Tensor(value)
        result = Tensor(self.data ** value.data, op = "power")
        def _backward(): 
            self.grad += (value.data * self.data**(value.data - 1)) * result.grad # ignoring derivative wrt exponent for now 
        result._backward = _backward 
        return result
    
    def __rpow1__(self, value):
        return self ** value

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        result = Tensor(t, (self, ), op="tanh")
        def _backward():
            self.grad = (1-t**2)*result.grad
        result._backward = _backward
        return result

       
    def backward(self):
        #TODO:
        """
        - Topilogical sort
        - reverse traversal
            """
        pass

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



