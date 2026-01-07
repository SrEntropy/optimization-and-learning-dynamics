#TODO: Minimal Tensor clas

"""
TODO 1:

- Scalar ops + backward
- Non-linearities + test
- clean-up + comments

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

class Tensor:
    def __init__(self, data, required_grad=True):
        self.data = float(data)
        self.grad = 0.0
        self.required_grad = required_grad
        self._backward = lambda: None
        self._parents = []

    def backward(self):
        #TODO:
        """
        - Topilogical sort
        - reverse traversal
            """
        pass


""" Validation"""
"""
-------Test 1: Simple chain-----------
"""
x = Tensor(2.0)
y = Tensor(3.0)
z = x * y + x
z.backward()
 
 """
Result:
dz/dx = y + 1 = 4
dz/dy = x =  2  
""" 

""" 
Test 2: Non-linearity
"""
x = Tensor(0.5)
y = x.tanh()
y.backward()


""" 
Result:
- Check against analytical derivative
"""

""" 
Test 3: Shared Sub-graph
"""
x = Tensor()
y = x * x + x
y.backward()
""" 
Result: Correct accumulation
- dy/dx = 2x + 1 = 5
"""