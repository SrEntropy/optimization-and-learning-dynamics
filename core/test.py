from tensor import Tensor
# Validation

"""
-------Test 1: Simple chain-----------
"""

"""
Result:
dz/dx = y + 1 = 4
dz/dy = x =  2  
""" 

""" 
Test 2: Non-linearity

x = Tensor(0.5)
y = x.tanh()
y.backward()

 
Result:
- Check against analytical derivative
"""

""" 
Test 3: Shared Sub-graph

x = Tensor()
y = x * x + x
y.backward() 
Result: Correct accumulation
- dy/dx = 2x + 1 = 5
"""