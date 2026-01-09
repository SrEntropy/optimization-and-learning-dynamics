from tensor import Tensor
# Validation

"""
-------Test 1: Simple chain-----------
"""
x1 = Tensor(2.0)
w1 = Tensor(-3.0)

x1w1 = x1*w1 


x2 = Tensor(0.0)
w2 = Tensor(1.0)

#Node2
x2w2 = x2*w2
print(x2w2._children, x2w2.op, x2w2)

#Layer-2
#Node1
x1x2 = x1w1 + x2w2
print( x1x2._children, x1x2.op, x1x2) 
#Node2
b = Tensor(6.88137)

#Layer-3
#Node1
n = x1x2+b 
print(n._children, n.op, n)
#Layer-4

o = n.tanh()
print(o._children, o.op, o)

o.grad = 1.0
print(o.grad)
o._chain_rule()
print("n =", n.data, "n_grad = ",n.grad)
"""
n._backward()
print("x1x2= ",x1x2.data, "x1x2_grad = ",x1x2.grad)
print("b =", "b_grad = ", b.grad)

x1x2._backward()
b._backward()
print("x1w1 = ",x1w1.data, "x1w1_grad = ",x1w1.grad)
print("x2w2 =", x2w2.data, "x2w2_grad = ", x2w2.grad)


x1w1._backward()
x2w2._backward()
print("x1 = ",x1.data, "x1_grad = ",x1.grad)
print("w1 = ",w1.data, "w1_grad = ",w1.grad)
print("x2 = ",x2.data, "x1_grad = ",x2.grad)
print("w2 = ",w2.data, "x1_grad = ",w2.grad)

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