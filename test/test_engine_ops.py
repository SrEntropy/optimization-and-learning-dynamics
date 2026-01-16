from learning_dynamics.core.populationNode import PopulationNode
from learning_dynamics.core.ops import tanh, sum_pop



def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ------------------------------------------------------------
# Test 1: Scalar test
# ------------------------------------------------------------
header("Test 1: Scalar Test (Simple Chain Rule)")

x = PopulationNode(2.0)
y = PopulationNode(3.0)
z = x * y + x
print(f"z =  (expected [8.0]): {z.data}")
z.backprop()

print(f"x.grad (expected [4.0]): {x.grad}")
print(f"y.grad (expected [2.0]): {y.grad}")

assert x.grad == [4.0], "Incorrect gradient for x"
assert y.grad == [2.0], "Incorrect gradient for y"

print("✓ Passed: gradients match analytical derivatives")

# ------------------------------------------------------------
# Test 2: Vector test
# ------------------------------------------------------------
header("Test 2: Vector Test (Simple Chain Rule)")

x = PopulationNode([1.0, 2.0])
y = PopulationNode([3.0, 4.0])
z = sum_pop(x+y)     # FIX: must call sum()
z.backprop()

print(f"x.grad (expected [1.0, 1.0]): {x.grad}")
print(f"y.grad (expected [1.0, 1.0]): {y.grad}")

assert x.grad == [1.0, 1.0], "Incorrect gradient for x"
assert y.grad == [1.0, 1.0], "Incorrect gradient for y"

print("✓ Passed: vector gradients match analytical derivatives")

# ------------------------------------------------------------
# Test 3: XOR-ready neuron
# ------------------------------------------------------------
header("Test 3: XOR-ready neuron")

w = PopulationNode([1.0, -1.0])
x = PopulationNode([1.0, 0.0])
b = PopulationNode(0.0)

z = sum_pop(w * x) + b
a = tanh(z)
a.backprop()

# expected derivative of tanh
expected = [1 - a.data[0]**2]   # scalar population of size 1
print(f"x.grad: {x.grad}")
print(f"w.grad: {w.grad}")
print(f"b.grad: {b.grad}")

assert len(w.grad) == 2, "Incorrect gradient length for weights"
assert b.grad == expected, "Incorrect gradient for bias (should be population size 1)"

print("✓ Passed: tanh derivative and neuron gradients are correct")
print("\nAll tests passed successfully.")
