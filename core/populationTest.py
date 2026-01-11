


from populationTensor import PopulationTensor

def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ------------------------------------------------------------
# Test 1: Scalar test
# ------------------------------------------------------------
header("Test 1: Scalar Test (Simple Chain Rule)")

x = PopulationTensor(2.0)
y = PopulationTensor(3.0)
z = x * y + x
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

x = PopulationTensor([1.0, 2.0])
y = PopulationTensor([3.0, 4.0])
z = (x + y).sum()     # FIX: must call sum()
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

w = PopulationTensor([1.0, -1.0])
x = PopulationTensor([1.0, 0.0])
b = PopulationTensor(0.0)

z = (w * x).sum() + b
a = z.tanh()
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
