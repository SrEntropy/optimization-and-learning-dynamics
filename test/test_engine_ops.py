from learning_dynamics.core.populationNode import PopulationNode
from learning_dynamics.core.ops import tanh, sum_pop

def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))

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

assert close(x.grad, [4.0]), "Incorrect gradient for x"
assert close(y.grad, [2.0]), "Incorrect gradient for y"

print("✓ Passed: gradients match analytical derivatives")

# ------------------------------------------------------------
# Test 2: Vector test
# ------------------------------------------------------------
header("Test 2: Vector Test (Vector Chain Rule)")

x = PopulationNode([1.0, 2.0])
y = PopulationNode([3.0, 4.0])
z = sum_pop(x + y)
z.backprop()

print(f"x.grad (expected [1.0, 1.0]): {x.grad}")
print(f"y.grad (expected [1.0, 1.0]): {y.grad}")

assert close(x.grad, [1.0, 1.0]), "Incorrect gradient for x"
assert close(y.grad, [1.0, 1.0]), "Incorrect gradient for y"

print("✓ Passed: vector gradients match analytical derivatives")

# ------------------------------------------------------------
# Test 3: Elementwise multiply + sum reduction + tanh neuron
# ------------------------------------------------------------
header("Test 3: (w * x) -> sum -> + b -> tanh")

w = PopulationNode([1.0, -1.0])
x = PopulationNode([1.0, 0.0])
b = PopulationNode(0.0)

z = sum_pop(w * x) + b
a = tanh(z)
a.backprop()

dadz = 1.0 - a.data[0] ** 2
expected_w = [dadz * x.data[0], dadz * x.data[1]]
expected_x = [dadz * w.data[0], dadz * w.data[1]]
expected_b = [dadz]

print(f"a.data: {a.data}")
print(f"w.grad: {w.grad} (expected {expected_w})")
print(f"x.grad: {x.grad} (expected {expected_x})")
print(f"b.grad: {b.grad} (expected {expected_b})")

assert close(w.grad, expected_w), "Incorrect gradient for weights"
assert close(x.grad, expected_x), "Incorrect gradient for inputs"
assert close(b.grad, expected_b), "Incorrect gradient for bias"

print("✓ Passed: tanh derivative and neuron gradients are correct")

# ------------------------------------------------------------
# Test 4: Gradient accumulation
# ------------------------------------------------------------
header("Test 4: Gradient accumulation (no zero_grad)")

x = PopulationNode(2.0)
y = PopulationNode(3.0)
z = x * y + x

z.backprop()
g1x, g1y = x.grad[:], y.grad[:]

z.backprop()  # no reset
g2x, g2y = x.grad[:], y.grad[:]

print(f"First x.grad: {g1x}, second x.grad: {g2x}")
print(f"First y.grad: {g1y}, second y.grad: {g2y}")

assert close(g2x, [2 * g1x[0]]), "Gradients should accumulate for x"
assert close(g2y, [2 * g1y[0]]), "Gradients should accumulate for y"

print("✓ Passed: gradients accumulate as expected")

print("\nAll tests passed successfully.")
