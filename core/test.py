from tensor import Tensor

def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ------------------------------------------------------------
# Test 1: Simple Chain Rule
# ------------------------------------------------------------
header("Test 1: Simple Chain Rule")

x = Tensor(2.0)
y = Tensor(3.0)
z = x * y + x
z.backprop()

print(f"x.grad (expected 4): {x.grad}")
print(f"y.grad (expected 2): {y.grad}")

assert abs(x.grad - 4.0) < 1e-6, "Incorrect gradient for x"
assert abs(y.grad - 2.0) < 1e-6, "Incorrect gradient for y"

print("✓ Passed: gradients match analytical derivatives")


# ------------------------------------------------------------
# Test 2: Non-linearity (tanh)
# ------------------------------------------------------------
header("Test 2: Non-linearity (tanh)")

x = Tensor(0.5)
y = x.tanh()
y.backprop()

expected = 1 - y.data**2
print(f"x.grad (expected {expected}): {x.grad}")

assert abs(x.grad - expected) < 1e-6, "Incorrect tanh derivative"

print("✓ Passed: tanh derivative matches analytical form")


# ------------------------------------------------------------
# Test 3: Shared Subgraph
# ------------------------------------------------------------
header("Test 3: Shared Subgraph")

x = Tensor(2.0)
y = x * x + x
y.backprop()

expected = 2 * x.data + 1
print(f"x.grad (expected {expected}): {x.grad}")

assert abs(x.grad - expected) < 1e-6, "Incorrect gradient accumulation"

print("✓ Passed: gradient accumulation over shared subgraph is correct")

print("\nAll tests passed successfully.")

header("Test 4: Freeze leaf x")
x = Tensor(2.0, required_grad=False)
y = Tensor(3.0)
z = x * y + x
z.backprop()

print("x.grad =", x.grad)   # expect 0
print("y.grad =", y.grad)   # expect 2


