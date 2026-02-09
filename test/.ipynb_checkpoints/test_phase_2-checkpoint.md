## TEST SET 1: Parameter Semantics
- Test 1.1: Parameter is a PopulationNode

Purpose: Verify abstraction separation.


```python
w = Parameter([1.0, 2.0, 3.0])

assert isinstance(w, PopulationNode)
assert w.data == [1.0, 2.0, 3.0]
assert w.grad == [0.0, 0.0, 0.0]
```

What this demonstrates?
- Parameters are signals plus learning capability
- No special casing hidden in Node

```python
# Test 1.2 — zero_grad works
w.grad = [0.5, -0.3, 1.2]
w.zero_grad()
assert w.grad == [0.0, 0.0, 0.0]
```

Conceptual check:
- Gradient accumulation is explicit
- No silent resets during backprop

```python
# Test 1.3 — step() performs GD update
w = Parameter([1.0, -2.0])
w.grad = [0.1, -0.2]
w.step(lr=0.5)

assert w.data == [0.95, -1.9]
```

This is critical:
- This is the learning rule
- You can point to it and say: “This is learning.”

## TEST SET 2: GD Optimizer Class
- Test 2.1 — Optimizer updates all parameters
```python
from core.optim import GD

w1 = Parameter([1.0])
w2 = Parameter([2.0])

w1.grad = [1.0]
w2.grad = [2.0]

opt = GD([w1, w2], lr=0.1)
opt.step()

assert w1.data == [0.9]
assert w2.data == [1.8]
```

- Test 2.2: Optimizer zero_grad
```python
opt.zero_grad()
assert w1.grad == [0.0]
assert w2.grad == [0.0]
```

Interpretation:
- Optimizer controls time
- Parameters control state

---

## TEST SET 3: Stability demos (1-D quadratic)

We test the classic discrete GD system on a quadratic:

- Loss:  L(θ) = 1/2 · a · θ²
- Gradient: dL/dθ = aθ
- Update: θ_{t+1} = θ_t - η a θ_t = (1 - ηa) θ_t

So:
- r = (1 - ηa)
- Stable (shrinks): |r| < 1  ⇔  0 < η < 2/a
- Diverges: |r| > 1

Suggested experiments (same as in the notebook):
- lr = 0.3 (converges)
- lr = 1.5 (oscillates but converges)
- lr = -0.2 (gradient ascent → diverges)
- lr = 2.2 (oscillates and diverges)
