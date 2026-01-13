from populationNode import  PopulationNode

print("LAYER 1 — WEEK 1 EXPERIMENTS (NO TRAINING)")
# These are micro-experiments. 
# Each one should fit in a small script or notebook cell.

print("--------------Experiment 1 — Gradient Shape Through Nonlinearity------")
print("------------------Core learning-signal intuition- -------")
xs = [-3, -2, -1, 0, 1, 2, 3]
x = PopulationNode(xs)
y = x.tanh()
y.sum().backprop()

"""
What to record
Where gradients are large
Where gradients vanish
Symmetry
"""

print("\n-----------Experiment 2 — Depth-Induced Gradient Decay--------------")
print("-----------------Vanishing gradient mechanism-----------")
x = PopulationNode([0.5])
y1 = x.tanh()
y2 = y1.tanh()
y3 = y2.tanh()
y3.backprop()
"""
x.grad, y1.grad, y2.grad
Key variation
- Repeat with:
"""
x = PopulationNode([2.0])

print("-----------Experiment 3 — Population vs Scalar Sensitivity----------------------")
print("---------------------Why population abstraction matters--------")
x_scalar = PopulationNode(0.5)
x_pop = PopulationNode([0.5, 0.5, 0.5, 0.5])

y_scalar = x_scalar.tanh()
y_pop = x_pop.tanh()

y_scalar.backprop()
y_pop.sum().backprop()

"""
Observe:
- Per-unit gradients
- Aggregate gradient magnitude
"""
print("------------Experiment 4 — Shared Subgraph in Population------------")
print("---------------------Credit assignment------------------------")
x = PopulationNode([1.0, 2.0, 3.0])
y = x * x + x
z = y.sum()
z.backprop()
"""
Verify analytically
For each element:
- 𝑑/𝑑𝑥(𝑥^2+𝑥)=2𝑥+ 1
"""

print("\n-----------------Experiment 5 — Reduction as Information Bottleneck---------------------")
print("\n-------------------Learning signal compression------------")
x = PopulationNode([1.0, 1.0, 1.0])
y = x.tanh()
z = y.sum()
z.backprop()

"""
Variation
Change one element:
"""
x = PopulationNode([1.0, 1.0, 3.0])
"""
Observe:
- Gradient redistribution
- Loss of population identity
"""

print("\n-------------------Experiment 6 — Gradient Symmetry Breaking-------------------------------")
print("\n-----------------------Why identical units diverge-----------")
x = PopulationNode([0.5, 0.5001])
y = x.tanh()
y.sum().backprop()
"""
Observe:
- Small differences in gradients
- Sensitivity to perturbations
"""
print("\n-------------------Experiment 7 — Gradient Flow Without Updates----------------------------")
print("\n------------------------Dynamics without learning-------------------")

#Setup
#Compute gradient repeatedly for different inputs:
for v in [-3, -1, 0, 1, 3]:
    x = PopulationNode(v)
    y = x.tanh()
    y.backprop()
    print(v, x.grad)

print("LAYER 2 — WHAT EACH EXPERIMENT DEMONSTRATES")
"""
| Experiment | Demonstrates                             |
| ---------- | ---------------------------------------- |
| 1          | Local derivative shapes learning signal  |
| 2          | Depth causes gradient decay              |
| 3          | Population encoding increases robustness |
| 4          | Proper credit assignment                 |
| 5          | Reduction compresses information         |
| 6          | Symmetry breaking without noise          |
| 7          | Learning signals exist without learning  |
"""
