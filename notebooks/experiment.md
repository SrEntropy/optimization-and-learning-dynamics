# Experiments
- 1 XOR From Scratch
  - Show failure with linear model
  - Show success with MLP
  - Explain representational necesity

- 2 Vanishing/Exploding Gradients[DONE]
  - Deep tanh network[DONE]
  - Show gradient norm across layers
- 3 Stability of Training
  - Large vs small step size
  - Oscillation vs convergence
- 4 Gradient Flow vs GD
  - Same loss, different dynamics
  - insightful plots

## Experiment 1: Gradient Shape Through Nonlinearity (Core learning-signal intuition)
```python
xs = [-6, -3.0, -0.5, 0, .5, 3.0, 6]
x = PopulationNode(xs)
print("x: ", x)
y = x.tanh()

y.sum().backprop()
```
```text
[NODE] op=leaf, value=[-6, -3.0, -0.5, 0, 0.5, 3.0, 6], grad=[2.4576547405286142e-05, 0.009866037165440211, 0.7864477329659274, 1.0, 0.7864477329659274, 0.009866037165440211, 2.4576547405286142e-05] | <-- Parents=[]
```
### Observation
- **Large gradients**: Inputs near 0, where tanh is steep and responsive.

- **Vanishing gradients**: Inputs far from 0, where tanh saturates and its derivative approaches 0.

- **Symmetry**: Large positive or negative values both produce small gradients (slow learning), while small values of either sign produce large gradients (fast learning).


## Experiment 2:  Depth-Induced Gradient Decay (Vanishing gradient mechanism)
 ```python
 x = PopulationNode([4.5])
y1 = x.tanh()
y2 = y1.tanh()
y3 = y2.tanh()
y4 = y3.tanh()
y5 = y4.tanh()
y6 = y5.tanh()
y7 = y6.tanh()
y8 = y7.tanh()
y8.backprop()
 ```
### Input is 0.5
```text
[NODE] op=tanh, value=[0.3258166487960561], grad=[1.0] | <-- Parents=[[0.33814088946538085]]
[NODE] op=tanh, value=[0.33814088946538085], grad=[0.8938435113673074] | <-- Parents=[[0.3519919121889121]]
:
[NODE] op=tanh, value=[0.4318081805950961], grad=[0.42602061160003407] | <-- Parents=[[0.46211715726000974]]
[NODE] op=tanh, value=[0.46211715726000974], grad=[0.34658553053894303] | <-- Parents=[[0.5]]
[NODE] op=leaf, value=[0.5], grad=[0.27257140477114494] | <-- Parents=[]
```
### Input is 4.5
```text
[NODE] op=tanh, value=[0.41346084639570524], grad=[1.0] | <-- Parents=[[0.43977852653283755]]
[NODE] op=tanh, value=[0.43977852653283755], grad=[0.8290501284977471] | <-- Parents=[[0.47195619292203916]]
:
[NODE] op=tanh, value=[0.7614904913621628], grad=[0.15305097953277297] | <-- Parents=[[0.9997532108480275]]
[NODE] op=tanh, value=[0.9997532108480275], grad=[0.06430164957431485] | <-- Parents=[[4.5]]
[NODE] op=leaf, value=[4.5], grad=[3.173398285314812e-05] | <-- Parents=[]
```
### Observation: 
  - **Gradients shrink** with depth because each layer multiplies its local derivative into the chain.

  - **Saturated tanh inputs (very negative or very positive)** have derivatives near 0, so multiplying many of these across layers causes the gradient to decay rapidly.

## Experiment 3: Population vs Scalar Sensitivity (Why population abstraction matters)
```python 
x_scalar = PopulationNode(0.5)
y_scalar = x_scalar.tanh()
y_scalar.backprop()
print("sum of unit gradient", sum(x_scalar.grad))
```
```text
sum of unit gradient 0.7864477329659274
```
```python
x_pop = PopulationNode([0.5, 0.5, 0.5, 0.5])
y_pop = x_pop.tanh()
y_pop.sum().backprop()
print("sum of population  gradient",sum(x_pop.grad))
```
```text
sum of population  gradient 3.1457909318637096
```
### Observation:
  - Each unit has the same local gradient as a scalar neuron, but  the total gradient magnitude scales with population size, demonstrating that population coding amplifies sensitivity and stabilizes by distributing the representation across many units.


## Experiment 4: Shared Subgraph in Population (Credit assignment)
```python
x = PopulationNode([1.0, 2.0, 3.0])
y = x * x + x
z = y.sum()
z.backprop()
```

```
[NODE] op=leaf, value=[1.0, 2.0, 3.0], grad=[3.0, 5.0, 7.0] | <-- Parents=[]
```

Verify analytically
For each element:
- 𝑑/𝑑𝑥($x^2$ + 𝑥) => 2𝑥+ 1
  - 2(1.0)+ 1 = 3
  - 2(2.0)+ 1 = 5
  - 2(3.0)+ 1 = 7

### Observation: 
-  Autodiff engine computes credit assignment in a population setting, even when a node is used multiple time in the computation graph. Each element of the population gets the correct accumulated gradients from all paths.ted gradient from all paths.

## Experiment 5 — Reduction as Information Bottleneck(Learning signal compression)
```python
x = PopulationNode([1.0, 1.0, 1.0])
y = x.tanh()
z = y.sum()
z.backprop()
sum:  1.2599230248420783
```
Variation
Change one element:
```python
x = PopulationNode([1.0, 1.0, 3.0])
sum:  0.849814720393492
```
### Observation: Information Bottleneck
- Gradient redistribution
- Loss of population identity

- When a population is reduced with sum(), all of its individual differences collapse into a single scalar. This destroys the structure of the population: every unit must share the same error signal, and only the total responsiveness survives. Responsive units strengthen the total signal, saturated units weaken it, and the original per‑unit contributions cannot be recovered.

## Experiment 6: Gradient Symmetry Breaking (Why identical units diverge)
```python
x = PopulationNode([0.5, 0.5001])
y = x.tanh()
y.sum().backprop()
```
```text
Input: [0.5, 0.5001]
Gradients: [0.7864477329659274, 0.7863750439424013]
```

### Observation:
- Tiny differences in initial values create tiny differences in local derivatives.
- These small mismatches make the system highly sensitive to even microscopic perturbations.
- After the reduction (sum()), both units receive the same upstream error signal, but their slightly different derivatives scale it differently. This tiny mismatch breaks the symmetry, causing their gradients — and eventually their learning trajectories — to diverge.

## Experiment 7: Gradient Flow Without Updates (Dynamics without learning)

#Setup
#Compute gradient repeatedly for different inputs:

```python
for v in [-3, -1, 0, 1, 3]:
    x = PopulationNode(v)
    y = x.tanh()
    y.backprop()
    print(v, x.grad)
```
```text
input Value:  -3 Gradient:  [0.009866037165440211]
input Value:  -1 Gradient:  [0.41997434161402614]
input Value:  0 Gradient:  [1.0]
input Value:  1 Gradient:  [0.41997434161402614]
input Value:  3 Gradient:  [0.009866037165440211]
```
### Observation:
- Intrinsic gradient flow is the natural sensitivity pattern of an activation function across its input space, measured before any learning or parameter updates occur.
  - **Intrinsic**: Built‑in, natural, already there
  - **Gradient**: How sensitive the output is to a small change in the input
  - **Flow**: How that sensitivity behaves across different inputs

## Summary:
```text
| Experiment | Demonstrates                             |
| ---------- | ---------------------------------------- |
| 1          | Local derivative shapes learning signal  |
| 2          | Depth causes gradient decay              |
| 3          | Population encoding increases robustness |
| 4          | Proper credit assignment                 |
| 5          | Reduction compresses information         |
| 6          | Symmetry breaking without noise          |
| 7          | Learning signals exist without learning  |
```

