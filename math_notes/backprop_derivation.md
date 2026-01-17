
# Backpropagation derivation for PopulationNode

This document derives backpropagation from first principles and connects the math directly to the semantics implemented by `population.py` file.

Proceed in layers:

1. Scalar chain rule  
2. Computation graphs and local derivatives  
3. Vector case and Jacobians  
4. Population‑wise (elementwise) operations  
5. Reductions (e.g. `sum`)  
6. Nonlinearity (e.g. `tanh`) and a neuron example  
7. General reverse‑mode backprop algorithm  

The goal is that every line of code in the engine can be traced back to an equation here.

 ## Resources & References:
- [Functions and respective derivatives of hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions): Useful for verifying nonlinearities like tanh
- [Chain rule and composit functions](https://en.wikipedia.org/wiki/Chain_rule): The mathematical foundation of backpropagation
- [Andrej Karpathy ->Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&t=727s): A hands‑on introduction to reverse‑mode autodiff
---

## 1. Scalar chain rule
- Suppose we have a scalar function $z = f(x)$, and $x \in \mathbb{R}$.
- If $f$ is differentiable, the derivative at a point is: $\frac{dz}{dx} = f'(x)$

The **chain rule** describes how derivatives compose when we have intermediate variables ($𝑥→𝑢→𝑧$).

**What does it mean?** it means that when a function is built from multiple sub‑functions (steps), the derivative of the whole function is the product of the derivatives of each sub‑function (step), chained together. 

Let $u$ and $z$ be the subfunctions: $$u = g(x), \quad z = f(u)$$

Then $z = f(g(x)) $, and the chain rule says:

$$\frac{dz}{dx} = \frac{dz}{du} \cdot \frac{du}{dx}$$

More explicitly:

Step 1: Compute the derivative of the first sub-function $\frac{dz}{du}$
$$\frac{dz}{du} = f'(u)$$

Step2: Compute the derivative of the seconde sub-function $\frac{du}{dx}$

$$\frac{du}{dx} = g'(x)$$

So: the product of  $\frac{dz}{du}$ and $\frac{du}{dx}$ is the derivative of the whole function $\frac{dz}{dx}$
$$\frac{dz}{dx} = f'(u) \cdot g'(x)$$


---
### Example: scalar chain rule used in Test 1

Consider:


$z = x \cdot y + x$

with scalar $ x, y \in \mathbb{R} $. We can define:

$u = x \cdot y, \quad z = u + x$

We want $ \frac{\partial z}{\partial x}$ and $\frac{\partial z}{\partial y}$.

1. Derivative of $z$ w.r.t.  $u$ and $x$:

$\frac{\partial z}{\partial u} = 1$, $\quad \frac{\partial z}{\partial x} \Big|_{\text{direct}} = 1$



2. Derivative of $ u = x \cdot y $:



$$
\frac{\partial u}{\partial x} = y, \quad \frac{\partial u}{\partial y} = x
$$

Now, using the chain rule for $ x $:

$$
\frac{\partial z}{\partial x} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial x} + \frac{\partial z}{\partial x} \Big|_{\text{direct}}
$$



So:



$$
\frac{\partial z}{\partial x} = 1 \cdot y + 1 = y + 1
$$



Similarly, for $ y $:



$$
\frac{\partial z}{\partial y} = \frac{\partial z}{\partial u} \cdot \frac{\partial u}{\partial y} = 1 \cdot x = x
$$



For the concrete values used in the test:

- $ x = 2 $  
- $ y = 3 $

we get:



$$
\frac{\partial z}{\partial x} = 3 + 1 = 4, \quad \frac{\partial z}{\partial y} = 2
$$



This matches the gradients observed in **test/test_engine/Test#1**.

---

## 2. Computation graphs and local derivatives

Backprop is most naturally described using a **computation graph**.

Each node in the graph represents:

- a variable (scalar or vector), or  
- the output of an operation applied to one or more parent nodes.

For scalars, consider the same example:



$$
u = x \cdot y, \quad z = u + x
$$



The graph is:

- leaf nodes: $ x, y $  
- intermediate: $ u = x \cdot y $  
- output: $ z = u + x $

Each edge carries a dependency, and each node has:

- a **value** (forward pass)  
- a **gradient** $ \frac{\partial z}{\partial (\text{node})} $ (backward pass)  

The key idea:

> Backprop computes gradients by applying the chain rule **locally** at each node and propagating gradients backward through the graph.

Every operation defines a **local derivative rule**. For scalar variables:

- Addition: $ z = x + y $

  

$$
  \frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1
  $$



- Multiplication: $ z = x \cdot y $

  

$$
  \frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x
  $$



During backprop, we maintain $ \bar{v} = \frac{\partial z}{\partial v} $ for each node $ v $. For a node $ v $ with parents $ p_1, p_2, \dots $, we distribute its gradient to parents using:



$$
\bar{p_i} \mathrel{+}= \bar{v} \cdot \frac{\partial v}{\partial p_i}
$$



This is the essence of reverse‑mode autodiff.

---

## 3. Vector case and Jacobians
**Jacobian** is a mathematical object that describes how a vector output changes(partial derivatives) with respect to a vector input. 

In **populationTensor class**, all operations have simple Jacobians (diagonal or broadcast), so backprop reduces to multiplying each gradient component by the appropriate local derivative.
Now consider vector‑valued variables. 

Let:

- $ \mathbf{x} \in \mathbb{R}^n $  
- $ \mathbf{u} = g(\mathbf{x}) \in \mathbb{R}^m $  
- $ z = f(\mathbf{u}) \in \mathbb{R} $

The derivative of a scalar $ z $ with respect to a vector $ \mathbf{x} $ is a row vector (or gradient):



$$
\frac{\partial z}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times n}
$$



The derivative of a vector $ \mathbf{u} $ with respect to $ \mathbf{x} $ is the **Jacobian**:

$$
\frac{\partial \mathbf{u}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}
$$

The chain rule in this case:

$$
\frac{\partial z}{\partial \mathbf{x}} = \frac{\partial z}{\partial \mathbf{u}} \cdot \frac{\partial \mathbf{u}}{\partial \mathbf{x}}
$$



where:

- $ \frac{\partial z}{\partial \mathbf{u}} \in \mathbb{R}^{1 \times m} $  
- $ \frac{\partial \mathbf{u}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n} $

and the matrix product yields:



$\frac{\partial z}{\partial \mathbf{x}} \in \mathbb{R}^{1 \times n}$



Reverse‑mode autodiff corresponds to propagating $ \frac{\partial z}{\partial \mathbf{u}} $ backward to get $ \frac{\partial z}{\partial \mathbf{x}} $, accumulating contributions along all paths.

---

## 4. Population‑wise (elementwise) operations

`PopulationNode` stores a **population of scalars**. Conceptually, you can view it as:



$$
\mathbf{x} = [x_1, x_2, \dots, x_n]
$$



Most operations are **elementwise**, meaning they apply independently to each component. This greatly simplifies the Jacobians.

### 4.1 Elementwise addition

Let:



$$
\mathbf{z} = \mathbf{x} + \mathbf{y}
$$



with $ \mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathbb{R}^n $, so:



$$
z_i = x_i + y_i
$$



The Jacobians:



$$
\frac{\partial z_i}{\partial x_j} = \delta_{ij}, \quad \frac{\partial z_i}{\partial y_j} = \delta_{ij}
$$



where $ \delta_{ij} $ is the Kronecker delta.

In matrix form, both Jacobians are identity matrices:



$$
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = I_n, \quad \frac{\partial \mathbf{z}}{\partial \mathbf{y}} = I_n
$$



In reverse mode, this means:



$$
\frac{\partial z}{\partial x_i} \mathrel{+}= \frac{\partial z}{\partial z_i} \cdot 1
$$

$$
\frac{\partial z}{\partial y_i} \mathrel{+}= \frac{\partial z}{\partial z_i} \cdot 1
$$

In other words: gradients are passed back **component‑wise**.

---

### 4.2 Elementwise multiplication

Let:

$$
\mathbf{z} = \mathbf{x} \odot \mathbf{y}
$$

where $ \odot $ is elementwise multiplication, so:

$$
z_i = x_i \cdot y_i
$$

The partial derivatives:

$$
\frac{\partial z_i}{\partial x_j} = 
\begin{cases}
y_i & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
\quad\Rightarrow\quad
\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = \text{diag}(y_1, \dots, y_n)
$$

$$
\frac{\partial z_i}{\partial y_j} =
\begin{cases}
x_i & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
\quad\Rightarrow\quad
\frac{\partial \mathbf{z}}{\partial \mathbf{y}} = \text{diag}(x_1, \dots, x_n)
$$

In reverse mode, for each component $ i $:

$$
\frac{\partial z}{\partial x_i} \mathrel{+}= \frac{\partial z}{\partial z_i} \cdot y_i
$$


$$
\frac{\partial z}{\partial y_i} \mathrel{+}= \frac{\partial z}{\partial z_i} \cdot x_i
$$

This matches the intuition from the scalar case, applied independently per population unit.

---

## 5. Reductions: sum (see notebooks/stage_1-Experiment-5)

A reduction collapses a vector into a scalar. For `PopulationNode`, the canonical example is:

$$
s = \text{sum}(\mathbf{x}) = \sum_{i=1}^n x_i
$$

We want $ \frac{\partial s}{\partial \mathbf{x}} $.

By direct differentiation:


$$
\frac{\partial s}{\partial x_i} = 1 \quad \forall i
$$


So:

$$
\frac{\partial s}{\partial \mathbf{x}} = [1, 1, \dots, 1]
$$

In backprop, if $ \bar{s} = \frac{\partial z}{\partial s} $ is the gradient flowing into the sum node, then for each element:

$$
\frac{\partial z}{\partial x_i} \mathrel{+}= \bar{s} \cdot \frac{\partial s}{\partial x_i} = \bar{s} \cdot 1 = \bar{s}
$$

In other words, the gradient from the scalar sum is **broadcast back** uniformly to all components of the vector.

This is exactly what appears in **Test 2**, where:

- $ \mathbf{z} = \mathbf{x} + \mathbf{y} $  
- $ s = \text{sum}(\mathbf{z}) $

and each component’s gradient with respect to $ \mathbf{x} $ and $ \mathbf{y} $ is $ 1 $.

---

## 6. Nonlinearity: tanh and a neuron example

Consider the hyperbolic tangent:

$$
a = \tanh(u)
$$

For scalar $ u $, the derivative is:

$$
\frac{da}{du} = 1 - \tanh^2(u) = 1 - a^2
$$

For a population‑wise operation:

$$
\mathbf{a} = \tanh(\mathbf{u})
$$

with components:

$$
a_i = \tanh(u_i)
$$

By the same elementwise principle:

$$
\frac{\partial a_i}{\partial u_i} = 1 - \tanh^2(u_i) = 1 - a_i^2
$$

and $ \frac{\partial a_i}{\partial u_j} = 0 $ for $ i \neq j $. Hence, the Jacobian is diagonal:

$$
\frac{\partial \mathbf{a}}{\partial \mathbf{u}} = \text{diag}\big(1 - a_1^2, \dots, 1 - a_n^2\big)
$$

In reverse mode, if $ \bar{a}_i = \frac{\partial z}{\partial a_i} $, then:


$$
\frac{\partial z}{\partial u_i} \mathrel{+}= \bar{a}_i \cdot (1 - a_i^2)
$$

---

### 6.1 Neuron example (as in test/test_engine.py - Test 3)

Consider a simple neuron with:

$$
u = w \cdot x + b
$$

where $ w, x \in \mathbb{R}^n $, and $ b \in \mathbb{R} $, and we define a scalar activation:

$$
a = \tanh(u)
$$

Here:

$$
u = \sum_{i=1}^n w_i x_i + b
$$

We want:

- $ \frac{\partial a}{\partial x_i} $  
- $ \frac{\partial a}{\partial w_i} $  
- $ \frac{\partial a}{\partial b} $

First, derive gradients of $ a $ w.r.t. $ u $:



$$
\frac{da}{du} = 1 - \tanh^2(u)
$$

Now, for $ u $:

$$
u = \sum_{i=1}^n w_i x_i + b
$$

So:

$$
\frac{\partial u}{\partial x_i} = w_i, \quad
\frac{\partial u}{\partial w_i} = x_i, \quad
\frac{\partial u}{\partial b} = 1
$$


Apply the chain rule:

$$
\frac{\partial a}{\partial x_i} = \frac{da}{du} \cdot \frac{\partial u}{\partial x_i} = (1 - \tanh^2(u)) \cdot w_i
$$


$$
\frac{\partial a}{\partial w_i} = \frac{da}{du} \cdot \frac{\partial u}{\partial w_i} = (1 - \tanh^2(u)) \cdot x_i
$$

$$
\frac{\partial a}{\partial b} = \frac{da}{du} \cdot \frac{\partial u}{\partial b} = (1 - \tanh^2(u)) \cdot 1
$$

In backprop notation, if $ \bar{a} = \frac{\partial z}{\partial a} $ is the gradient flowing from above, then:

$$
\bar{u} = \frac{\partial z}{\partial u} = \bar{a} \cdot (1 - \tanh^2(u))
$$

Then:

$$
\frac{\partial z}{\partial x_i} \mathrel{+}= \bar{u} \cdot w_i
$$

$$
\frac{\partial z}{\partial w_i} \mathrel{+}= \bar{u} \cdot x_i
$$

$$
\frac{\partial z}{\partial b} \mathrel{+}= \bar{u} \cdot 1 = \bar{u}
$$

This is exactly what is validated in **Test 3**: the neuron’s gradients w.r.t. inputs, weights, and bias match the analytical expressions above.

---

## 7. General reverse‑mode backprop algorithm

We now summarize backprop in an abstract form that directly mirrors what the engine does.

### 7.1 Forward pass

1. **Topologically construct the graph** as you perform operations:
   - Each node stores:
     - its value
     - references to its parents
     - a function that, given its gradient, distributes it to its parents (the local derivative rule).

2. **Compute the output** $ z $ by evaluating operations in forward order.

### 7.2 Backward pass (reverse mode)

Let $ z $ be the scalar output whose gradient we want.

1. Initialize all node gradients to zero:

$$
\bar{v} = 0 \quad \text{for all nodes } v
$$

2. Set the output gradient:

$$
\bar{z} = 1
$$

3. Traverse the nodes in **reverse topological order** (from outputs back to leaves). For each node $ v $:

   - Let its gradient be $ \bar{v} = \frac{\partial z}{\partial v} $  
   - For each parent $ p $ of $ v $, apply the local derivative rule:

$$
     \bar{p} \mathrel{+}= \bar{v} \cdot \frac{\partial v}{\partial p}
     $$

   where $ \frac{\partial v}{\partial p} $ may be a scalar, a vector, or an elementwise expression depending on the operation.

In the `PopulationNode` context:

- Each node is a population (vector) of scalars  
- Gradients are population‑wise, with elementwise rules for `+`, `*`, and `tanh`  
- Reductions like `sum` broadcast gradient back to all components  
- The implementation follows the exact scalar and vector rules derived above, applied per population unit.

---
## 8. Common Pitfalls in Autodiff
1. **Forgetting to reset gradients between backward passes
Gradients accumulate by design** 
    - If you don’t clear them, you’ll get incorrect results on the next backward pass.

2. **Confusing local derivatives with full gradients
Local derivatives describe how one operation behaves**

    - Gradients describe how the final output changes.
Backprop always multiplies them together.

3. **Using = instead of += in backward functions
Variables can influence the output through multiple paths**
    - Their gradients must accumulate, not overwrite.

4. **Misunderstanding reductions like sum()
sum() is not elementwise.**
    - It’s one scalar depending on all inputs, so the incoming gradient is broadcast to every element.

5. **Assuming elementwise ops need full Jacobians
Elementwise ops have diagonal Jacobians**
    - Backprop reduces to simple componentwise multiplication.

6. **Not storing enough information during the forward pass
Backprop requires:**
    - parents
    - output value
    - local derivative rule
If you don’t store these, you can’t compute gradients later.

7. **Incorrect topological order**
    - Backprop must traverse nodes in reverse creation order.
If the order is wrong, gradients flow incorrectly.

8. **Forgetting that gradients sum across multiple paths**
    - If a variable affects the output through multiple routes, its gradient is the sum of all contributions.

9. **Confusing broadcasting with vector–vector Jacobians**
    - Broadcasting repeats values but does not mix components.
It stays elementwise.

10. **Not testing with simple scalar functions first**
    - Before testing vector ops, verify:

    - x + y

    - x * y

    - tanh(x)

    - sum(x)

If these fail, everything else will too.
## 9. Summary

- The **scalar chain rule** is the foundation of backprop.  
- **Computation graphs** encode dependencies between intermediate results.  
- **Jacobian‑based chain rules** describe how gradients propagate in the vector case.  
- For `PopulationNode`, most operations are **elementwise**, which makes Jacobians diagonal and simplifies gradient flow.  
- **Reductions** like `sum` broadcast scalar gradients back to all components.  
- **Nonlinearities** like `tanh` have simple, well‑known derivatives that apply population‑wise.  
- A **neuron** with $ a = \tanh(w \cdot x + b) $ is a composition of dot product, addition, and nonlinearity; its gradients follow directly from the chain rule.  
- The **reverse‑mode algorithm** (backprop) is: initialize output gradient to 1, then walk the graph backward, applying local derivative rules and accumulating gradients in parents.

This document provides the mathematical specification that the `PopulationNode` engine implements. Any deviation between code and these derivations should be treated as a bug in the implementation.

