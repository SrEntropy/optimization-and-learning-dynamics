# Gradient Descent: From First Principles to Dynamics

This note derives gradient descent (GD) step by step from basic calculus principles. The note, continuous_vs_discrete.md, further this study by analyzing GD's behavior in discrete and continuous time. Especially in the context of deep learning, where **vanishing/exploding gradients** and **stability** become critical.

Treat optimization as a dynamical process: how parameters evolve in **time** to minimize a loss, cost, error, etc.

## 1. Deriving Gradient Descent from First Principles
### a. Problem setup

Consider a differentiable scalar loss function $L: \mathbb{R}^n \to \mathbb{R}$, $L(\theta)$, where $\theta \in \mathbb{R}^n$ are the parameters (weights, biases, etc.) to adjust.

Goal: find $\theta*$ such that $L(\theta^*) = \min_\theta L(\theta)$ (find $\theta$ that minimizes the Loss function).
Starting from $\theta_0$ build a sequence: $\theta_0, \theta_1, \theta_2, ...$ such that $L(\theta_{t + 1}) <= L(\theta)$ and ideally $\theta_t$ approaches a (local) minimum.

- Note: GD is just a principled way to choose each update $\theta_{t + 1}$ from $\theta_t$
### b. First Principle: Local Approximation Via First-order Taylor Expansion

At current point $\theta_t$, for a small displacement $\Delta\theta$, the first-order Taylor expansion gives:

$$
L(\theta_t + \Delta\theta) \approx L(\theta_t) + \nabla L(\theta_t)^\top \Delta\theta + \mathcal{O}(\|\Delta\theta\|^2)
$$

To decrease $L$ the most with a small step of fixed length $\eta$, minimize the linear term while constraining $\|\Delta\theta\| \approx \eta$ (the total distance of movement from the starting point  is approx. $\eta$).

Terms:
- $\theta_t$:  Current parameter
- $\Delta\theta$: A small change/step away from the starting point $\theta_t$.
- $L(\theta_t + \Delta\theta)$: Approx. value of the function at the new, shifted point.
- $\nabla L(\theta_t)$: Gradient of all partial derivatives at point $\theta_t$ 
- $\nabla L(\theta_t)^\top\Delta\theta$: Directional change (Dot product  of the gradient and displacement vector $\Delta\theta$) of the approx. change in Loss.

So, the approx. change in Loss is $\Delta\theta) \approx \nabla_{\theta} L(\theta_t)^\top\Delta\theta$

### c. Descent Condition and the GD Update Rule
- Goal: Make the loss go downhill, not uphill.

$L(\theta_t + \Delta\theta) < L(\theta_t)$

Using Linear Approx., $L(\theta)+\nabla_{\theta} L(\theta_t)^\top\Delta\theta < L(\theta)$ => $\nabla _{\theta}L(\theta_t)^\top\Delta\theta < 0$; So, the descent condition is met. Therefore, the dot product between the gradient  and the step must be negative to go downhill. 

The direction that **maximizes the decrease** in the linear approximation is the **negative gradient** direction:

So,
$$\Delta\theta = -\eta \frac{\nabla L(\theta_t)}{\|\nabla L(\theta_t)\|} $$
Normalized steepest descent; when a vector is divided by its own length (norm). 
This means stripping its magnitude/speed to conserve only the direction.

Since the gradient points up (increasing error), multiply by negative one to point down (decreasing error). So, the gradient is descending. So the descent condition is satisfy by

 $$\Delta\theta_t = - \eta\nabla_{\theta} L(\theta_t)$$

 where:
 -  $\Delta\theta_t$ is the actual update step
 -  $\eta > 0$ is the learning rate (step size)


$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

This is the **gradient descent update rule**.

Why does this make sense?

- $-\nabla L(\theta_t)$ is the **direction of steepest descent** (by definition of the gradient).

This checks the descent condition:
$$\nabla_{\theta} L(\theta_t)^\top\Delta\theta_t = \nabla_{\theta} L(\theta_t)^\top(-\eta\nabla_{\theta}L(\theta_t) =-\eta||\nabla_{\theta}L(\theta_t)^\top||^2<0  $$
 

- $\eta$ controls trade-off: 
    - Too little → slow progress
    - Medium → Oscillate by converges
    - Too large → overshoot/divergence.

whenever $\nabla_{\theta} L(\theta_t)$ not=$0$, for sufficiently small $\eta$, the update is guaranteed to decrease the $Loss Function$. 

Therefore, the GD update rule is

$$ 
\theta_{[t+1)} = \theta_t + \Delta\theta_t = \theta_t-\eta\nabla_\theta L(\theta_t)
$$ 


### 2 For Vanishing, Exploding, and gradient stability, see [vanishing.md](https://github.com/SrEntropy/optimization-and-learning-dynamics/edit/main/math_notes/vanishing.md) file.
