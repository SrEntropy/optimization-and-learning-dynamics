# Gradient Descent: From First Principles to Dynamics

This note derives gradient descent (GD) step by step from basic calculus principles, the note continuous_vs_discrete.md further this study by analyzing GD's behavior in discrete and continuous time. Especially in the context of deep learning where **vanishing/exploding gradients** and **stability** become critical.

Treat optimization as a dynamical process: how parameters evolve in **time** to minimize a loss, cost, error, etc.

## 1. Deriving Gradient Descent from First Principles
### a. Problem setup

Consider a differentiable scalar loss function $L: \mathbb{R}^n \to \mathbb{R}$, $L(\theta)$, where $\theta \in \mathbb{R}^n$ are the parameters (weights, biases, etc.) to adjust.

Goal: find $\theta^*$ such that $L(\theta^*) = \min_\theta L(\theta)$ (find $\theta$ that minimizes the Loss function).
Starting from $\theta_0$ build a sequence: $\theta_0, \theta_1, \theta_2, ...$ such that $L(\theta_{t + 1}) <= L(\theta)$ and ideally $\theta_t$ approaches a (local) minimum.

- Note: GD is a just principled way to choose each update $\theta_{t + 1}$ from $\theta_t$
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
- Goal: Make the loss go downhill not uphill.

$L(\theta_t + \Delta\theta) < L(\theta_t)$

Using Linear Approx., $L(\theta)+\nabla_{\theta} L(\theta_t)^\top\Delta\theta < L(\theta)$ => $\nabla _{\theta}L(\theta_t)^\top\Delta\theta < 0$; So, the descent condition is met. Therfore, the dot product between the gradient  and the step must be negative to go downhill. 

The direction that **maximizes the decrease** in the linear approximation is the **negative gradient** direction:

So,
$$
\Delta\theta = -\eta \frac{\nabla L(\theta_t)}{\|\nabla L(\theta_t)\|} $$
Normalized steepest descent; when a vector is divided by its own lenght (norm). 
This means, strip its magnitude/speed to conserve only the direction.

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

This check the descent condition:
$$\nabla_{\theta} L(\theta_t)^\top\Delta\theta_t = \nabla_{\theta} L(\theta_t)^\top(-\eta\nabla_{\theta}L(\theta_t) =-\eta||\nabla_{\theta}L(\theta_t)^\top||^2<0  $$
 

- $\eta$ controls trade-off: 
    - Too small → slow progress
    - Medium → Oscillate by converges
    - Too large → overshoot / divergence.

whenever $\nabla_{\theta} L(\theta_t)$ not=$0$, for sufficiently small $\eta$, the update is guaranteed to decrease the $Loss Function$. 

Therefore, the GD update rule is $$ \theta_{[t+1)} = \theta_t + \Delta\theta_t = \theta_t-\eta\nabla_\theta L(\theta_t)$$ 


## 2. Vanishing and Exploding Gradients

In deep neural networks, $L(\theta)$ is composed through many layers, so $\nabla L(\theta)$ involves the product of many Jacobian-like terms via the chain rule (backpropagation).

Consider a simple feedforward net with $L$ layers, each with roughly similar weight matrices. The gradient w.r.t. early-layer weights scales like the product of $L$ terms.

Typical behaviors:

- **Vanishing gradients** — if layer Jacobians have spectral norm < 1 (e.g., small weights, saturating activations like sigmoid), repeated multiplication → gradient norm → 0 exponentially fast → early layers barely update.

- **Exploding gradients** — if spectral norm > 1, gradients grow exponentially → huge updates → numerical instability / NaNs.

These are direct consequences of the **discrete multiplicative dynamics** of backprop through depth — analogous to instability in linear discrete-time systems.

Mitigations (historical & modern): proper initialization (Xavier/He), batch norm, residual connections, better activations (ReLU family), gradient clipping, etc.

## 3. Stability of Discrete Updates

Consider the **linear-quadratic** case for stability insight (common toy model for local behavior near a minimum).

Assume $L(\theta) = \frac{1}{2} \theta^\top A \theta$ with $A \succ 0$ symmetric positive definite (strongly convex quadratic bowl).

Then $\nabla L(\theta) = A \theta$, and GD becomes:

$$
\theta_{k+1} = \theta_k - \eta A \theta_k = (I - \eta A) \theta_k
$$

This is a linear discrete-time system $\theta_{k+1} = M \theta_k$ with $M = I - \eta A$.

**Asymptotic stability** requires all eigenvalues of $M$ to have magnitude < 1.

Let $\lambda_{\max}, \lambda_{\min}$ be the extreme eigenvalues of $A$ (curvature).

The eigenvalues of $M$ are $1 - \eta \lambda_i$.

For convergence to 0 (the minimum):

$$
\max_i |1 - \eta \lambda_i| < 1
$$

This holds if

$$
0 < \eta < \frac{2}{\lambda_{\max}(A)}
$$

- When $\eta < 1/\lambda_{\max}$: monotonic decay (overdamped-like).
- When $1/\lambda_{\max} < \eta < 2/\lambda_{\max}$: oscillatory convergence (underdamped).
- When $\eta > 2/\lambda_{\max}$: divergence (explosion).

The **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ controls worst-case speed: slow modes along small-curvature directions.

In deep nets, $\lambda_{\max}$ can be very large (sharp directions) while $\lambda_{\min}$ is tiny (flat directions) → poor conditioning → vanishing-like slow progress in flat basins.
