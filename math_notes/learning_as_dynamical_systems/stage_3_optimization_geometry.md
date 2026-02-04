# Stage 3: Learning & Optimization as a Dynamical System

Read: Notebook
These notes are part of **Stage 3 (of 4)** of the repo **“Optimization and Learning Dynamics”**.  

Goal: **own the math** behind why gradient-based learning behaves like a **discrete-time dynamical system**, why it can **zigzag**, **oscillate**, or **diverge**, and how **curvature** (the Hessian) predicts stability.

If you learn one thing: **quadratic losses are the perfect “physics lab”** for understanding optimization dynamics. They are simple enough to solve exactly, but rich enough to show the key behaviors that also appear (locally) in deep learning.

---

## Contents

1. [Why quadratics matter](#why-quadratics-matter)
2. [Derivatives: slope vs curvature](#derivatives-slope-vs-curvature)
3. [Gradient in many dimensions](#gradient-in-many-dimensions)
4. [Hessian and curvature](#hessian-and-curvature)
5. [Eigenvectors & eigenvalues (intuition first)](#eigenvectors--eigenvalues-intuition-first)
6. [Quadratic loss: gradient and Hessian](#quadratic-loss-gradient-and-hessian)
7. [Gradient descent as a discrete-time dynamical system](#gradient-descent-as-a-discrete-time-dynamical-system)
8. [Ill-conditioning, condition number, anisotropy, zigzag](#ill-conditioning-condition-number-anisotropy-zigzag)
9. [Discrete-time stability: oscillation vs divergence](#discrete-time-stability-oscillation-vs-divergence)
10. [Momentum dynamics: second-order discrete-time system](#momentum-dynamics-second-order-discrete-time-system)
11. [Predicting failure via curvature](#predicting-failure-via-curvature)
12. [Exercises (from first principles)](#exercises-from-first-principles)
13. [Cheat sheet / recap](#cheat-sheet--recap)

---

## Why quadratics matter

Most smooth losses can be approximated **locally** by a quadratic using a Taylor expansion.  
So if you understand  on quadratics, you understand the *localoptimization* behavior of gradient descent near many points in real training.

Quadratics also let us:
- compute the **gradient** and **Hessian** exactly,
- analyze stability precisely,
- connect geometry (ellipses/valleys) to trajectories (zigzag/oscillation).

---

## Derivatives: slope vs curvature

Start in **1D**: a function is a curve.

### First derivative (slope)
For a function $f(x)$,
- $f'(x)$ measures the **slope** at $x$.
- If $f'(x) > 0$: moving right increases $f$.
- If $f'(x) < 0$: moving right decreases $f$.
- Gradient descent uses the slope to go downhill.

### Second derivative (curvature)
- $f''(x)$ measures **curvature**: how the slope changes.
- Large $f''(x)$ means the curve bends sharply (steep “walls”).
- Curvature controls **how aggressive your step size can be** without overshooting.

**Mental model:**  
- **first derivative** = “which direction do I go?”  
- **second derivative** = “how careful must I be stepping?”

---

## Gradient in many dimensions

In $\mathbb{R}^d$, your parameter vector is:

$$
x = (x_1, x_2, \dots, x_d)^\top
$$

### Partial derivatives
$$
\frac{\partial f}{\partial x_i}
$$
means “change only coordinate $x_i$ a tiny bit and see how $f$ changes.”

### Gradient definition
$$
\nabla f(x) =
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\vdots \\
\frac{\partial f}{\partial x_d}
\end{bmatrix}
$$

**Key geometric fact:**  
$\nabla f(x)$ points in the direction of **steepest increase**.  
So $-\nabla f(x)$ is the direction of **steepest decrease**.

### Gradient descent update
$$
x_{k+1} = x_k - \alpha \nabla f(x_k)
$$

**Definitions**
- $k$: iteration number (discrete time index)
- $\alpha>0$: step size (learning rate)

---

## Hessian and curvature

The Hessian is the “second derivative” in many dimensions.

### Hessian definition
$$
\nabla^2 f(x) =
\begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_d \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_d^2}
\end{bmatrix}
$$

**Interpretation**
- diagonal entries: curvature along each coordinate axis
- off-diagonals: how directions interact (the bowl is “tilted” / rotated)

**Why it matters**
- the Hessian tells you **local curvature**,
- local curvature tells you **stability limits** for discrete updates.

---

## Eigenvectors & eigenvalues (intuition first)

A matrix $A$ transforms vectors: input $v$ → output $Av$.

Usually, a transformation changes both:
- **direction**, and
- **length**.

### Eigenvector definition
A nonzero vector $v\neq 0$ is an eigenvector of $A$ if:

$$
Av = \lambda v
$$

**Meaning**
- direction stays the same (still on the same line/span as $v$)
- the vector is scaled by $\lambda$ (and flipped if $\lambda<0$)

### Why eigenvectors matter here
For quadratic losses, the Hessian matrix $A$ defines curvature.  
Eigenvectors give the special directions where curvature is “pure” and independent.  
Eigenvalues $\lambda_i$ tell the curvature strength in those directions.

---

## Quadratic loss: gradient and Hessian

A general quadratic in $\mathbb{R}^d$:

$$
f(x)=\frac{1}{2}x^\top A x - b^\top x + c
$$

**Definitions**
- $x\in\mathbb{R}^d$: parameter vector
- $A\in\mathbb{R}^{d\times d}$: symmetric matrix (assume $A=A^\top$)
- $b\in\mathbb{R}^d$, $c\in\mathbb{R}$: constants
- $x^\top A x$ is a scalar

### Gradient
$$
\nabla f(x) = Ax - b
$$

### Hessian
$$
\nabla^2 f(x) = A
$$

So: **quadratic curvature is constant everywhere**.

### Minimizer (optimum)
Solve $\nabla f(x^\star)=0$:
$$
Ax^\star - b = 0 \quad \Rightarrow \quad x^\star = A^{-1}b
$$
(assuming $A$ is invertible)

---

## Gradient descent as a discrete-time dynamical system

Gradient descent:

$$
x_{k+1} = x_k - \alpha (Ax_k - b)
$$

Define the **error** relative to optimum:

$$
e_k := x_k - x^\star
$$

Because $b = Ax^\star$, we get:

$$
Ax_k - b = A(x_k - x^\star) = A e_k
$$

So the update becomes:

$$
e_{k+1} = e_k - \alpha A e_k = (I - \alpha A)e_k
$$

This is a **linear discrete-time dynamical system**:

- state: $e_k$
- update matrix: $M = I - \alpha A$

---

## Ill-conditioning, condition number, anisotropy, zigzag

### Anisotropy (direction-dependent scaling)
If curvature differs strongly by direction, the quadratic’s level sets become stretched ellipses (a narrow valley).

This happens when eigenvalues of $A$ vary widely.

### Condition number
Assume $A$ is **symmetric positive definite** (all eigenvalues $>0$).  
Define:

$$
\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

**Meaning**
- $\kappa \approx 1$: bowl is close to spherical → GD is efficient
- $\kappa \gg 1$: narrow valley → GD zigzags and slows down

### Zigzag intuition (2D)
Let:

$$
f(x,y) = \frac{1}{2}(\lambda_1 x^2 + \lambda_2 y^2), \quad \lambda_2 \gg \lambda_1
$$

Gradient:
$$
\nabla f(x,y) = (\lambda_1 x, \lambda_2 y)
$$

GD updates:
$$
x_{k+1} = (1-\alpha\lambda_1)x_k,\quad
y_{k+1} = (1-\alpha\lambda_2)y_k
$$

If $\alpha$ is chosen near the stability limit set by $\lambda_2$, then the $y$-component can flip sign each step (bouncing across the valley), while $x$ decays slowly → **zigzag**.

---

## Discrete-time stability: oscillation vs divergence

Diagonalize $A$ (symmetric case):

$$
A = Q\Lambda Q^\top
$$
- $Q$: orthonormal eigenvectors
- $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)$: eigenvalues

Change coordinates: $z_k := Q^\top e_k$. Then:

$$
z_{k+1} = (I - \alpha\Lambda)z_k
$$

This becomes independent 1D recurrences:

$$
z^{(i)}_{k+1} = (1-\alpha\lambda_i)\,z^{(i)}_k
$$

Let $r_i = 1-\alpha\lambda_i$.

### Per-direction behavior
- **convergent** if $|r_i|<1$
- **oscillatory but convergent** if $-1<r_i<0$ (sign flips each step)
- **divergent** if $|r_i|>1$

### Global GD stability condition
Need $|1-\alpha\lambda_i|<1$ for all $i$.  
For $\lambda_i>0$, this is equivalent to:

$$
0 < \alpha < \frac{2}{\lambda_{\max}}
$$

### When does oscillation start?
In direction $i$, oscillation occurs when $r_i<0$:

$$
1-\alpha\lambda_i < 0 \quad \Rightarrow \quad \alpha > \frac{1}{\lambda_i}
$$

So you can predict oscillations **from eigenvalues**.

---

## Key intuitions (FAQ style)

### Why does GD slow down in narrow valleys?

Because the loss surface has **high curvature** in one direction and **low curvature** in another.

- The step size (learning rate) is constrained by the **steepest** direction (largest Hessian eigenvalue).
- That forces **small steps overall** to avoid instability.
- Then progress along flat directions becomes extremely slow.

This mismatch produces **zig-zag motion** (bouncing across steep walls) and **poor convergence speed** (slow drift down the valley).

---

### How do eigenvalues of the Hessian control step size limits?

For a quadratic loss, stability of gradient descent requires:

$$
\eta < \frac{2}{\lambda_{\max}}
$$

where $\lambda_{\max}$ is the **largest eigenvalue** of the Hessian.

- If $\eta$ exceeds this bound, updates overshoot and can **oscillate** or **diverge**.
- Thus $\lambda_{\max}$ determines the **global** safe step size, even if other directions are gentle.

---

### Why does momentum help even when gradients are correct?

Because **correct gradients do not imply efficient dynamics**.

Momentum introduces **state memory** (a running velocity). It:
- accumulates velocity across steps,
- reduces back-and-forth motion in high-curvature directions,
- and speeds travel in low-curvature directions.

So it can **rebalance learning** across the geometry of the loss and reduce zig-zag behavior.

---

### What is the discrete-time view of momentum?

Momentum turns optimization into a **second-order discrete-time dynamical system**:

$$
v_{t+1} = \beta v_t - \eta \nabla L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t + v_{t+1}
$$

The system has internal state ($v_t$), so behavior depends on:
- the current gradient, and
- the **history** of motion.

This explains inertia, damping, and overshoot phenomena.

---

### Why does anisotropic scaling change learning speed?

Because gradient descent is **not invariant to parameterization**.

If you scale parameters differently across coordinates, you also rescale:
- gradient magnitudes, and
- curvature (Hessian eigenvalues),

in different ways. That changes the **effective step size per direction**.  
So identical “looking” gradients can produce different dynamics depending on how the loss is scaled along each axis.

---

### How does zig-zag trajectory relate to $\lambda_{\max}/\lambda_{\min}$?

The zig-zag severity increases with the **condition number**:

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

A large ratio means:
- one direction forces small steps (steep curvature),
- while other directions could tolerate much larger steps.

GD repeatedly overshoots across steep directions while making slow progress along flat ones, producing zig-zag trajectories.

---

### How can you predict GD failure by inspecting curvature?

By analyzing the Hessian spectrum:

- Large $\lambda_{\max}$ $\rightarrow$ strict step size limit
- High condition number $\kappa$ $\rightarrow$ slow convergence / zig-zag
- Mixed-sign eigenvalues $\rightarrow$ saddle directions and potential instability

If the learning rate violates curvature-imposed bounds or curvature is highly anisotropic, GD failure (oscillation, divergence, stagnation) is often predictable **before training begins**.


## Momentum dynamics: second-order discrete-time system

A common momentum update (heavy ball / Polyak form):

$$
x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta(x_k-x_{k-1})
$$

**Definitions**
- $\alpha$: step size
- $\beta\in[0,1)$: momentum coefficient
- $x_k-x_{k-1}$: velocity-like term

For quadratic centered at optimum ($\nabla f(x)=Ax$):

$$
x_{k+1} = (I-\alpha A + \beta I)x_k - \beta x_{k-1}
$$

In eigen-coordinates (per eigenvalue $\lambda$):

$$
z_{k+1} = (1-\alpha\lambda+\beta)z_k - \beta z_{k-1}
$$

This is **second-order** because it depends on $z_k$ and $z_{k-1}$.

### Characteristic equation (predict stability)
Try a solution $z_k=r^k$. Substitute:

$$
r^2 - (1-\alpha\lambda+\beta)r + \beta = 0
$$

For stability, both roots must satisfy $|r|<1$.  
Momentum can speed convergence and reduce zigzag, but aggressive $\alpha,\beta$ can create strong oscillations or divergence.

---

## Predicting failure via curvature

For quadratics, the largest curvature is $\lambda_{\max}(A)$.  
GD requires:

$$
\alpha < \frac{2}{\lambda_{\max}}
$$

In deep nets, curvature changes during training.  
If local curvature spikes (largest Hessian eigenvalue increases), a learning rate that used to be safe can become unstable → loss spikes/divergence.

**Rule of thumb**
- high curvature + large step = instability risk

---

## Exercises (from first principles)

### Exercise 1 — 1D slope and curvature
Let $f(x)=\tfrac12\lambda x^2$, $\lambda>0$.
1. Compute $f'(x)$ and $f''(x)$.
2. If $\lambda$ increases, what happens to curvature?
3. Write GD: $x_{k+1}=x_k-\alpha f'(x_k)$. Simplify the recursion.

---

### Exercise 2 — Gradient in 2D
Let $f(x,y)=x^2+10y^2$.
1. Compute $\nabla f(x,y)$.
2. Evaluate at $(1,1)$.
3. Which direction is “steeper”? Explain using the gradient.

---

### Exercise 3 — Hessian in 2D
For $f(x,y)=x^2+10y^2$:
1. Compute the Hessian $\nabla^2 f(x,y)$.
2. Is it constant or does it depend on $(x,y)$?
3. Interpret the diagonal entries.

---

### Exercise 4 — Eigenvector intuition
Let $A=\begin{bmatrix}2&0\\0&10\end{bmatrix}$.
1. Verify $A(1,0)=2(1,0)$ and $A(0,1)=10(0,1)$.
2. Compute $A(1,1)$. Does it stay in the span of $(1,1)$?
3. Explain why only some directions are eigenvectors.

---

### Exercise 5 — Zigzag vs monotone in 2D
Let $f(x,y)=\tfrac12(2x^2+10y^2)$.  
GD updates:
$$
x_{k+1}=(1-2\alpha)x_k,\quad y_{k+1}=(1-10\alpha)y_k
$$
Start $x_0=5,y_0=1$.
1. With $\alpha=0.05$, compute $(x_1,y_1)$ and $(x_2,y_2)$. Does $y$ oscillate?
2. With $\alpha=0.15$, repeat. What changes?
3. With $\alpha=0.25$, what happens? (Hint: check $1-10\alpha$.)

---

### Exercise 6 — Condition number and safe step size
For $A=\mathrm{diag}(2,10)$:
1. $\lambda_{\max}, \lambda_{\min}$, and $\kappa$?
2. What is the GD stability upper bound $2/\lambda_{\max}$?
3. Explain why $x$-direction converges slower than $y$-direction for safe $\alpha$.

Compare to $A=\mathrm{diag}(5,6)$. Which is better conditioned?

---

### Exercise 7 — Predict oscillation and divergence from eigenvalues
Suppose eigenvalues are $\lambda_1=2,\lambda_2=10$.  
For each $\alpha$, compute multipliers:
$$
r_1=1-2\alpha,\quad r_2=1-10\alpha
$$
Classify each direction: monotone / oscillatory / divergent.
- $\alpha=0.05$
- $\alpha=0.12$
- $\alpha=0.25$

---

### Exercise 8 — Momentum recurrence (1D)
Let $f(x)=\tfrac12\lambda x^2$ with $\lambda=10$.  
Momentum:
$$
x_{k+1}=(1-\alpha\lambda+\beta)x_k-\beta x_{k-1}
$$
Choose $\alpha=0.1,\beta=0.8$, start $x_0=1,x_1=1$.
1. Compute $x_2,x_3,x_4$.
2. Is it shrinking to 0 or growing?
3. Try $\beta=0.3$. Compare behavior.

---

### Exercise 9 — Curvature safety check
If an estimate of the largest curvature is $\lambda_{\max}\approx 200$:
1. What is the safe GD bound $2/\lambda_{\max}$?
2. Is $\alpha=0.02$ safe by this rule?

---

## Cheat sheet / recap

### Quadratic loss
$$
f(x)=\tfrac12 x^\top A x - b^\top x + c
$$
$$
\nabla f(x)=Ax-b,\quad \nabla^2 f(x)=A
$$

### Error dynamics
Let $e_k=x_k-x^\star$. Then:
$$
e_{k+1}=(I-\alpha A)e_k
$$

### Eigen-decomposition (symmetric $A$)
$$
A=Q\Lambda Q^\top
$$
Per eigen-direction:
$$
z^{(i)}_{k+1}=(1-\alpha\lambda_i)z^{(i)}_k
$$

### GD stability
$$
0<\alpha<\frac{2}{\lambda_{\max}}
$$

### Condition number
$$
\kappa=\frac{\lambda_{\max}}{\lambda_{\min}}
$$
Large $\kappa$ → anisotropy → zigzag + slow convergence.

### Momentum (1D per eigenvalue $\lambda$)
$$
z_{k+1}=(1-\alpha\lambda+\beta)z_k-\beta z_{k-1}
$$
Characteristic equation:
$$
r^2-(1-\alpha\lambda+\beta)r+\beta=0
$$
Stability requires $|r|<1$ for both roots.

---

**Next step for the repo:** implement small experiments that visualize these dynamics:
- 2D trajectories for different $\alpha$ and $\kappa$
- oscillation onset at $\alpha>1/\lambda_i$
- divergence at $\alpha>2/\lambda_{\max}$
- momentum acceleration vs instability regions
