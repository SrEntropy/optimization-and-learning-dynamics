# Optimization and Learning Dynamics (from Scratch)

A research-driven codebase that rebuilds **learning mechanics from first principles** (NumPy only): autodiff, backprop, optimization dynamics, and curvature-based stability.  
This repo is designed to answer one question clearly:

> **Do I understand how learning works under the hood — mathematically and mechanistically — not just how to use a framework?**

---

## Why this repo exists

Modern ML frameworks hide the physics of training. This project exposes it.

What I’m proving here:
- I can **derive** learning rules (not copy them)
- I can **analyze** learning as a **dynamical system**
- I can connect **loss geometry (Hessian spectrum)** to **training behavior**
- I can write clean, auditable research code and explain it

---

## Project roadmap (4 stages)

### Stage 1: How does credit flow?
**Autodiff + population signals**
- Minimal `Tensor` class + computation graph
- Reverse-mode autodiff (handwritten backward passes)
- Early experiments with distributed / population-coded signals

### Stage 2: How does learning unfold in time?
**Learning = discrete-time dynamics**
Core insight:
- *Correct gradients do not guarantee learning.*
- Gradient descent defines a **state update rule**; stability, symmetry, and step size determine outcomes.

What I learned:
- GD is a **difference equation**:  
  $$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$
- Step size controls **stability regimes** (convergent / oscillatory / divergent)
- Instability is predictable via the **Jacobian** (linearized dynamics)
- Symmetry can trap learning; **symmetry breaking** enables specialization

### Stage 3: How do geometry & optimizers shape learning?
**Curvature, conditioning, zig-zag, momentum**
Unified picture:
> **Gradient descent is a discrete-time dynamical system governed by curvature (Hessian eigenvalues) and parameterization.**

Key results:
- The Hessian spectrum predicts stability through:
  $$\eta < \frac{2}{\lambda_{\max}}$$
- Narrow valleys (high condition number) cause slow progress + zig-zag:
  $$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$
- Momentum introduces internal state (velocity), turning learning into a **second-order** discrete-time system:
  $$v_{t+1} = \beta v_t - \eta \nabla L(\theta_t),\quad \theta_{t+1} = \theta_t + v_{t+1}$$

**Stage 3 notes:** see `math_notes/stage3_math_notes_with_dollars_v2.md`

### Stage 4: How does learning become a continuous dynamical system?
**Gradient flow + NeuroAI bridge**
- Interpret GD as Euler discretization of gradient flow:
  $$\dot{\theta} = -\nabla L(\theta)$$
- Simulate continuous-time dynamics with ODE solvers
- Compare discrete vs continuous behavior
- Bridge toward biological / neuromorphic learning perspectives

---

## Repo structure (high level)

