# Optimization and Learning Dynamics (from Scratch)
---
A research-driven codebase that rebuilds **learning mechanics from first principles** (NumPy only): autodiff, backprop, optimization dynamics, and curvature-based stability.  
This repo is designed to answer one question clearly:

> **Do I understand how learning works under the hood, mathematically and mechanistically, not just how to use a framework?**

This repo rebuilds learning from first principles, from credit assignment to discrete‑time learning dynamics to curvature and stability to continuous‑time gradient flow.

---
## Why does this matter?
A rigorous understanding of learning as a dynamical system is foundational to the research directions I intend to pursue at the [2026 Telluride Neuromorphic AI Workshop](https://www.neuropac.info/event/2026-telluride-neuromorphic-ai-workshop/), my MSc thesis in Reinforcement Learning and Embodied AI, and will lead to a PhD focused on intelligent adaptive systems. Optimization governs how agents update internal representations, stabilize behavior, and integrate information over time by studying learning from first principles. redit assignment, discrete‑time updates, curvature, and continuous‑time gradient flow. This project develops the mathematical framework for analyzing and designing learning rules that are stable, interpretable, and biologically relevant. This perspective is central to modern work in neuromorphic robotics, multimodal learning, and the control of embodied agents.

---
| Stage | Theme | Key Ideas |
|-------|--------|-----------|
| **1** | Credit Assignment (Autodiff) | Computation graph → reverse-mode autodiff → gradient flow |
| **2** | Discrete-Time Learning (GD) | θ_{t+1} = θ_t - η ∇L(θ_t); stability from step size & Jacobian |
| **3** | Geometry & Optimizers | Hessian eigenvalues → curvature → conditioning → momentum dynamics |
| **4** | Continuous-Time Learning | GD = Euler discretization of ẋ = -∇L(x); continuous vs discrete behavior |


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
---
### Stage 1: How does credit flow?
- Math note: [backprop and autodiff engine](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/math_notes/learning_as_dynamical_systems/stage_1_backprop_derivation.md)
- Notebook: [Experiments and Observations](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/notebooks/phase_1_gradient_population.ipynb)

**Core idea:** Learning begins with credit assignment: how does a system know which internal components contributed to an error?
  
**Autodiff, computation graphs, and population-coded signals**

$$ \frac{\partial L}{\partial x_i} = \sum_{j \in \text{children}(i)} \frac{\partial L}{\partial x_j} \frac{\partial x_j}{\partial x_i} $$
 
Autodiff computes gradients by applying the chain rule backward through a graph, accumulating contributions from all downstream nodes.

Built this from first principles:
- A minimal Tensor class that stores values and gradients
- A computation graph where each node records its parents
- Reverse‑mode autodiff that walks the graph backward
- Hand‑written backward passes that reveal how gradients propagate
- Early experiments with population-coded signals (distributed representations)

**What this stage teaches:**
- Every computation is a node in a graph
- Backpropagation is just the chain rule applied in reverse
- Gradients are signals flowing backward through a network
- Population-coded representations allow gradients to be shared across many units
- Credit assignment is the foundation of all learning systems

**Summary:** Stage 1 shows that learning begins with information flow: how errors propagate backward through a structured computational graph.

---

### Stage 2: How does learning unfold in time?
- Math notes:[Gradient Descent](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/math_notes/learning_as_dynamical_systems/stage_2_1_gradient_descent.md), [Continuous Vs Discrete systems](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/math_notes/learning_as_dynamical_systems/stage_2_3_continuous_vs_discrete.md)
- Notebook: [Experiments and Observations](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/notebooks/phase_2_gradient_Inestability.ipynb%3AZone.Identifier), [Learning stability and dynamics](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/notebooks/phase_2_learning_stability_dynamics.ipynb) 
**Learning = discrete-time dynamics**
Core insight:
- *Correct gradients do not guarantee learning.*
- Learning is not just “compute gradient → update parameters.” It is a dynamical system evolving over time.
- Gradient descent defines a **state update rule**; stability, symmetry, and step size determine outcomes.

What I learned:
- GD is a **difference equation**:  

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

- Step size controls **stability regimes** (convergent/oscillatory/divergent)
- Instability is predictable via the **Jacobian** (linearized dynamics)
- Symmetry can trap learning; **symmetry breaking** enables specialization

**The key conceptual leap:**  
When deriving GD from a Taylor expansion, I implicitly discovered:
- GD is Forward Euler
- Forward Euler approximates a continuous ODE
- Discretization introduces artifacts (oscillation, overshoot)
- Stability is no longer guaranteed
- Step size becomes a numerical stability parameter, not just a “learning rate.”

**Summary**: Stage 2 shows that learning is a discrete-time dynamical system whose stability depends on step size, symmetry, and local linearization.

---

### Stage 3: How do geometry & optimizers shape learning?
- Math notes:[Optimization, Geometry and Hessian, Eigen Values](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/math_notes/learning_as_dynamical_systems/stage_3_optimization_geometry.md)
- Notebook: [Experiments and Observations](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/notebooks/phase_3_optimization_geometry_dynamcal_system.ipynb)
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

**Summary**: By analyzing the Hessian, we can predict when gradient descent will converge, oscillate, or diverge based on the eigenvalues. Introducing momentum turns the update into a second‑order dynamical system, mathematically equivalent to a damped harmonic oscillator. The resulting quadratic characteristic equation allows us to classify the behavior as overdamped, critically damped, or underdamped, providing a complete picture of stability for momentum‑based learning.

---

### Stage 4: How does learning become a continuous dynamical system?
- Math notes:[Continuous vs Discrete](https://github.com/SrEntropy/optimization-and-learning-dynamics/blob/main/math_notes/learning_as_dynamical_systems/stage_2_3_continuous_vs_discrete.md)
- Notebook: [Experiments and Observations]()
Gradient flow, ODEs, and the bridge to NeuroAI
**Core idea:**  

Gradient descent is not the “true” learning process; it is a **numerical approximation** of a smooth, continuous‑time system.
Understanding this limit reveals the ideal behavior that discrete updates aim to approximate.

**What you discovered:**
- Gradient descent is **Forward Euler** applied to the ODE

$$\theta = -\nabla L(\theta_t)$$
  
- The continuous system (“gradient flow”) is perfectly smooth:
  - no oscillation
  - no overshoot
   -no step‑size instability

- Discrete GD introduces artifacts because it samples the ODE with a finite step size:
  - oscillations
  - divergence
  - numerical instability

- ODE solvers (Runge–Kutta, adaptive methods) reveal the “ideal” trajectory that GD approximates
- This viewpoint connects directly to:
  - biological learning (continuous adaptation)
  - neuromorphic systems
  - Neural ODEs
  - stability theory and Lyapunov analysis

**Summary:** Stage 4 reframes learning as a continuous physical process, with gradient descent appearing as a discretized approximation whose stability and behavior depend on numerical integration.


## Repo structure (high level)

