# Optimization and learning-dynamics
A research‑driven exploration of how learning emerges from optimization, signal flow, and dynamical systems implemented entirely from first principles using NumPy.

This project demonstrates mathematical maturity, engineering clarity, and the ability to reason about learning at a mechanistic level.
## Project Vision
Modern ML frameworks hide the mechanics of learning.
This repository rebuilds them from scratch to answer four foundational questions:

#
## Stage 1:  How does credit flow?
### Autodiff + population signals
- Construct a minimal Tensor class
- Build computation graphs
- Implement reverse‑mode autodiff
- Explore distributed credit assignment through population‑coded activations

#

## Phasee 2:  What I Learned
### Core Insight
- Learning in neural networks is not explained by gradients alone. It is governed by the dynamics of the update rule, which turns optimization into a discrete-time dynamical system. Correct gradients do not guarantee learning; stability, symmetry, and step size critically shape outcomes.

### 1. Gradients Can Be Correct While Learning Fails

- We learned that gradients can be locally correct yet still fail to produce learning because optimization unfolds over time through repeated updates. Gradient descent defines a state evolution process, not a single optimization step. If the update dynamics are unstable, oscillatory, or poorly conditioned, learning fails despite valid gradients.

**Key takeaway:** Learning failure is often a dynamical failure, not a gradient computation error.

### 2. Gradient Descent Is a Discrete-Time Dynamical System

- Gradient descent is a difference equation:

$$
𝜃_{𝑡+1}=𝜃_𝑡−𝜂∇𝐿(𝜃_𝑡)
$$

This recursive rule evolves parameters over discrete time steps, approximating continuous gradient flow only when the step size is sufficiently small.

**Key takeaway:** Training trajectories must be analyzed using tools from dynamical systems, not just optimization theory.

### 3. Step Size Controls Stability, Not Just Speed

- The learning rate determines whether updates:
    - converge smoothly
    - oscillate
    - diverge
    - explode.

Large step sizes break the approximation to continuous gradient flow and can push the system into unstable regimes.

**Key takeaway:** Step size defines the stability regime of learning dynamics, not merely training speed.

4. Instability Has a Precise Mathematical Meaning

- Instability occurs when small perturbations grow over time. Formally, this happens when the Jacobian of the update map has eigenvalues with magnitude ≥ 1.

**Key takeaway:** Instability is diagnosable and predictable using linearized dynamics.

### 5. Population Symmetry Emerges from Architecture and Initialization

- When units are:
    - identically initialized, governed by the same update rules, and architecturally interchangeable, the system becomes permutation-equivariant, causing units to evolve identically.

**Key takeaway:** Symmetry is a structural property of the model and its dynamics, not an accident.

### 6. Symmetry Breaking Enables Learning

- Symmetry breaks when gradients differ across units. This can arise from:
    - random initialization
    - noise
    - architectural bottlenecks,
    - unstable Jacobian modes.

Once symmetry breaks, units specialize and learning becomes expressive.

**Key takeaway:** Learning often requires symmetry breaking.

### 7. Parameters and Tensors Play Fundamentally Different Roles

- Parameters are state variables of the learning dynamical system.
- Tensors are intermediate values used for computation.

This separation clarifies why only parameters accumulate history and evolve across time.

**Key takeaway:** Learning dynamics act on parameters, not on transient computational values.

### 8. Failure Is Informative

- Training failures expose:

    - architectural limitations,
    - unstable regimes
    - symmetry traps
    - poor dynamical conditioning.

Rather than being discarded, failures provide diagnostic insight into why learning is impossible under certain conditions.

**Key takeaway:** Failure reveals the structure and constraints of the learning system.

### Week 2 Summary

By the end of Week 2, we shifted from viewing training as “gradient optimization” to understanding it as dynamical system evolution. This reframing explains instability, symmetry, failure modes, and the central role of step size—laying the foundation for deeper analysis of learning dynamics in neural and biologically inspired systems.

### Stage 3: How do geometry & optimizers shape learning?
#### Loss surfaces + momentum
- Visualize 1D/2D loss landscapes
- Examine curvature, ridges, and basins
- Implement momentum, RMSProp, Adam
- Show how geometry influences optimizer trajectories

#


### Stage 4 — How does learning become a dynamical system?
#### Gradient flow + NeuroAI bridge
- Treat learning as an ODE
- Simulate continuous‑time neural dynamics
- Connect optimization principles to biological learning
- Explore population‑based representations as dynamical systems

### What This Repository Demonstrates
This project shows the ability to:
- derive learning rules mathematically
- reason about stability and convergence
- connect discrete optimization to continuous dynamics
- visualize and interpret loss geometry
- design clean, research‑grade software
- communicate complex ideas clearly and rigorously

These are core competencies for graduate‑level ML, robotics, and NeuroAI research.

### Core Implementations
- Minimal Tensor class
- Reverse‑mode autograd engine
- Backpropagation from first principles
- Gradient descent + momentum‑based optimizers
- Loss landscape visualization tools
- Continuous‑time gradient flow simulators
- Simple ODE‑based neural dynamics

Experiments:
- XOR from scratch
- Vanishing & exploding gradients
- Stability of training
- Gradient flow vs. gradient descent
- Population‑coded activations & credit assignment

Each experiment is designed to reveal a specific phenomenon in learning dynamics.
- A research‑driven exploration of how learning emerges from optimization, signal flow, and dynamical systems — implemented entirely from first principles using NumPy.

This project demonstrates mathematical maturity, engineering clarity, and the ability to reason about learning at a mechanistic level.

## Motivation: 
- Can I derive learning rules?
- Can I reason about stability
- Can I connect discrete optimization to continuous dynamics?
How does credit flow?
- Autodiff + population signals
How does learning unfold in time?
- GD + stability


- Mathematical Foundations
- Design decisions
- Key experiments
- What does this teach about learning systems?
- How does this connect to robotics and NeuroAI?  
 
## What it contains
- Backprop from scratch 
- Autograd engine (minimal)
- Gradient descent variants
- Visualization of loss landscapes
- Simple ODE-based neural dynamics

## Tech
- Numpy only
- No Pytorch here

## Core Components (implementation)
- A. Minimal Tensor + Autograd Engine
- B. Backpropagation From First Principles
- C. Optimization Algorithms
- D. Loss Landscapes & Geometry
- E. Continuous-Time Gradient Flow

# Core Experiments
- 1 XOR From Scratch
- 2 Vanishing/Exploding Gradients
- 3 Stability of Training
- 4 Gradient Flow vs GD














