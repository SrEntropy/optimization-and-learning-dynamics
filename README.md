# Optimization and learning-dynamics
A research‑driven exploration of how learning emerges from optimization, signal flow, and dynamical systems implemented entirely from first principles using NumPy.

This project demonstrates mathematical maturity, engineering clarity, and the ability to reason about learning at a mechanistic level.
## Project Vision
Modern ML frameworks hide the mechanics of learning.
This repository rebuilds them from scratch to answer four foundational questions:

### Stage 1 — How does credit flow?
#### Autodiff + population signals
- Construct a minimal Tensor class
- Build computation graphs
- Implement reverse‑mode autodiff
- Explore distributed credit assignment through population‑coded activations

### Stage 2 — How does learning unfold in time?
#### Gradient descent + stability
- Derive GD from first principles
- Analyze vanishing/exploding gradients
- Study stability of discrete updates
- Compare discrete GD with continuous‑time gradient flow

### Stage 3 — How do geometry & optimizers shape learning?
#### Loss surfaces + momentum
- Visualize 1D/2D loss landscapes
- Examine curvature, ridges, and basins
- Implement momentum, RMSProp, Adam
- Show how geometry influences optimizer trajectories

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














