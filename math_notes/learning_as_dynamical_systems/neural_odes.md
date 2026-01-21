# Neural ODEs (Placeholder Overview)

Neural ODEs extend the idea of residual networks by treating depth as a
continuous variable. The goal of this note is not to implement Neural ODEs, but
to show how the *discrete* forward/backward passes used in standard neural
networks relate to their *continuous-time* counterparts.

## From Discrete Steps to a Continuous Flow

A ResNet block can be written as a Forward Euler step:



$$
h_{k+1} = h_k + \Delta t \, f_\theta(h_k, t_k).
$$



Where:
- $h_k$ = hidden state at step $k$
- $h_{k+1}$ = hidden state at step $k+1$
- $\Delta t$ = step size
- $f_\theta(h_k, t_k)$ = slope/velocity at the current point

Rewriting:



$$
\frac{h_{k+1} - h_k}{\Delta t} = f_\theta(h_k, t_k).
$$



Taking the limit as $\Delta t \to 0$ gives the continuous-time dynamics:



$$
\frac{dh(t)}{dt} = f_\theta(h(t), t).
$$



This is the **Neural ODE**: a continuous-depth version of a ResNet.

## What “Forward Pass” and “Backpropagation” Mean Here

Important:  
The terms *forward pass* and *backpropagation* in this context refer to the
**continuous-time** versions used in Neural ODEs — not the discrete forward/backward
passes used in my autodiff engine.

- **Forward pass (Neural ODE):** solve the ODE from $t=0$ to $t=T$ .
- **Backward pass (Neural ODE):** solve the *adjoint ODE* backward in time to compute
  gradients efficiently.

These are the continuous analogues of:
- computing activations (forward pass)
- propagating gradients using transpose Jacobians (backprop)

in a standard discrete network.

## Status

This file is currently a high-level placeholder. I will explore Neural ODEs in detail later if they remain within the scope of my three repos
(optimization-and-learning-dynamics, autodiff engine, and robotics/NeuroAI).

For now, the key ideas are:

- ResNets ≈ Forward Euler steps  
- Neural ODEs = continuous-depth limit  
- Discrete backprop ≈ adjoint method  
- Continuous backprop = adjoint ODE (reverse-time sensitivity dynamics)

More details will be added as I progress through my learning roadmap.
