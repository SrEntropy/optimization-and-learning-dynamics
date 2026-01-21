# Biological Learning Rules (Placeholder Overview)

Biological learning rules: 
- Hebbian plasticity
- STDP
- membrane potential dynamics
- population dynamics

are naturally expressed as **continuous-time differential
equations**. This makes them conceptually similar to gradient flow:



$$
\frac{d\theta}{dt} = -\nabla L(\theta).
$$


The goal of this file is to eventually connect:

- continuous-time biological learning  
- discrete-time backpropagation  
- dynamical systems  
- stability and Jacobians  
- population dynamics  

At this stage, I have not explored biological learning rules in depth. I will
study them later if they remain within the scope of my three repos.

For now, the key idea is:

> Biological learning = continuous-time dynamics  
> Backpropagation = discrete-time dynamics  
> Both fit into the same dynamical-systems framework.

More content will be added as I progress.
