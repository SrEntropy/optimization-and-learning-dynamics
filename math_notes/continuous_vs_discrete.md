# Continuous-Time Gradient Flow: NeuroAI bridge
- Task:
  - Interpret  GD as Euler discretication of:
    - 0= VL(0)
    - Compare discret vs continuous dynamics
- Why? This liks to:
  - Backprop
  - Dynamic systems
  - Biological learning rules



## 4. Discrete GD vs Continuous-Time Gradient Flow

Gradient descent is a **discretization** of a continuous-time dynamical system.

The **gradient flow** ODE is the limit $\eta \to 0$ with time-rescaling:

$$
\frac{d\theta(t)}{dt} = -\nabla L(\theta(t))
$$

This is the steepest descent ODE in the Euclidean metric.

GD with step size $\eta$ is the **forward Euler** discretization:

$$
\theta_{t+1} - \theta_t = -\eta \nabla L(\theta_t) \quad \Leftrightarrow \quad \frac{\theta_{t+1} - \theta_t}{\eta} \approx \frac{d\theta}{dt}\bigg|_{t = t\eta}
$$

**Advantages of the continuous view**

- Analytical tools: Lyapunov functions, contraction rates, convergence in continuous time often cleaner.
- No step-size tuning artifacts in the limit.
- For convex smooth $L$, gradient flow converges to global min; rate depends on strong convexity / PL inequality / smoothness.

**Differences / discretization artifacts**

- Discrete GD can diverge for large $\eta$ even when flow is stable.
- Oscillations / **edge of stability** phenomena appear in discrete time but not in continuous flow (e.g., sharpness increase beyond continuous limit).
- Momentum / accelerated methods can be viewed as better integrators (e.g., Nesterov ≈ symplectic-like).
- In very deep nets, large effective $\eta$ pushes GD far from the continuous flow trajectory.

**When is discrete close to continuous?**

When $\eta$ is small relative to local Lipschitz constants / curvature, or when the trajectory stays in well-behaved regions (e.g., homogeneous nets, good init).

In practice, modern training often lives near the **edge of stability** — discrete effects are essential, not artifacts.

## Next steps / Experiments to add

- Simulate quadratic bowl with different $\eta$ → visualize convergence / divergence.
- Plot gradient norms through depth in a toy MLP → show vanishing/exploding.
- Compare trajectories of GD vs. numerical integration of gradient flow (e.g., RK4) on same loss landscape.

See also: `notebooks/gradient_flow_comparison.ipynb` (to be written).

.....................
