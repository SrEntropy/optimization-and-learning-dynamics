## 2. Vanishing and Exploding Gradients

In deep feedforward networks, the gradient of the loss with respect to parameters in layer $\ell$ is computed via the chain rule:

$$
\frac{\partial L}{\partial W^{(\ell)}} \propto \left( \prod_{k=\ell+1}^{L} \frac{\partial z^{(k)}}{\partial a^{(k)}} \frac{\partial a^{(k)}}{\partial z^{(k)}} \right) \odot (\text{earlier terms})
$$

More precisely, the backpropagated error signal $\delta^{(L)} = \nabla_{a^{(L)}} L$ flows backward multiplied by Jacobians:

$$
\delta^{(\ell)} = \left( W^{(\ell+1)\top} \delta^{(\ell+1)} \right) \odot \sigma'(z^{(\ell)})
$$

The gradient norm at early layers therefore scales roughly with the product of many terms whose magnitude is determined by:

- Weight initialization (e.g., variance of $W^{(\ell)}$)
- Activation derivative $\sigma'(z)$
- Layer width and spectral properties

### Concrete Examples

**Example 1: Vanishing gradients with saturating activations (sigmoid / tanh)**

Consider a deep MLP (e.g., 20–50 layers) with **sigmoid** activation $\sigma(z) = (1 + e^{-z})^{-1}$:

- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \leq 0.25$
- If pre-activations $z$ are not tiny, $\sigma'(z) \approx 0$ in saturated regions (near 0 or 1)
- Repeated multiplication by factors $\lesssim 0.25$ → gradient norm decays **exponentially** with depth: $\|\nabla_{\theta_{\text{early}}}\| \sim (0.25)^{L-\ell}$

Classic behavior (observed in early 2010s networks before ReLU / better init):

- Early layers receive gradients < 10^{-10} after ~10–15 layers → effectively zero updates
- Loss plateaus quickly; model behaves like a shallow net

**tanh** is similar but zero-centered: max derivative = 1, but still saturates → vanishing remains dominant in deep nets without careful scaling.

**Example 2: Exploding gradients with poor initialization or large weights**

In the same deep MLP but with weights initialized from $\mathcal{N}(0,1)$ (too large variance) or during unstable training phases:

- If $\|W^{(\ell)}\|_2 > 1$ on average, forward activations explode → saturating activations → tiny $\sigma'$
- But if activations stay moderate and $\|W^{(\ell)}\|_2 > 1$ persistently, backward signal **grows exponentially**: $\|\delta^{(\ell)}\| \sim c^{L-\ell}$ with $c > 1$
- Result: gradients become huge (10^6–10^{20}) → NaN / Inf in float32 → training divergence

Classic case: RNNs / very deep feedforward nets without gradient clipping before ~2014–2015.

**Example 3: Modern mitigation — ReLU and variants**

ReLU: $\sigma(z) = \max(0,z)$, derivative = 1 (when $z>0$) or 0

- No saturation for positive activations → gradients pass through unchanged in active paths
- He/Kaiming init ($\text{Var}(W) = 2 / n_{\text{in}}$ for ReLU) keeps forward variance ~1 per layer
- Result: gradient norms remain roughly constant or decay only mildly with depth (unless many dead ReLUs)

Variants like **Leaky ReLU** ($\alpha=0.01$), **GELU**, **Swish** further stabilize.

Even with ReLU, exploding can still occur near the **edge of stability** in very large learning-rate regimes (e.g., LARS / LAMB optimizers), but vanishing is largely solved.

### Suggested Figures for This Section

Insert these as images generated from your repo's notebooks (e.g., `notebooks/vanishing_exploding_demo.ipynb` using PyTorch or JAX).

1. **Gradient norm vs. layer depth (log scale)**

   - Plot: log₁₀(‖∇W^{(ℓ)}‖₂) vs. layer index ℓ (from output → input, i.e., right to left)
   - Curves:
     - Sigmoid/tanh network (steep exponential decay → vanishing)
     - Random large-init network (exponential growth → exploding)
     - ReLU + He init (relatively flat or slow decay)
   - Caption: “Gradient norm explosion/vanishing as a function of depth in a 50-layer MLP on MNIST/CIFAR. Backprop from right (output) to left (input).”
   - Typical look: sigmoid curve drops ~10 orders of magnitude by layer 10–15; ReLU stays within 1–2 orders.

2. **Training loss curves comparison**

   - Plot: train loss vs. epoch for same architecture but different activations (sigmoid vs. ReLU vs. tanh)
   - Sigmoid/tanh: flat plateau after few epochs (vanishing)
   - ReLU: continues decreasing meaningfully
   - Caption: “Loss stagnation due to vanishing gradients in saturating activations.”

## 3. Stability of Discrete Updates

(Keep most of the existing content; add a figure here too)

### Suggested Figure

**Convergence / divergence trajectories in quadratic bowl**

- 2D quadratic loss: $L(\theta) = \frac{1}{2} (\theta_1^2 + 100 \theta_2^2)$ (ill-conditioned, $\kappa=100$)
- Plot parameter trajectories starting from (1,1) for different $\eta$:
  - Small $\eta = 0.001$: smooth convergence to (0,0)
  - Medium $\eta \approx 0.019$: oscillatory convergence (underdamped)
  - Large $\eta = 0.021 > 2/\lambda_{\max}$: divergence (oscillations grow)
- Contour lines of loss + arrows/points showing steps
- Caption: “Gradient descent trajectories on a poorly conditioned quadratic bowl. Divergence occurs when $\eta > 2/\lambda_{\max}$.”

Use `matplotlib` + `quiver` or just scatter + line plots.

## 4. Discrete GD vs Continuous-Time Gradient Flow

(Add visual comparison)

### Suggested Figure

**Discrete GD vs. continuous gradient flow trajectories**

- Same quadratic or simple non-convex loss (e.g., Rosenbrock or shallow MLP landscape)
- Plot parameter path:
  - Blue: continuous gradient flow (numerical ODE solver, e.g., `scipy.integrate.solve_ivp` or `torchdiffeq`)
  - Orange: discrete GD with moderate $\eta$
  - Green: discrete GD with large $\eta$ (shows overshooting / different path)
- Caption: “Trajectories diverge when discrete step size is large; GD can escape regions that gradient flow would traverse slowly.”
- Alternative: loss vs. effective time ($t = k\eta$) showing continuous limit as $\eta \to 0$.

In notebooks: use `torch.autograd` for gradients + simple Euler/Runge-Kutta for flow.





