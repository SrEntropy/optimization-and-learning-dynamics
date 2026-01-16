 ## Core Components (implementation)
### A. Minimal Tensor + Autograd Engine
  - Requirements:
    - Scalar and matrix suppport
    - Computational graph
    - Reverse-mode autodiff
    - Manual backward functions
  - Non-negotiable ops
    - add, sub
    - mul
    - matmul
    - relu, tahn, sigmoid
    - sum, mean

- Design Rule: Every backward function must be written by hand with comments explaining the derivative.
- Why? This is where depth shows.

### B. Backpropagation From First Principles
- Deliverables:
  - math_notes/backprop_derivatives.md
    - Chain rule
    - Jacobians
    - Vectorized backprop
>Code must mirror the math extacly.

### C. Optimization Algorithms: Implement only a few, but do an in depth analysis of each.
- Must implement:
  - Gradietn Descent
  - Momentum
  - RMSProp (Optional)

- For each:
  - Update rule
  - Convergence behavior
  - Failure modes

### D. Loss Landscapes & Geometry
- Experiments:
  - Visualize 2D loss surfaces
  - Show saddle points
  - Show flat vs sharp minima

- Why? This connects directly to:
  - Generalization
  - Stability
  - Control theory later
  
### E. Continuous-Time Gradient Flow: NeuroAI bridge
- Task:
  - Interpret  GD as Euler discretication of:
    - 0= VL(0)
    - Compare discret vs continuous dynamics
- Why? This liks to:
  - Backprop
  - Dynamic systems
  - Biological learning rules









