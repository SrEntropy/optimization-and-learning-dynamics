# Vanishing and Exploding Gradients
*A mathematical explanation using Jacobians, backprop error signals, and repeated multiplication*

Deep networks suffer from **vanishing** and **exploding** gradients because backpropagation multiplies many local derivatives together. This file builds the phenomenon from first principles, starting with the 1‑D quadratic case and generalizing to deep networks using Jacobians and backprop error signals.

---

## 1. Core mechanism: repeated multiplication
### a. The Quadratic loss
In the 1‑D quadratic case:

$$
L(\theta) = \frac{1}{2} a\theta^2, a >0
$$

Derivative: 

$$
L'(\theta) = a\theta$$


gradient descent gives:

$$
\theta_{(t+1)} = \theta_t - \eta L'(\theta_t) = \theta_t - \eta a \theta_t   .
$$

Factor out $\theta_t$ :

$$
\theta_{t+1} = (1 - \eta a)\theta_t.
$$



After $t$ steps:


$$
\theta_t = (1 - \eta a)^t \theta_0.
$$

This is the origin of the repeated multiplication.

### b. Identify the multiplier

Define:

$$
r = 1- \eta a
$$

Then the update becomes this, which is a linear dynamical system:
   - Each step multiplies the previous by the same constant $r$ . 

$$ \theta_{t+1} = r \theta_t$$

### c. Applying the update repeatedly

Let's unroll it:
- Step 1:
   - $\theta_1 = r \theta_0$
- Step 2:
   - $\theta_2 = r \theta_1 =  r (r \theta_0) = r^2 \theta_0$

- Step 3:
   - $\theta_3 = r \theta_2=  r (r^2 \theta_0) = r^3 \theta_0$

Pattern
- $\theta_t = r^t \theta_0$

This is repeated multiplication by a scalar:

### d. When does the repeated multiplication shrink
So, at each step, $\theta_t$ must approach $0$  

$$\theta_t = r^t \theta_0  → 0$$

This happens if and only if:
$$|r| < 1$$

why?
- if $|r| < 1$ , then $r^t  → 0$ (Converges).
- if $|r| > 1$ , then $r^t  → ∞$ (Diverges).
- if $r = -0.5$, it oscillates but shrinks.
- if $r = 1.2$, it explodes.
- if $r = -1.2$,  it oscillates and explodes.

  
Therefore, the inequalities that met this condition ($r  = 1- \eta a$) are 
- If $|1 - \eta a| < 1$, the sequence shrinks → **vanishing**.
- If $|1 - \eta a| > 1$, the sequence grows → **exploding**.

### f. To prove this, solve the inequality to find a final stability condition for the 1-D quadratic Loss.
Absolute value inequality:

$$|x| < 1,  and -1< x <1$$

So

 $$-1 < 1 - \eta a<1$$

Solve the inequalities

Left side:

$$-1 < 1 - \eta a$$

- Subtract 1:
  
$$-2 < - \eta a$$

- Multiply by -1 to flip the inequality:

 
$$2> \eta a$$

- Right side:

$$1 - \eta a<1$$

- Subtract $-1$ :
  
$$- \eta a< 0$$

Multiply by -1:

$$ \eta a > 0$$

Combining both sides:

$$  0 < \eta a <2 $$

Since a > 0, divide by a:

$$  0 < \eta <2/a $$

This is the final stability condition. Therefore, the repeated multiplication shrinks only when $|r| < 1$. So, the learning rate must be greater than $0$ and less than $2/a$. 

Deep networks behave the same way, except instead of multiplying by a scalar, we multiply by **Jacobians**.

---

## 2. Backpropagation as repeated Jacobian multiplication

When a function maps scalars, the chain rule looks like:

$$\frac{\partial L}{\partial x} = \frac{\partial y}{\partial x}*\frac{\partial L}{\partial y}$$ 

Deep networks are vector-valued, so instead of a single derivative $\frac{\partial y}{\partial x}$ , we have a matrix of partial derivatives (a Jacobian).

## 2.1 Layer structure and notation
A deep network consists of layers:

$$
h^{(0)} = x, \qquad
h^{(\ell)} = f^{(\ell)}(h^{(\ell-1)}),
$$


Each layer:
 - **Input**: vector $h^{(\ell -1)}$
 - **output**: vector $h^{(\ell)}$


The loss $L$ depends on the final output $h^{(L)}$.

## 2.2 Backprop error signal (**Delta**)
Defining the gradient of the loss with respect to the layer's output:



$$
\delta^{(\ell)} = \frac{\partial L}{\partial h^{(\ell)}}.
$$

This is the error signal that gets propagated backward.


## 2.3 Jacobian of a layer
The Jacobian of layer $\ell$ is:

$$j_{\ell} =\frac{\partial h^{(\ell)}}{\partial h^{(\ell -1)}}$$


 This matrix contains all **local partial derivatives** of that layer

## 2.4 Vector chain rule for backprop
The vector chain rule gives:

$$
\delta^{(\ell)} 
= \left( \frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}} \right)^\top \delta^{(\ell+1)}
= J_{\ell+1}^\top \delta^{(\ell+1)},
$$

This is the exact vector generalization of the scalar rule:

$$\frac{\partial L}{\partial x} = \frac{\partial y}{\partial x}*\frac{\partial L}{\partial y}$$ 

## 2.5 Backprop through the entire network
Applying the chain rule repeatedly:


$$\delta^{(L-1)} = J_L^\top \delta^{(L)},$$
$$\delta^{(L-2)} = J_{L-1}^\top \delta^{(L-1)} = J_{L-1}^\top J_L^\top \delta^{(L)} \delta^{(L)},$$

and so on.

Unrolling backprop through all layers:

$$
\delta^{(0)} 
= J_1^\top J_2^\top \cdots J_L^\top \delta^{(L)}.
$$



This is the multidimensional analogue of:



$$
\theta_t = r^t \theta_0.
$$

- In the scalar case, backprop multiplies by a number 𝑟.
- In deep networks, backprop multiplies by Jacobians.

This repeated multiplication is the mathematical root of **vanishing** and **exploding** gradients

---
# 3. When do gradients vanish or explode?
*(Spectral norms and singular values)*

The magnitude of the full backpropagated gradient:



$$
\delta^{(0)} = J_1^\top J_2^\top \cdots J_L^\top \, \delta^{(L)}
$$



is controlled by the **spectral norms** (largest singular values) of the Jacobians.

If each Jacobian satisfies:



$$
\|J_\ell\|_2 < 1,
$$



then:



$$
\|J_1^\top J_2^\top \cdots J_L^\top\|_2
\;\le\;
\prod_{\ell=1}^L \|J_\ell\|_2
\;\longrightarrow\; 0
\quad \text{as } L \to \infty.
$$



→ **Vanishing gradients**

If instead:


$$
\|J_\ell\|_2 > 1,
$$


Then the product grows exponentially.

→ **Exploding gradients**

This is the exact same mechanism as the scalar condition  
$|r| < 1$ or $|r| > 1$,  
but now applied to **matrices** instead of scalars.

---

# 4. Where do Jacobian norms come from?

Each layer’s Jacobian is the product of:

1. **Weight matrix**  
   

$$
   W^{(\ell)}
   $$



2. **Activation derivative**  
   

$$
   \sigma'(z^{(\ell)}),
   \qquad
   z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}.
   $$



Thus:



$$
J_\ell=
\mathrm{diag}(\sigma'(z^{(\ell)})) \, W^{(\ell)}.
$$



So the spectral norm satisfies:



$$
\|J_\ell\|_2
\;\le\;
\|\mathrm{diag}(\sigma'(z^{(\ell)}))\|_2
\cdot
\|W^{(\ell)}\|_2.
$$



This gives two independent sources of vanishing/exploding gradients.

---

## 4.1 Activation derivatives (saturation)

For tanh:



$$
\sigma'(z) = 1 - \tanh^2(z).
$$



- When \(z \approx 0\): derivative ≈ 1 → good gradient flow  
- When \(|z|\) is large: derivative ≈ 0 → **vanishing**

Saturation directly shrinks the Jacobian norm.

---

## 4.2 Weight matrices (spectral norms)

- If $\|W^{(\ell)}\|_2 > 1$, the Jacobian can **amplify** gradients  
- If $\|W^{(\ell)}\|_2 < 1$, the Jacobian **shrinks** gradients  

Across many layers, these effects multiply.

---

# 5. Deep networks as a dynamical system

Backprop through $L$ layers:



$$
\delta^{(0)}=
\left( J_1^\top J_2^\top \cdots J_L^\top \right)
\delta^{(L)}.
$$



This is a **discrete-time linear dynamical system**:



$
x_{t+1} = A_t x_t,
\qquad
A_t = J_{t+1}^\top.
$



The behavior is determined by the product of matrices:

- If the product norm → 0 → **vanishing**
- If the product norm → ∞ → **exploding**
- If the product norm stays near 1 → **stable gradient flow**

This is the exact multidimensional generalization of the scalar update:


$$
\theta_{t+1} = r \theta_t.
$$



---

# 6. Summary

Vanishing and exploding gradients arise because backpropagation multiplies many Jacobians together:


$$
\delta^{(0)} =
J_1^\top J_2^\top \cdots J_L^\top \, \delta^{(L)} .
$$



Each Jacobian contains:

- a **weight matrix** $W^{(\ell)}$ ,  
- an **activation derivative** $\sigma'(z^{(\ell)})$ .

The spectral norms of these Jacobians determine whether gradients:

- **vanish** (product norms < 1),  
- **explode** (product norms > 1),  
- **remain stable** (product norms ≈ 1).

This is the same mechanism as the 1‑D quadratic case, but extended to many dimensions and many layers.



