import os
import numpy as np
import matplotlib.pyplot as plt

from core.parameter import Parameter
from core.ops import matvec, mul, sum_pop


# ---------- Paths / saving ----------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig(outdir: str, name: str):
    ensure_dir(outdir)
    path = os.path.join(outdir, name)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"[saved] {path}")


# ---------- Quadratic construction ----------

def rotation_matrix(theta_rad: float) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def make_quadratic_A(l1: float, l2: float, rot_rad: float = 0.0) -> np.ndarray:
    """
    Build a 2x2 SPD matrix with eigenvalues l1,l2, optionally rotated.

    A = R diag(l1,l2) R^T
    """
    D = np.diag([float(l1), float(l2)])
    R = rotation_matrix(rot_rad)
    return R @ D @ R.T


def eigs(A: np.ndarray):
    w = np.linalg.eigvalsh(A)  # symmetric
    w = np.sort(w)
    lmin, lmax = float(w[0]), float(w[-1])
    kappa = lmax / lmin if lmin > 0 else np.inf
    return w, lmin, lmax, kappa


# ---------- Loss + training loop ----------

def quadratic_loss(A: np.ndarray, theta: Parameter):
    """
    L(theta) = 1/2 * theta^T A theta

    Implemented using your autodiff primitives:
      y = A @ theta
      dot = sum(theta * y)
      loss = 0.5 * dot
    """
    y = matvec(A, theta)
    dot = sum_pop(mul(theta, y))      # scalar
    loss = mul(0.5, dot)              # scalar (broadcasted scalar multiply)
    return loss


def run_gd(A: np.ndarray, theta0, lr: float, steps: int):
    theta = Parameter(theta0)
    traj = [theta.data.copy()]
    losses = []

    for _ in range(steps):
        loss = quadratic_loss(A, theta)
        loss.zero_grad_graph()
        loss.backprop()

        # gradient descent update
        for i in range(len(theta.data)):
            theta.data[i] -= lr * theta.grad[i]

        losses.append(loss.data[0])
        traj.append(theta.data.copy())

    return np.array(traj), np.array(losses)


def run_momentum(A: np.ndarray, theta0, lr: float, beta: float, steps: int):
    theta = Parameter(theta0)
    v = [0.0 for _ in theta.data]

    traj = [theta.data.copy()]
    v_traj = [v.copy()]
    losses = []

    for _ in range(steps):
        loss = quadratic_loss(A, theta)
        loss.zero_grad_graph()
        loss.backprop()

        # v <- beta*v - lr*grad
        for j in range(len(theta.data)):
            v[j] = beta * v[j] - lr * theta.grad[j]

        # theta <- theta + v
        for j in range(len(theta.data)):
            theta.data[j] += v[j]

        losses.append(loss.data[0])
        traj.append(theta.data.copy())
        v_traj.append(v.copy())

    return np.array(traj), np.array(v_traj), np.array(losses)
