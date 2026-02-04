import numpy as np
import matplotlib.pyplot as plt

from learning_dynamics.experiments.utils import eigs, run_gd, savefig


def main():
    outdir = "outputs/week3"

    # Two coordinate scalings. Same "shape" of quadratic but reparameterized.
    # Idea: If you scale coordinates, GD changes behavior because gradients are not parameterization-invariant.

    # Base A (diagonal for clarity)
    A1 = np.diag([20.0, 1.0])  # anisotropic curvature
    w1, lmin1, lmax1, kappa1 = eigs(A1)

    # Reparameterize by scaling coordinates: theta = S z
    # In z-coordinates, the effective quadratic becomes:
    #   L = 1/2 z^T (S^T A S) z
    # Pick S that changes relative scaling of axes.
    S = np.diag([0.2, 3.0])
    A2 = S.T @ A1 @ S
    w2, lmin2, lmax2, kappa2 = eigs(A2)

    print(f"A1 eigs={w1}, kappa={kappa1:.1f}")
    print(f"A2 eigs={w2}, kappa={kappa2:.1f}")

    theta0 = [6.0, 6.0]
    steps = 70

    # Use the same lr in both parameterizations to show behavior changes
    lr = 0.05

    traj1, loss1 = run_gd(A1, theta0, lr=lr, steps=steps)
    traj2, loss2 = run_gd(A2, theta0, lr=lr, steps=steps)

    # Trajectory compare
    plt.figure()
    plt.plot(traj1[:, 0], traj1[:, 1], marker="o", markersize=2, label="A1 (original coords)")
    plt.plot(traj2[:, 0], traj2[:, 1], marker="o", markersize=2, label="A2 (scaled coords)")
    plt.title("Experiment 5: Anisotropic scaling changes GD dynamics")
    plt.xlabel("theta1")
    plt.ylabel("theta2")
    plt.legend()
    plt.axis("equal")
    savefig(outdir, "05_anisotropic_scaling_trajectory.png")
    plt.close()

    # Loss compare
    plt.figure()
    plt.plot(loss1, label="A1")
    plt.plot(loss2, label="A2")
    plt.yscale("log")
    plt.title("Experiment 5: Loss curves differ under scaling")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.legend()
    savefig(outdir, "05_anisotropic_scaling_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
