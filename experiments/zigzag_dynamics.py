import numpy as np
import matplotlib.pyplot as plt

from learning_dynamics.experiments.utils import make_quadratic_A, eigs, run_gd, savefig


def main():
    outdir = "outputs/week3"

    # Very anisotropic valley produces zig-zag
    A = make_quadratic_A(l1=80.0, l2=1.0, rot_rad=np.deg2rad(35))
    w, lmin, lmax, kappa = eigs(A)
    print(f"Eigenvalues: {w} | kappa={kappa:.1f}")

    theta0 = [7.0, 2.0]
    steps = 80

    # Choose lr within stable range but big enough to show oscillation in stiff direction
    lr = 0.02
    print(f"eta_crit=2/lmax≈{2/lmax:.4f}, using lr={lr:.4f}")

    traj, losses = run_gd(A, theta0, lr=lr, steps=steps)

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2)
    plt.title("Experiment 3: Zig-zag dynamics in a narrow valley (GD)")
    plt.xlabel("theta1")
    plt.ylabel("theta2")
    plt.axis("equal")
    savefig(outdir, "03_zigzag_dynamics_trajectory.png")
    plt.close()

    plt.figure()
    plt.plot(losses)
    plt.yscale("log")
    plt.title("Experiment 3: Loss over time (GD)")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    savefig(outdir, "03_zigzag_dynamics_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
