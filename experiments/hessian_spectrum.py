import numpy as np
import matplotlib.pyplot as plt

from learning_dynamics.experiments.utils import make_quadratic_A, eigs, run_gd, savefig


def main():
    outdir = "outputs/week3"

    # Moderate conditioning to focus on stability boundary behavior
    A = make_quadratic_A(l1=10.0, l2=2.0, rot_rad=np.deg2rad(15))
    w, lmin, lmax, kappa = eigs(A)

    eta_crit = 2.0 / lmax
    print(f"Eigenvalues: {w}, eta_crit ≈ {eta_crit:.4f}")

    theta0 = [5.0, -4.0]
    steps = 40

    # Below / near / above stability boundary
    lrs = [0.5 * eta_crit, 0.99 * eta_crit, 1.10 * eta_crit]

    # Plot trajectories
    plt.figure()
    for lr in lrs:
        traj, losses = run_gd(A, theta0, lr=lr, steps=steps)
        plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, label=f"lr={lr:.4f}")

    plt.title("Experiment 2: Stability boundary from Hessian spectrum")
    plt.xlabel("theta1")
    plt.ylabel("theta2")
    plt.legend()
    plt.axis("equal")
    savefig(outdir, "02_hessian_spectrum_stability_trajectory.png")
    plt.close()

    # Plot loss curves (divergence will blow up)
    plt.figure()
    for lr in lrs:
        traj, losses = run_gd(A, theta0, lr=lr, steps=steps)
        plt.plot(losses, label=f"lr={lr:.4f}")
    plt.yscale("log")
    plt.title("Experiment 2: Loss vs step near stability boundary")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.legend()
    savefig(outdir, "02_hessian_spectrum_stability_loss.png")
    plt.close()

    # Show eigenvalues explicitly
    plt.figure()
    plt.stem(w)
    plt.title("Experiment 2: Hessian eigenvalues (spectrum)")
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    savefig(outdir, "02_hessian_spectrum_eigs.png")
    plt.close()


if __name__ == "__main__":
    main()
