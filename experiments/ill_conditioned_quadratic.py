import numpy as np
import matplotlib.pyplot as plt

from learning_dynamics.experiments.utils import make_quadratic_A, eigs, run_gd, savefig


def main():
    outdir = "outputs/week3"

    # Ill-conditioned: lambda1 >> lambda2
    A = make_quadratic_A(l1=50.0, l2=1.0, rot_rad=np.deg2rad(25))
    w, lmin, lmax, kappa = eigs(A)

    print(f"Eigenvalues: {w}  | lambda_max={lmax:.3f}  lambda_min={lmin:.3f}  kappa={kappa:.2f}")
    print(f"GD stability heuristic: eta < 2/lambda_max = {2.0/lmax:.4f}")

    theta0 = [6.0, 6.0]
    steps = 60

    # Show stable vs near-unstable learning rates
    lrs = [0.01, 0.03, (2.0 / lmax) * 0.99]  # last one is very close to boundary

    plt.figure()
    for lr in lrs:
        traj, losses = run_gd(A, theta0, lr=lr, steps=steps)
        plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2, label=f"lr={lr:.4f}")

    plt.title("Experiment 1: Ill-conditioned quadratic (trajectory)")
    plt.xlabel("theta1")
    plt.ylabel("theta2")
    plt.legend()
    plt.axis("equal")
    savefig(outdir, "01_ill_conditioned_quadratic_trajectory.png")
    plt.close()

    # Loss curves
    plt.figure()
    for lr in lrs:
        traj, losses = run_gd(A, theta0, lr=lr, steps=steps)
        plt.plot(losses, label=f"lr={lr:.4f}")
    plt.yscale("log")
    plt.title("Experiment 1: Loss decay under ill-conditioning")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.legend()
    savefig(outdir, "01_ill_conditioned_quadratic_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
