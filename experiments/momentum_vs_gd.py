import numpy as np
import matplotlib.pyplot as plt

from learning_dynamics.experiments.utils import make_quadratic_A, eigs, run_gd, run_momentum, savefig


def main():
    outdir = "outputs/week3"

    A = make_quadratic_A(l1=60.0, l2=1.0, rot_rad=np.deg2rad(30))
    w, lmin, lmax, kappa = eigs(A)
    print(f"Eigenvalues: {w} | kappa={kappa:.1f}")

    theta0 = [6.0, 6.0]
    steps = 80

    lr = 0.02
    beta = 0.9

    gd_traj, gd_losses = run_gd(A, theta0, lr=lr, steps=steps)
    mom_traj, mom_v, mom_losses = run_momentum(A, theta0, lr=lr, beta=beta, steps=steps)

    # Trajectories
    plt.figure()
    plt.plot(gd_traj[:, 0], gd_traj[:, 1], marker="o", markersize=2, label="GD")
    plt.plot(mom_traj[:, 0], mom_traj[:, 1], marker="o", markersize=2, label=f"Momentum beta={beta}")
    plt.title("Experiment 4: Momentum vs GD (trajectory)")
    plt.xlabel("theta1")
    plt.ylabel("theta2")
    plt.legend()
    plt.axis("equal")
    savefig(outdir, "04_momentum_vs_gd_trajectory.png")
    plt.close()

    # Loss curves
    plt.figure()
    plt.plot(gd_losses, label="GD")
    plt.plot(mom_losses, label=f"Momentum beta={beta}")
    plt.yscale("log")
    plt.title("Experiment 4: Momentum vs GD (loss)")
    plt.xlabel("step")
    plt.ylabel("loss (log scale)")
    plt.legend()
    savefig(outdir, "04_momentum_vs_gd_loss.png")
    plt.close()

    # Phase-space (theta, v) for momentum in 1D slice (just plot theta1 vs v1)
    plt.figure()
    plt.plot(mom_traj[:, 0], mom_v[:, 0], marker="o", markersize=2)
    plt.title("Experiment 4: Momentum phase space (theta1 vs v1)")
    plt.xlabel("theta1")
    plt.ylabel("v1")
    savefig(outdir, "04_momentum_phase_space_theta1_v1.png")
    plt.close()


if __name__ == "__main__":
    main()
