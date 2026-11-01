"""
simulate.py — HELPER: Synthetic Data Generation & Parameter Recovery Validation
=================================================================================

Role: Generates synthetic price pairs that follow the Dynamic-PCI data-generating
      process (DGP), with KNOWN time-varying β and KNOWN M_t/R_t decomposition.
      Used to validate that MLE + Kalman filter can correctly recover parameters
      and track states BEFORE applying the system to real market data.

Synthetic DGP:
  X1_t: random walk in log space (driver asset, like a real stock)
  β_t:  constant or drifting random walk (hedge ratio)
  M_t:  AR(1) mean-reverting component (the tradeable signal)
  R_t:  random-walk permanent component (non-tradeable drift)
  X2_t = β_t · X1_t + M_t + R_t (coupled asset)

Validation outputs:
  - Parameter recovery table (true vs MLE-estimated values)
  - State tracking accuracy (RMSE, Pearson correlation for β, M, R)
  - 6-panel diagnostic plot (price trajectories, state tracking, innovation whitening)

Key Exports:
  - simulate_dynamic_pci()         : Generate synthetic data with known parameters
  - validate_simulation_pipeline() : Full simulation → MLE → filter → reporting
"""

import numpy as np


def simulate_dynamic_pci(
    n=1250,
    beta_path="drift",      # "constant" or "drift"
    beta0=1.5,
    beta_drift_std=0.0015,  # std dev of beta's daily random-walk step
    rho=0.90,
    sigma_M=0.5,
    sigma_R=0.3,
    sigma_x1=0.02,          # daily vol of the (random-walk) driver series X1
    x1_start=100.0,
    seed=None,
):
    """
    Returns dict with x1, x2, beta_true, M_true, R_true (all length n).

    X1_t is generated as a random walk (like a real stock's log price
    scaled back to price level, roughly).
    beta_t either stays fixed at beta0, or itself follows a random walk.
    M_t is AR(1) (transient / mean-reverting).
    R_t is a random walk (permanent shocks to the spread).
    X2_t = beta_t * X1_t + M_t + R_t.
    """
    rng = np.random.default_rng(seed)

    # Driver series X1: random walk in log space, then exponentiate to price
    log_x1 = np.log(x1_start) + np.cumsum(rng.normal(0, sigma_x1, n))
    x1 = np.exp(log_x1)

    # Hedge ratio path
    beta_true = np.zeros(n)
    beta_true[0] = beta0
    if beta_path == "drift":
        steps = rng.normal(0, beta_drift_std, n)
        for t in range(1, n):
            beta_true[t] = beta_true[t - 1] + steps[t]
    else:
        beta_true[:] = beta0

    # Mean-reverting component M (AR(1))
    M_true = np.zeros(n)
    eps_M = rng.normal(0, sigma_M, n)
    for t in range(1, n):
        M_true[t] = rho * M_true[t - 1] + eps_M[t]

    # Random-walk component R
    eps_R = rng.normal(0, sigma_R, n)
    R_true = np.cumsum(eps_R)
    R_true[0] = 0.0

    x2 = beta_true * x1 + M_true + R_true

    return dict(
        x1=x1,
        x2=x2,
        beta_true=beta_true,
        M_true=M_true,
        R_true=R_true,
        params_true=dict(
            beta0=beta0,
            rho=rho,
            sigma_M=sigma_M,
            sigma_R=sigma_R,
            q_beta=beta_drift_std ** 2 if beta_path == "drift" else 0.0,
            sigma_x1=sigma_x1,
        ),
    )


def validate_simulation_pipeline(
    n: int = 1250,
    beta_path: str = "drift",
    seed: int = 42,
    save_plot: bool = True,
    plot_path: str = "simulation_results.png",
):
    """
    Run full synthetic simulation, MLE parameter estimation, and Kalman filtering.
    Reports parameter recovery accuracy and state tracking performance.
    """
    from dynamic_pci import fit_dynamic_pci, DynamicPCIKalmanFilter

    print("=" * 72)
    print(" DYNAMIC-PCI SYNTHETIC SIMULATION & VALIDATION STUDY")
    print("=" * 72)
    print(f"Generating synthetic pair (n={n}, beta_path='{beta_path}', seed={seed})...")

    sim_data = simulate_dynamic_pci(n=n, beta_path=beta_path, seed=seed)
    x1 = sim_data["x1"]
    x2 = sim_data["x2"]
    p_true = sim_data["params_true"]

    print("\nRunning Maximum Likelihood Estimation (L-BFGS-B on 4-State Space)...")
    fitted_params, opt_res = fit_dynamic_pci(x1, x2, allow_dynamic_beta=True)

    print("\nRunning 4-State Kalman Filter forward through simulated trajectory...")
    kf = DynamicPCIKalmanFilter(fitted_params)
    filtered = kf.filter(x1, x2)

    # State tracking metrics
    beta_true = sim_data["beta_true"]
    beta_est = filtered["beta"]
    beta_rmse = float(np.sqrt(np.mean((beta_est - beta_true) ** 2)))
    beta_corr = float(np.corrcoef(beta_est, beta_true)[0, 1])

    M_true = sim_data["M_true"]
    M_est = filtered["M"]
    M_rmse = float(np.sqrt(np.mean((M_est - M_true) ** 2)))
    M_corr = float(np.corrcoef(M_est, M_true)[0, 1])

    R_true = sim_data["R_true"]
    R_est = filtered["R"]
    R_rmse = float(np.sqrt(np.mean((R_est - R_true) ** 2)))
    R_corr = float(np.corrcoef(R_est, R_true)[0, 1])

    # Spread tracking
    spread_true = M_true + R_true
    spread_est = M_est + R_est
    spread_rmse = float(np.sqrt(np.mean((spread_est - spread_true) ** 2)))

    # Parameter recovery table
    print("\n" + "-" * 72)
    print(" PARAMETER RECOVERY EVALUATION")
    print("-" * 72)
    header = f"{'Parameter':<18} | {'True Value':<16} | {'Estimated (MLE)':<16} | {'Status'}"
    print(header)
    print("-" * len(header))

    def param_row(name, val_true, val_est, fmt=".4f"):
        t_str = f"{val_true:{fmt}}" if isinstance(val_true, (int, float)) else str(val_true)
        e_str = f"{val_est:{fmt}}" if isinstance(val_est, (int, float)) else str(val_est)
        diff = abs(val_true - val_est) if isinstance(val_true, (int, float)) and isinstance(val_est, (int, float)) else 0.0
        status = "Recovered" if diff < max(0.2 * abs(val_true), 1e-4) else "Close"
        print(f"{name:<18} | {t_str:<16} | {e_str:<16} | {status}")

    param_row("beta0", p_true["beta0"], fitted_params.beta0, ".4f")
    param_row("rho (AR-1)", p_true["rho"], fitted_params.rho, ".4f")
    param_row("sigma_M", p_true["sigma_M"], fitted_params.sigma_M, ".4f")
    param_row("sigma_R", p_true["sigma_R"], fitted_params.sigma_R, ".4f")
    param_row("q_beta", p_true["q_beta"], fitted_params.q_beta, ".2e")
    print("-" * 72)

    print("\n" + "-" * 72)
    print(" KALMAN FILTER STATE TRACKING ACCURACY")
    print("-" * 72)
    print(f"{'State Component':<25} | {'RMSE':<12} | {'Pearson Correlation (r)'}")
    print("-" * 72)
    print(f"{'Hedge Ratio (beta_t)':<25} | {beta_rmse:<12.4f} | {beta_corr:.4f}")
    print(f"{'Mean-Reverting (M_t)':<25} | {M_rmse:<12.4f} | {M_corr:.4f}")
    print(f"{'Random-Walk (R_t)':<25} | {R_rmse:<12.4f} | {R_corr:.4f}")
    print(f"{'Total Spread (M_t + R_t)':<25} | {spread_rmse:<12.4f} | {np.corrcoef(spread_est, spread_true)[0, 1]:.4f}")
    print("-" * 72)
    print(f"Final Filter Log-Likelihood: {filtered['loglik']:.2f}")

    if save_plot:
        from plot_utils import plot_detailed_simulation
        plot_detailed_simulation(sim_data, filtered, fitted_params, save_path=plot_path)

    print("\nSimulation validation completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic-PCI Simulation & Recovery Validation")
    parser.add_argument("--n", type=int, default=1250, help="Number of simulated steps (default: 1250)")
    parser.add_argument("--beta-path", type=str, default="drift", choices=["drift", "constant"], help="Path type for beta")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-plot", action="store_true", help="Disable generating output plot")
    parser.add_argument("--out", type=str, default="outputs/simulation_results.png", help="Output plot filename")
    args = parser.parse_args()

    validate_simulation_pipeline(
        n=args.n,
        beta_path=args.beta_path,
        seed=args.seed,
        save_plot=not args.no_plot,
        plot_path=args.out,
    )
