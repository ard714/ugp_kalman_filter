"""
run_neural_demo.py — RUNNER: Neural KalmanBOT vs Classical Dynamic-PCI Comparison
==================================================================================

Role: Head-to-head comparison of the GRU-based Neural KalmanBOT (learned Kalman gain)
      against the classical analytic Dynamic-PCI Kalman filter on a single pair.
      Called by generate_all_plots.py, or run standalone.

What It Does:
  1. Fits classical Dynamic-PCI parameters (MLE) on formation data
  2. Trains Neural KalmanBOT: GRU learns K_t via two-stage protocol
     Stage 1: MSE pre-training on X2 reconstruction (15 epochs)
     Stage 2: Differentiable Sharpe fine-tuning (15 epochs)
  3. Runs both models out-of-sample on trading data
  4. Backtests both and compares Sharpe, return, drawdown
  5. Generates side-by-side equity curve plot
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import fetch_pair_data
from dynamic_pci import fit_dynamic_pci, DynamicPCIKalmanFilter
from neural_kalman_pci import (
    KalmanNetGain,
    prepare_tensors_4state,
    train_neural_kalman_4state,
    run_neural_filter_4state,
)
from backtest import backtest_pair, performance_summary


def compare_models(ticker1="KO", ticker2="PEP", output_plot="outputs/neural_vs_classical.png"):
    print(f"Fetching Data for {ticker1} - {ticker2}...")
    pair_data = fetch_pair_data(ticker1, ticker2)

    # Base-1.0 Normalization
    s1, s2 = float(pair_data.x1_all[0]), float(pair_data.x2_all[0])
    x1_norm = pair_data.x1_all / s1
    x2_norm = pair_data.x2_all / s2
    form_n = len(pair_data.x1_formation)
    
    x1_form_n = x1_norm[:form_n]
    x2_form_n = x2_norm[:form_n]
    x1_test_n = x1_norm[form_n:]
    x2_test_n = x2_norm[form_n:]

    # 1. Fit classical model
    print("Fitting classical Dynamic-PCI on formation data...")
    fitted_dyn, _ = fit_dynamic_pci(x1_form_n, x2_form_n, allow_dynamic_beta=True)
    dyn_kf = DynamicPCIKalmanFilter(fitted_dyn)
    res_test = dyn_kf.filter(x1_test_n, x2_test_n)

    # 2. Train Neural KalmanNet
    print("\nTraining Neural KalmanBOT (4-state architecture)...")
    batch_y, batch_returns, batch_H = prepare_tensors_4state(x1_form_n, x2_form_n, dyn_kf)
    dataset = TensorDataset(batch_y, batch_returns, batch_H)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = KalmanNetGain(state_dim=4, obs_dim=2, hidden_dim=32)
    model = train_neural_kalman_4state(model, dataloader, dyn_kf, epochs_stage1=15, epochs_stage2=15)

    # 3. Run Out-of-Sample
    print("\nRunning Models Out-of-Sample...")
    neural_beta, neural_M = run_neural_filter_4state(model, x1_test_n, x2_test_n, dyn_kf)

    df_neural, trades_neural = backtest_pair(x1_test_n, x2_test_n, neural_beta, neural_M, z_window=80)
    perf_neural = performance_summary(df_neural)

    df_classic, trades_classic = backtest_pair(x1_test_n, x2_test_n, res_test["beta"], res_test["M"], z_window=80)
    perf_classic = performance_summary(df_classic)

    print("\n" + "=" * 58)
    print(f" RESULTS COMPARISON: {ticker1} vs {ticker2}")
    print("=" * 58)
    header = f"{'Metric':<22} | {'Classical Dynamic-PCI':<22} | {'Neural KalmanBOT'}"
    print(header)
    print("-" * len(header))
    for k in perf_classic.keys():
        val_c = perf_classic[k]
        val_n = perf_neural[k]
        if isinstance(val_c, float):
            print(f"{k:<22} | {val_c:<22.4f} | {val_n:.4f}")
        else:
            print(f"{k:<22} | {str(val_c):<22} | {str(val_n)}")
    print("-" * len(header))
    print(f"{'Completed Trades':<22} | {len(trades_classic):<22d} | {len(trades_neural)}")

    # Plot
    os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
    dates_trade = pair_data.dates_trading
    plt.figure(figsize=(12, 6))
    plt.plot(dates_trade, df_classic["equity"].values, label=f"Classical Dynamic-PCI (Sharpe: {perf_classic['sharpe']:.2f}, Ret: {perf_classic['total_return']*100:.1f}%)", color="#2563eb", linewidth=1.8)
    plt.plot(dates_trade, df_neural["equity"].values, label=f"Neural KalmanBOT (Sharpe: {perf_neural['sharpe']:.2f}, Ret: {perf_neural['total_return']*100:.1f}%)", color="#7c3aed", linewidth=1.8)
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    plt.title(f"Out-of-Sample Cumulative Returns: {ticker1} vs {ticker2}", fontsize=12, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Growth of $1")
    plt.legend(framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot to: {output_plot}")


if __name__ == "__main__":
    compare_models()
