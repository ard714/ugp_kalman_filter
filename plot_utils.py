"""
plot_utils.py — HELPER: Publication-Quality Plotting & Visualization Suite
===========================================================================

Role: Contains ALL matplotlib plotting functions used by the project. No model
      logic — pure visualization. Called by generate_all_plots.py, run_all_pairs.py,
      and simulate.py.

Key Exports:
  - plot_detailed_simulation()       : 6-panel synthetic DGP recovery diagnostic
  - plot_benchmark_metrics()         : Grouped bar charts (Sharpe, Return, Max DD) + executive dashboard
  - plot_multi_pair_equity_grid()    : Multi-panel equity curve grid across all pairs
  - plot_regime_screening_analysis() : 4-panel screening diagnostic (R²_MR, half-life, ADF, correlation)
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Cohesive, publication-quality color palette
MODEL_COLORS = {
    "static": "#dc2626",      # Crimson
    "dyn": "#2563eb",         # Royal Blue
    "gated": "#059669",       # Emerald Green
    "neural": "#7c3aed",      # Purple / Violet
    "classic": "#2563eb",
}

MODEL_LABELS = {
    "static": "Static-PCI (Fixed Beta)",
    "dyn": "Dynamic-PCI (Kalman Beta)",
    "gated": "Regime-Gated Dyn-PCI",
    "neural": "Neural KalmanBOT (Learned Gain)",
    "classic": "Classical Dynamic-PCI",
}


def ensure_dir(filepath: str):
    """Ensure parent directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)


def plot_detailed_simulation(
    sim_data: dict,
    filtered: dict,
    fitted_params,
    save_path: str = "outputs/simulation_results.png",
) -> None:
    """
    Generate an exhaustive 6-panel simulation and state-tracking diagnostic dashboard.
    """
    ensure_dir(save_path)
    n = len(sim_data["x1"])
    t = np.arange(n)

    x1 = sim_data["x1"]
    x2 = sim_data["x2"]
    beta_true = sim_data["beta_true"]
    beta_est = filtered["beta"]
    M_true = sim_data["M_true"]
    M_est = filtered["M"]
    R_true = sim_data["R_true"]
    R_est = filtered["R"]
    p_true = sim_data["params_true"]

    # Metrics
    beta_rmse = float(np.sqrt(np.mean((beta_est - beta_true) ** 2)))
    beta_corr = float(np.corrcoef(beta_est, beta_true)[0, 1])
    M_rmse = float(np.sqrt(np.mean((M_est - M_true) ** 2)))
    M_corr = float(np.corrcoef(M_est, M_true)[0, 1])
    R_rmse = float(np.sqrt(np.mean((R_est - R_true) ** 2)))
    R_corr = float(np.corrcoef(R_est, R_true)[0, 1])

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1.1, 1.1, 1.3], hspace=0.32, wspace=0.22)

    # 1. Price Series (Spans both columns)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, x1, label="Driver Asset X1 (Random Walk)", color="#2563eb", linewidth=1.6)
    ax0.plot(t, x2, label="Coupled Asset X2 (beta*X1 + M + R)", color="#d97706", linewidth=1.6)
    ax0.set_title("1. Synthetic Asset Price Trajectories (n = 1250 days)", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Price Level", fontsize=10)
    ax0.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax0.grid(True, linestyle=":", alpha=0.6)

    # 2. Beta Tracking (Top Left)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, beta_true, label="Ground Truth beta_t", color="#111827", linewidth=2.0, linestyle="--")
    ax1.plot(t, beta_est, label=f"Kalman Estimate (r = {beta_corr:.3f})", color="#2563eb", linewidth=1.7)
    ax1.fill_between(t, beta_est, beta_true, color="#93c5fd", alpha=0.3, label=f"Abs Error (RMSE={beta_rmse:.4f})")
    ax1.set_title("2. Dynamic Hedge Ratio (beta_t) Recovery", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Hedge Ratio", fontsize=10)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 3. Mean-Reverting Spread M_t (Top Right)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t, M_true, label="Ground Truth M_t (AR-1)", color="#059669", linewidth=1.8, linestyle="--")
    ax2.plot(t, M_est, label=f"Kalman Filter M_t (r = {M_corr:.3f})", color="#0d9488", linewidth=1.5)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_title("3. Transient Spread Component (M_t)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("M_t Value", fontsize=10)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 4. Random-Walk Component R_t (Bottom Left)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(t, R_true, label="Ground Truth R_t (RW)", color="#dc2626", linewidth=1.8, linestyle="--")
    ax3.plot(t, R_est, label=f"Kalman Filter R_t (r = {R_corr:.3f})", color="#ea580c", linewidth=1.5)
    ax3.set_title("4. Permanent Shock Component (R_t)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("R_t Value", fontsize=10)
    ax3.legend(loc="upper left", framealpha=0.9, fontsize=8.5)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 5. Innovation Whitening Check (Bottom Right)
    ax4 = fig.add_subplot(gs[2, 1])
    innov = filtered.get("innovations", np.zeros(n))
    if len(innov) > 1:
        innov_clean = innov[1:]
        std_innov = np.std(innov_clean)
        norm_innov = (innov_clean - np.mean(innov_clean)) / (std_innov + 1e-8)
        ax4.hist(norm_innov, bins=40, density=True, alpha=0.65, color="#6366f1", label="Innovation Residuals")
        # Standard normal overlay
        xs = np.linspace(-3.5, 3.5, 100)
        norm_pdf = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * xs ** 2)
        ax4.plot(xs, norm_pdf, color="#dc2626", linewidth=1.8, label="N(0, 1) Reference")
    ax4.set_title("5. Normalized Innovation Residuals (Whitening Test)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Normalized Innovation (z-score)", fontsize=10)
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # 6. Parameter Recovery Bar Chart (Spans both columns)
    ax5 = fig.add_subplot(gs[3, :])
    param_names = ["beta0", "rho (AR-1)", "sigma_M", "sigma_R"]
    true_vals = [p_true["beta0"], p_true["rho"], p_true["sigma_M"], p_true["sigma_R"]]
    est_vals = [fitted_params.beta0, fitted_params.rho, fitted_params.sigma_M, fitted_params.sigma_R]

    x_indices = np.arange(len(param_names))
    width = 0.32
    rects1 = ax5.bar(x_indices - width / 2, true_vals, width, label="True Parameter Value", color="#3b82f6", alpha=0.9)
    rects2 = ax5.bar(x_indices + width / 2, est_vals, width, label="MLE Estimated Value", color="#10b981", alpha=0.9)

    ax5.set_title("6. MLE Parameter Recovery Evaluation (Ground Truth vs L-BFGS-B Estimate)", fontsize=11, fontweight="bold")
    ax5.set_xticks(x_indices)
    ax5.set_xticklabels(param_names, fontsize=10, fontweight="bold")
    ax5.set_ylabel("Parameter Value", fontsize=10)
    ax5.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax5.grid(True, linestyle=":", alpha=0.6)

    # Add labels on bars
    for rect1, rect2 in zip(rects1, rects2):
        h1 = rect1.get_height()
        h2 = rect2.get_height()
        ax5.annotate(f"{h1:.4f}", xy=(rect1.get_x() + rect1.get_width() / 2, h1),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        ax5.annotate(f"{h2:.4f}", xy=(rect2.get_x() + rect2.get_width() / 2, h2),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color="#065f46", fontweight="bold")

    plt.suptitle("DYNAMIC-PCI GROUND TRUTH SIMULATION & VALIDATION STUDY", fontsize=13, fontweight="bold", y=0.995)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved detailed simulation diagnostic plot to: {save_path}")


def plot_benchmark_metrics(
    results: List[dict],
    model_names: List[str],
    save_dir: str = "outputs",
) -> Dict[str, str]:
    """
    Generate grouped comparative bar charts for Sharpe, Total Return, and Max Drawdown across all pairs.
    """
    os.makedirs(save_dir, exist_ok=True)
    pairs = [r["pair"] for r in results]
    n_pairs = len(pairs)
    n_models = len(model_names)

    x = np.arange(n_pairs)
    width = 0.8 / n_models

    generated_files = {}

    fig_w = max(13.0, n_pairs * 1.35)

    # 1. Sharpe Ratio Comparison
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    for i, m_name in enumerate(model_names):
        vals = [r[m_name]["sharpe"] if not pd.isna(r[m_name]["sharpe"]) else 0.0 for r in results]
        offset = (i - (n_models - 1) / 2) * width
        color = MODEL_COLORS.get(m_name, "#6b7280")
        label = MODEL_LABELS.get(m_name, m_name.upper())
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.88, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.axhline(0.5, color="#10b981", linestyle="--", linewidth=1.0, alpha=0.7, label="Benchmark Target (Sharpe = 0.5)")
    ax.set_title("Out-of-Sample Sharpe Ratio by Pair and Model Architecture", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=10, fontweight="bold")
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    sharpe_path = os.path.join(save_dir, "benchmark_sharpe_comparison.png")
    plt.savefig(sharpe_path, dpi=180)
    plt.close(fig)
    generated_files["sharpe"] = sharpe_path

    # 2. Total Return Comparison (%)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    for i, m_name in enumerate(model_names):
        vals = [r[m_name]["total_return"] * 100.0 if not pd.isna(r[m_name]["total_return"]) else 0.0 for r in results]
        offset = (i - (n_models - 1) / 2) * width
        color = MODEL_COLORS.get(m_name, "#6b7280")
        label = MODEL_LABELS.get(m_name, m_name.upper())
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.88, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Out-of-Sample Total Return (%) Across Evaluated Pairs (Net of 10bps Costs)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=10, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return_path = os.path.join(save_dir, "benchmark_returns_comparison.png")
    plt.savefig(return_path, dpi=180)
    plt.close(fig)
    generated_files["returns"] = return_path

    # 3. Max Drawdown Comparison (%)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))
    for i, m_name in enumerate(model_names):
        vals = [r[m_name]["max_drawdown"] * 100.0 if not pd.isna(r[m_name]["max_drawdown"]) else 0.0 for r in results]
        offset = (i - (n_models - 1) / 2) * width
        color = MODEL_COLORS.get(m_name, "#6b7280")
        label = MODEL_LABELS.get(m_name, m_name.upper())
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.88, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title("Maximum Out-of-Sample Drawdown (%) by Model Architecture", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=10, fontweight="bold")
    ax.set_ylabel("Maximum Drawdown (%)", fontsize=10)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    dd_path = os.path.join(save_dir, "benchmark_drawdown_comparison.png")
    plt.savefig(dd_path, dpi=180)
    plt.close(fig)
    generated_files["drawdown"] = dd_path

    # 4. Multi-Metric Executive Dashboard (All 3 in 1 figure)
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, 15), sharex=True)
    metrics_info = [
        ("sharpe", "Sharpe Ratio", lambda p: p["sharpe"] if not pd.isna(p["sharpe"]) else 0.0, "{:.2f}"),
        ("returns", "Total Return (%)", lambda p: p["total_return"] * 100 if not pd.isna(p["total_return"]) else 0.0, "{:.1f}%"),
        ("drawdown", "Max Drawdown (%)", lambda p: p["max_drawdown"] * 100 if not pd.isna(p["max_drawdown"]) else 0.0, "{:.1f}%"),
    ]
    for ax_idx, (m_key, title, val_fn, fmt) in enumerate(metrics_info):
        ax = axes[ax_idx]
        for i, m_name in enumerate(model_names):
            vals = [val_fn(r[m_name]) for r in results]
            offset = (i - (n_models - 1) / 2) * width
            color = MODEL_COLORS.get(m_name, "#6b7280")
            label = MODEL_LABELS.get(m_name, m_name.upper())
            bars = ax.bar(x + offset, vals, width, label=label if ax_idx == 0 else "", color=color, alpha=0.88, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylabel(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(f"[{chr(65+ax_idx)}] {title} Across Target Pairs", fontsize=11, fontweight="bold", pad=6)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(pairs, fontsize=10, fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(model_names), fontsize=9.5, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    dash_path = os.path.join(save_dir, "benchmark_summary_dashboard.png")
    plt.savefig(dash_path, dpi=200)
    plt.close(fig)
    generated_files["dashboard"] = dash_path

    print(f"Saved benchmark metric comparison plots to: {save_dir}/")
    return generated_files


def plot_multi_pair_equity_grid(
    equity_series_dict: Dict[str, Dict[str, pd.Series]],
    save_path: str = "outputs/all_pairs_equity_grid.png",
) -> None:
    """
    Generate a multi-panel grid plotting out-of-sample equity curves for every pair.
    
    equity_series_dict: {pair_str: {model_name: pd.Series(equity)}}
    """
    ensure_dir(save_path)
    pairs = list(equity_series_dict.keys())
    n = len(pairs)
    if n == 0:
        return

    # 3x5 Grid Layout (3 rows, 5 columns for up to 15 pairs)
    rows = 3
    cols = 5
    if n > 15:
        cols = 5
        rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(22, 12), sharex=False)
    axes = np.array(axes).flatten()

    for idx, pair in enumerate(pairs):
        ax = axes[idx]
        models_dict = equity_series_dict[pair]
        for m_name, s_equity in models_dict.items():
            color = MODEL_COLORS.get(m_name, "#6b7280")
            label = MODEL_LABELS.get(m_name, m_name.upper())
            dates = s_equity.index
            ax.plot(dates, s_equity.values, label=label, color=color, linewidth=1.5)

        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"{pair} (OOS Growth)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Growth of $1", fontsize=8.5)
        ax.legend(loc="upper left", framealpha=0.85, fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.tick_params(axis='x', rotation=25, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Hide extra subplots if less than rows*cols
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("FULL SYSTEM BENCHMARK: OUT-OF-SAMPLE EQUITY CURVES ACROSS ALL 14 PAIRS (3x5 GRID)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved all pairs equity grid plot (3x5) to: {save_path}")


def plot_regime_screening_analysis(
    screening_records: List[dict],
    save_path: str = "outputs/regime_screening_analysis.png",
) -> None:
    """
    Generate an empirical diagnostic figure explaining why pairs succeed or fail
    (contrasting cointegrated pairs like KO-PEP vs non-cointegrated pairs like AAP-IBM).
    """
    ensure_dir(save_path)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    pairs = [r["pair"] for r in screening_records]
    r2_mrs = [r["r2_mr"] * 100 for r in screening_records]
    half_lifes = [min(r["half_life"], 500.0) for r in screening_records]
    adf_pvals = [r["adf_pvalue"] for r in screening_records]
    corrs = [r["return_corr"] for r in screening_records]
    is_suitable = [r["is_suitable"] for r in screening_records]

    bar_colors = ["#059669" if s else "#dc2626" for s in is_suitable]

    x = np.arange(len(pairs))

    # Panel 1: Spread Variance Breakdown (R2_MR)
    axes[0, 0].bar(x, r2_mrs, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    axes[0, 0].axhline(15.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Suitability Threshold (R2_MR >= 15%)")
    axes[0, 0].set_title("1. Mean-Reverting Spread Share (R2_MR %)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(pairs, fontsize=9.5, fontweight="bold", rotation=25)
    axes[0, 0].set_ylabel("% Mean-Reverting Variance", fontsize=9.5)
    axes[0, 0].legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Mean Reversion Half-Life (days)
    axes[0, 1].bar(x, half_lifes, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    axes[0, 1].axhline(252.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Max Viable Half-Life (252 days = 1 year)")
    axes[0, 1].set_title("2. Estimated Spread Half-Life (Days to 50% Mean Reversion)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(pairs, fontsize=9.5, fontweight="bold", rotation=25)
    axes[0, 1].set_ylabel("Half-Life (Trading Days)", fontsize=9.5)
    axes[0, 1].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel 3: ADF Test p-value on Spread
    axes[1, 0].bar(x, adf_pvals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    axes[1, 0].axhline(0.10, color="#dc2626", linestyle="--", linewidth=1.2, label="Cointegration Significance (p <= 0.10)")
    axes[1, 0].set_title("3. Augmented Dickey-Fuller Test p-value on Spread", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(pairs, fontsize=9.5, fontweight="bold", rotation=25)
    axes[1, 0].set_ylabel("ADF p-value (Lower = Stationary)", fontsize=9.5)
    axes[1, 0].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel 4: Daily Return Correlation
    axes[1, 1].bar(x, corrs, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    axes[1, 1].axhline(0.50, color="#2563eb", linestyle="--", linewidth=1.2, label="Moderate Correlation Benchmark (0.50)")
    axes[1, 1].set_title("4. Formation Period Return Correlation", fontsize=11, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(pairs, fontsize=9.5, fontweight="bold", rotation=25)
    axes[1, 1].set_ylabel("Pearson Correlation", fontsize=9.5)
    axes[1, 1].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("STATISTICAL PAIR SCREENING & COINTEGRATION REGIME ANALYSIS\n(Green = Suitable / Cointegrated | Red = Rejected / Non-Mean-Reverting)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved regime screening analysis plot to: {save_path}")
