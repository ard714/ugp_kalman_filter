"""
run_demo.py — RUNNER: Single-Pair Pipeline (Fit → Filter → Backtest → Dashboard)
==================================================================================

Role: Runs the full 3-model pipeline (Static-PCI, Dynamic-PCI, Regime-Gated) on a
      single pair and produces a comprehensive 4-section visualization.
      Called by generate_all_plots.py for each pair, or run standalone.

Pipeline Steps per Pair:
  1. Fit MLE parameters on formation data (Dynamic-PCI + Static-PCI)
  2. Run 4-state Kalman filter on full series using formation parameters
  3. Compute regime mask and backtest 3 strategies on trading period
  4. Generate 4-section dashboard: prices, beta tracking, M_t spread, equity curves

Dashboard Sections:
  Section 1: Normalized asset prices (base 100) with formation/trading split
  Section 2: Dynamic Kalman β_t vs static β_0 (hedge ratio tracking)
  Section 3: Filtered mean-reverting spread component M_t
  Section 4: Out-of-sample cumulative equity curves (Dynamic, Gated, Static)

Usage:
  python run_demo.py                     # Default pair (KO, PEP) → outputs/pair_KO_PEP.png
  python run_demo.py --ticker1 EWA --ticker2 EWC  # → outputs/pair_EWA_EWC.png
  python run_demo.py --all-pairs         # → one dashboard per pair
"""

import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import (
    fetch_pair_data,
    fetch_all_target_pairs,
    print_data_summary,
    TARGET_PAIRS,
    CANONICAL_PAIRS,
    COMMODITY_ENERGY_PAIRS,
    ALL_PAIRS,
    DEFAULT_FORMATION_START,
    DEFAULT_FORMATION_END,
    DEFAULT_TRADING_START,
    DEFAULT_TRADING_END,
    PairData,
)
from dynamic_pci import fit_dynamic_pci, DynamicPCIKalmanFilter
from backtest import backtest_pair, performance_summary


def plot_unified_pair_dashboard(
    pair_data: PairData,
    out_dyn: dict,
    out_static: dict,
    fitted_dyn,
    fitted_static,
    df_dyn: pd.DataFrame,
    df_static: pd.DataFrame,
    perf_dyn: dict,
    perf_static: dict,
    save_path: str,
    df_gated: pd.DataFrame = None,
    perf_gated: dict = None,
) -> None:
    """
    Generate a single, unified 4-section dashboard plot for the pair.
    """
    t1, t2 = pair_data.ticker1, pair_data.ticker2
    df = pair_data.df_all
    dates_all = pair_data.dates_all
    dates_trade = pair_data.dates_trading
    split_date = pd.to_datetime(pair_data.summary["trading_start_date"])

    norm1 = (df[t1] / df[t1].iloc[0]) * 100.0
    norm2 = (df[t2] / df[t2].iloc[0]) * 100.0

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 14), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.1, 1.1, 1.3]}
    )

    # -------------------------------------------------------------
    # SECTION 1: Normalized Asset Prices & Regimes
    # -------------------------------------------------------------
    axes[0].plot(dates_all, norm1, label=f"{t1} (X1, Base=100)", color="#1f77b4", linewidth=1.7)
    axes[0].plot(dates_all, norm2, label=f"{t2} (X2, Base=100)", color="#ff7f0e", linewidth=1.7)
    axes[0].axvline(split_date, color="#dc2626", linestyle="--", linewidth=1.5,
                    label=f"Trading Start ({split_date.strftime('%Y-%m-%d')})")
    axes[0].axvspan(dates_all[0], split_date, color="#e0f2fe", alpha=0.5, label="Formation Period (In-Sample)")
    axes[0].axvspan(split_date, dates_all[-1], color="#fef3c7", alpha=0.35, label="Trading Period (Out-of-Sample)")
    axes[0].set_title(
        f"[Section 1] Normalized Asset Prices: {t1} vs {t2}  "
        f"(Formation Corr: {pair_data.summary['formation_return_corr']:.2f} | Trading Corr: {pair_data.summary['trading_return_corr']:.2f})",
        fontsize=12, fontweight="bold", pad=8
    )
    axes[0].set_ylabel("Normalized Price (100)", fontsize=10)
    axes[0].legend(loc="upper left", framealpha=0.9, fontsize=9, ncol=2)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # SECTION 2: Dynamic vs Static Hedge Ratio (Beta_t)
    # -------------------------------------------------------------
    axes[1].plot(dates_all, out_dyn["beta"], label=f"Dynamic-PCI Filtered Beta (q_beta={fitted_dyn.q_beta:.2e})",
                 color="#2563eb", linewidth=1.6)
    axes[1].axhline(fitted_static.beta0, label=f"Static-PCI Beta (fixed = {fitted_static.beta0:.4f})",
                 color="#dc2626", linestyle="--", linewidth=1.4)
    axes[1].axvline(split_date, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[1].axvspan(dates_all[0], split_date, color="#e0f2fe", alpha=0.4)
    axes[1].axvspan(split_date, dates_all[-1], color="#fef3c7", alpha=0.3)
    axes[1].set_title(f"[Section 2] Hedge Ratio Tracking: Dynamic Kalman Beta_t vs. Static Estimate",
                      fontsize=12, fontweight="bold", pad=8)
    axes[1].set_ylabel("Hedge Ratio (Beta)", fontsize=10)
    axes[1].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # SECTION 3: Filtered Mean-Reverting Spread Component (M_t)
    # -------------------------------------------------------------
    axes[2].plot(dates_all, out_dyn["M"],
                 label=f"Dynamic-PCI Filtered M_t (rho={fitted_dyn.rho:.2f}, half-life={fitted_dyn.half_life:.1f}d, R2_MR={fitted_dyn.r2_mr:.2f})",
                 color="#059669", linewidth=1.3)
    axes[2].plot(dates_all, out_static["M"],
                 label=f"Static-PCI M_t (half-life={fitted_static.half_life:.1f}d, R2_MR={fitted_static.r2_mr:.2f})",
                 color="#9ca3af", linestyle=":", linewidth=1.1, alpha=0.85)
    axes[2].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axes[2].axvline(split_date, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[2].axvspan(dates_all[0], split_date, color="#e0f2fe", alpha=0.4)
    axes[2].axvspan(split_date, dates_all[-1], color="#fef3c7", alpha=0.3)
    axes[2].set_title(f"[Section 3] Filtered Mean-Reverting Spread Component (M_t)",
                      fontsize=12, fontweight="bold", pad=8)
    axes[2].set_ylabel("M_t", fontsize=10)
    axes[2].legend(loc="upper right", framealpha=0.9, fontsize=9)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # SECTION 4: Out-of-Sample Cumulative Backtest Equity Curve
    # -------------------------------------------------------------
    ret_dyn_pct = perf_dyn['total_return'] * 100
    ret_stat_pct = perf_static['total_return'] * 100
    axes[3].plot(dates_trade, df_dyn["equity"].values,
                 label=f"Dynamic-PCI: Sharpe={perf_dyn['sharpe']:.2f}, Ret={ret_dyn_pct:.1f}%, MaxDD={perf_dyn['max_drawdown']*100:.1f}%",
                 color="#2563eb", linewidth=1.8)
    if df_gated is not None and perf_gated is not None:
        ret_gate_pct = perf_gated['total_return'] * 100
        axes[3].plot(dates_trade, df_gated["equity"].values,
                     label=f"Regime-Gated Dyn: Sharpe={perf_gated['sharpe']:.2f}, Ret={ret_gate_pct:.1f}%, MaxDD={perf_gated['max_drawdown']*100:.1f}%",
                     color="#059669", linewidth=2.0)
    axes[3].plot(dates_trade, df_static["equity"].values,
                 label=f"Static-PCI: Sharpe={perf_static['sharpe']:.2f}, Ret={ret_stat_pct:.1f}%, MaxDD={perf_static['max_drawdown']*100:.1f}%",
                 color="#dc2626", linestyle="--", linewidth=1.5)
    axes[3].axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    axes[3].axvline(split_date, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[3].axvspan(split_date, dates_all[-1], color="#fef3c7", alpha=0.3)
    axes[3].set_title(f"[Section 4] Out-of-Sample Backtest Equity Curve (Cumulative Growth of $1)",
                      fontsize=12, fontweight="bold", pad=8)
    axes[3].set_ylabel("Cumulative Growth ($)", fontsize=10)
    axes[3].set_xlabel("Date", fontsize=10)
    axes[3].legend(loc="best", framealpha=0.9, fontsize=9.5)
    axes[3].grid(True, linestyle=":", alpha=0.6)

    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=150)
        print(f"Saved unified pair dashboard to: {save_path}")
    plt.close(fig)


from regime_filter import test_pair_suitability, compute_online_regime_mask


def run_pair_pipeline(pair_data: PairData, save_image_path: str = None):
    t1, t2 = pair_data.ticker1, pair_data.ticker2
    x1_form, x2_form = pair_data.x1_formation, pair_data.x2_formation
    x1_all, x2_all = pair_data.x1_all, pair_data.x2_all
    form_n = len(x1_form)
    
    print("\n" + "=" * 75)
    print(f" PIPELINE EXECUTION FOR PAIR: {t1} (X1) vs {t2} (X2)")
    print("=" * 75)
    print_data_summary(pair_data)

    # Base-1.0 Normalization (ensures numerical consistency across price levels)
    s1, s2 = float(x1_all[0]), float(x2_all[0])
    x1_norm = x1_all / s1
    x2_norm = x2_all / s2
    x1_form_n = x1_norm[:form_n]
    x2_form_n = x2_norm[:form_n]

    # 1. Fit parameters on the FORMATION period only
    print("\n" + "=" * 75)
    print("STEP 1: Fit parameters on the FORMATION period only (In-Sample)")
    print("=" * 75)
    fitted_dyn, res_dyn = fit_dynamic_pci(x1_form_n, x2_form_n, allow_dynamic_beta=True, maxiter=100)
    fitted_static, res_static = fit_dynamic_pci(x1_form_n, x2_form_n, allow_dynamic_beta=False, maxiter=100)

    print(f"\nDynamic-PCI  : beta0={fitted_dyn.beta0:.4f}  rho={fitted_dyn.rho:.3f}  "
          f"sigma_M={fitted_dyn.sigma_M:.3f}  sigma_R={fitted_dyn.sigma_R:.3f}  "
          f"q_beta={fitted_dyn.q_beta:.2e}  R2_MR={fitted_dyn.r2_mr:.3f}  "
          f"half-life={fitted_dyn.half_life:.1f}d")
    print(f"Static-PCI   : beta0={fitted_static.beta0:.4f}  rho={fitted_static.rho:.3f}  "
          f"sigma_M={fitted_static.sigma_M:.3f}  sigma_R={fitted_static.sigma_R:.3f}  "
          f"(q_beta fixed at 0)  R2_MR={fitted_static.r2_mr:.3f}  "
          f"half-life={fitted_static.half_life:.1f}d")

    ll_dyn, ll_static = -res_dyn.fun, -res_static.fun
    lr = 2 * (ll_dyn - ll_static)
    print(f"\nLog-likelihood: dynamic={ll_dyn:.2f}  static={ll_static:.2f}  "
          f"LR stat={lr:.2f} (1 extra param; compare to chi2(1) crit ~3.84 at 5%)")

    # Statistical Screening
    is_suitable, metrics = test_pair_suitability(x1_form_n, x2_form_n, fitted_dyn, static_params=fitted_static)
    status_label = "PASS (Cointegrated)" if is_suitable else "REJECT (Non-Mean-Reverting)"
    print(f"\nStatistical Screening: R2_MR={metrics['r2_mr']:.4f}, Half-Life={metrics['half_life']:.1f}d, ADF-p={metrics['adf_pvalue']:.4f} -> {status_label}")

    # 2. Filter the full series
    print("\n" + "=" * 75)
    print("STEP 2: Filter the FULL series using formation-period parameters")
    print("=" * 75)
    out_dyn = DynamicPCIKalmanFilter(fitted_dyn).filter(x1_norm, x2_norm)
    out_static = DynamicPCIKalmanFilter(fitted_static).filter(x1_norm, x2_norm)

    # 3. Backtest on OUT-OF-SAMPLE trading period
    print("\n" + "=" * 75)
    print("STEP 3: Backtest on the OUT-OF-SAMPLE trading period")
    print("=" * 75)
    sl = slice(form_n, len(x1_all))
    regime_mask = compute_online_regime_mask(out_dyn["M"][sl], rolling_window=60)
    df_dyn, trades_dyn = backtest_pair(x1_norm[sl], x2_norm[sl], out_dyn["beta"][sl], out_dyn["M"][sl], z_window=80)
    df_static, trades_static = backtest_pair(x1_norm[sl], x2_norm[sl], out_static["beta"][sl], out_static["M"][sl], z_window=80)
    df_gated, trades_gated = backtest_pair(x1_norm[sl], x2_norm[sl], out_dyn["beta"][sl], out_dyn["M"][sl], z_window=80, regime_mask=regime_mask)

    perf_dyn = performance_summary(df_dyn)
    perf_static = performance_summary(df_static)
    perf_gated = performance_summary(df_gated)

    print("\nDynamic-PCI  out-of-sample performance:")
    for k, v in perf_dyn.items():
        print(f"    {k:20s}: {v:.4f}" if isinstance(v, float) else f"    {k:20s}: {v}")
    print(f"    completed trades   : {len(trades_dyn)}")

    print("\nRegime-Gated Dynamic-PCI out-of-sample performance:")
    for k, v in perf_gated.items():
        print(f"    {k:20s}: {v:.4f}" if isinstance(v, float) else f"    {k:20s}: {v}")
    print(f"    completed trades   : {len(trades_gated)}")

    print("\nStatic-PCI   out-of-sample performance:")
    for k, v in perf_static.items():
        print(f"    {k:20s}: {v:.4f}" if isinstance(v, float) else f"    {k:20s}: {v}")
    print(f"    completed trades   : {len(trades_static)}")

    # 4. Save ONE unified picture for this pair containing all sections
    dashboard_name = save_image_path or f"pair_{t1}_{t2}.png"
    plot_unified_pair_dashboard(
        pair_data=pair_data,
        out_dyn=out_dyn,
        out_static=out_static,
        fitted_dyn=fitted_dyn,
        fitted_static=fitted_static,
        df_dyn=df_dyn,
        df_static=df_static,
        perf_dyn=perf_dyn,
        perf_static=perf_static,
        save_path=dashboard_name,
        df_gated=df_gated,
        perf_gated=perf_gated,
    )

    return {
        "pair": (t1, t2),
        "perf_dyn": perf_dyn,
        "perf_gated": perf_gated,
        "perf_static": perf_static,
        "fitted_dyn": fitted_dyn,
        "fitted_static": fitted_static,
        "image_file": dashboard_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Dynamic-PCI Pairs Trading Demo on Real Market Data.")
    parser.add_argument("--ticker1", type=str, default="KO", help="First ticker (X1)")
    parser.add_argument("--ticker2", type=str, default="PEP", help="Second ticker (X2)")
    parser.add_argument("--all-pairs", action="store_true", help="Run full pipeline across all 11 target pairs")
    parser.add_argument("--canonical", action="store_true", help="Run across the 7 canonical equity/ETF pairs")
    parser.add_argument("--commodities", action="store_true", help="Run across the 4 commodity/energy pairs")
    parser.add_argument("--formation-start", type=str, default=DEFAULT_FORMATION_START, help="Formation start (YYYY-MM-DD)")
    parser.add_argument("--formation-end", type=str, default=DEFAULT_FORMATION_END, help="Formation end (YYYY-MM-DD)")
    parser.add_argument("--trading-start", type=str, default=DEFAULT_TRADING_START, help="Trading start (YYYY-MM-DD)")
    parser.add_argument("--trading-end", type=str, default=DEFAULT_TRADING_END, help="Trading end (YYYY-MM-DD)")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    if args.all_pairs or args.canonical or args.commodities:
        if args.canonical:
            run_pairs = CANONICAL_PAIRS
        elif args.commodities:
            run_pairs = COMMODITY_ENERGY_PAIRS
        else:
            run_pairs = ALL_PAIRS

        print(f"\nRunning Dynamic-PCI pairs trading on {len(run_pairs)} target pairs...")
        all_data = fetch_all_target_pairs(
            pairs=run_pairs,
            formation_start=args.formation_start,
            formation_end=args.formation_end,
            trading_start=args.trading_start,
            trading_end=args.trading_end,
        )
        
        results = []
        for (t1, t2), pdata in all_data.items():
            res = run_pair_pipeline(pdata, save_image_path=f"outputs/pair_{t1}_{t2}.png")
            results.append(res)
            
        print("\n" + "=" * 80)
        print("ALL PAIRS OUT-OF-SAMPLE PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"{'Pair':<12} | {'Dyn Sharpe':<12} | {'Stat Sharpe':<12} | {'Dyn Total Ret':<14} | {'Stat Total Ret':<14}")
        print("-" * 80)
        for r in results:
            t1, t2 = r["pair"]
            sd = r["perf_dyn"]["sharpe"]
            ss = r["perf_static"]["sharpe"]
            rd = r["perf_dyn"]["total_return"] * 100
            rs = r["perf_static"]["total_return"] * 100
            print(f"{f'{t1}-{t2}':<12} | {sd:<12.3f} | {ss:<12.3f} | {f'{rd:.2f}%':<14} | {f'{rs:.2f}%':<14}")
        print("=" * 80)
        print("\nGenerated unified plot for each pair:")
        for r in results:
            print(f"  - {r['image_file']}")
    else:
        pair_data = fetch_pair_data(
            ticker1=args.ticker1,
            ticker2=args.ticker2,
            formation_start=args.formation_start,
            formation_end=args.formation_end,
            trading_start=args.trading_start,
            trading_end=args.trading_end,
        )
        run_pair_pipeline(pair_data, save_image_path=f"outputs/pair_{args.ticker1}_{args.ticker2}.png")


if __name__ == "__main__":
    main()
