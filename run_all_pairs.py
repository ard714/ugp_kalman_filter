"""
run_all_pairs.py — BENCHMARK RUNNER: Multi-Model Comparison Across All Pairs
==============================================================================

Role: Runs all 4 model variants on every target pair and produces a performance
      comparison table and comparative diagnostic charts.
      
Models Compared (4-State Dynamic-PCI Architecture):
  1. Static-PCI       : beta fixed after MLE formation fit (Clegg & Krauss baseline)
  2. Dynamic-PCI      : beta evolves as random walk, tracked by 4-state Kalman filter (Yu extension)
  3. Regime-Gated     : Dynamic-PCI + online regime mask (rolling AR(1) half-life + vol ratio gate)
  4. Neural KalmanBOT : GRU learns Kalman gain matrix K_t via two-stage training
                        (Stage 1: MSE reconstruction, Stage 2: differentiable Sharpe)

Statistical Screening:
  Every pair is evaluated against 4 criteria before trading:
    - R2_MR >= 15% (Mean-reverting variance proportion)
    - ADF p-value <= 0.10 (Unit-root rejection)
    - Half-life <= 252 days (Finite reversion time)
    - rho < 0.999 (Stationary AR(1) persistence)

Usage:
  python run_all_pairs.py                  # Full 4-model benchmark on all pairs
  python run_all_pairs.py --quick          # Fast benchmark on 2 pairs (reduced epochs)
  python run_all_pairs.py --pairs KO-PEP   # Specific pair(s)
  python run_all_pairs.py --paper          # Archived 3-state IEEE paper mode
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_loader import (
    fetch_all_target_pairs,
    TARGET_PAIRS,
    CANONICAL_PAIRS,
    COMMODITY_ENERGY_PAIRS,
    ALL_PAIRS,
    DEFAULT_FORMATION_START,
    DEFAULT_FORMATION_END,
    DEFAULT_TRADING_START,
    DEFAULT_TRADING_END,
)
from dynamic_pci import fit_dynamic_pci, DynamicPCIKalmanFilter
from backtest import backtest_pair, performance_summary
from neural_kalman_pci import (
    KalmanNetGain,
    prepare_tensors_4state,
    train_neural_kalman_4state,
    run_neural_filter_4state,
)
from regime_filter import test_pair_suitability, compute_online_regime_mask
from plot_utils import (
    plot_benchmark_metrics,
    plot_multi_pair_equity_grid,
    plot_regime_screening_analysis,
)

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def print_and_save_results(results, model_names, out_path):
    """Format and save performance benchmark results table."""
    header = f"{'Pair':<16} | {'Model':<15} | {'Tot Return':<12} | {'Ann Return':<12} | {'Sharpe':<10} | {'Max DD':<10}"
    sep = "-" * len(header)

    lines = ["FULL PAIRS TRADING PERFORMANCE REPORT", "=" * len(header), header, sep]
    for r in results:
        pair = r["pair"]
        if r.get("pass_strict", False):
            suit_tag = " [PASS]"
        elif r.get("pass_pci_aware", False):
            suit_tag = " [PASS-PCI]"
        else:
            suit_tag = " [REJ]"
        for m_name in model_names:
            perf = r[m_name]
            tot_ret = f"{perf['total_return']*100:.2f}%" if not pd.isna(perf["total_return"]) else "NaN"
            ann_ret = f"{perf['ann_return']*100:.2f}%" if not pd.isna(perf["ann_return"]) else "NaN"
            sharpe = f"{perf['sharpe']:.2f}" if not pd.isna(perf["sharpe"]) else "NaN"
            max_dd = f"{perf['max_drawdown']*100:.2f}%" if not pd.isna(perf["max_drawdown"]) else "NaN"
            lines.append(f"{pair + suit_tag:<16} | {m_name:<15} | {tot_ret:<12} | {ann_ret:<12} | {sharpe:<10} | {max_dd:<10}")
        lines.append(sep)

    output = "\n".join(lines)
    print("\n" + output, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nResults saved to: {out_path}", flush=True)


def run_benchmark(
    pairs=None,
    epochs_stage1: int = 15,
    epochs_stage2: int = 15,
    save_dir: str = OUTPUTS_DIR,
    save_txt: str = "results_comparison.txt",
    save_plots: bool = True,
):
    """Run full 4-model benchmark across target pairs.

    Returns:
      results, model_names, all_equity_data, all_screening_records
    """
    if pairs is None:
        pairs = TARGET_PAIRS

    model_names = ["static", "dyn", "gated", "neural"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nFetching data for {len(pairs)} target pairs...", flush=True)
    all_data = fetch_all_target_pairs(
        pairs=pairs,
        formation_start=DEFAULT_FORMATION_START,
        formation_end=DEFAULT_FORMATION_END,
        trading_start=DEFAULT_TRADING_START,
        trading_end=DEFAULT_TRADING_END,
    )

    results = []
    all_equity_data = {}
    all_screening_records = []

    for (t1, t2), pdata in all_data.items():
        p_name = f"{t1}-{t2}"
        print(f"\n{'='*20} PAIR: {p_name} {'='*20}", flush=True)

        # Base-1.0 Normalization
        s1, s2 = float(pdata.x1_all[0]), float(pdata.x2_all[0])
        x1_norm = pdata.x1_all / s1
        x2_norm = pdata.x2_all / s2
        form_n = len(pdata.x1_formation)

        x1_form = x1_norm[:form_n]
        x2_form = x2_norm[:form_n]
        x1_test = x1_norm[form_n:]
        x2_test = x2_norm[form_n:]
        dates_trade = pdata.dates_trading

        # 1. Classical State-Space Models
        dyn_params, _ = fit_dynamic_pci(x1_form, x2_form, allow_dynamic_beta=True, maxiter=100)
        static_params, _ = fit_dynamic_pci(x1_form, x2_form, allow_dynamic_beta=False, maxiter=100)
        dyn_pci = DynamicPCIKalmanFilter(dyn_params)
        static_pci = DynamicPCIKalmanFilter(static_params)

        out_dyn = dyn_pci.filter(x1_test, x2_test)
        out_static = static_pci.filter(x1_test, x2_test)

        # 2. Statistical Suitability Screening
        is_suitable, metrics = test_pair_suitability(x1_form, x2_form, dyn_params, static_params=static_params)
        status_label = "PASS (Cointegrated)" if is_suitable else "REJECT (Non-Mean-Reverting)"
        print(
            f"  [Screening] R2_MR={metrics['r2_mr']:.4f}, Half-Life={metrics['half_life']:.1f}d, "
            f"ADF-p={metrics['adf_pvalue']:.4f} -> {status_label}",
            flush=True,
        )

        # 3. Dynamic online regime mask
        regime_mask = compute_online_regime_mask(out_dyn["M"], rolling_window=60)

        # 4. Neural KalmanNet Training
        print(f"  Training Neural KalmanBOT for {p_name} (epochs={epochs_stage1}+{epochs_stage2})...", flush=True)
        batch_y, batch_returns, batch_H = prepare_tensors_4state(x1_form, x2_form, dyn_pci)
        dataset = TensorDataset(batch_y, batch_returns, batch_H)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model = KalmanNetGain(state_dim=4, obs_dim=2, hidden_dim=32)
        model = train_neural_kalman_4state(
            model,
            dataloader,
            dyn_pci,
            epochs_stage1=epochs_stage1,
            epochs_stage2=epochs_stage2,
            verbose=False,
        )

        beta_neural, M_neural = run_neural_filter_4state(model, x1_test, x2_test, dyn_pci)

        # 5. Backtests
        df_static, _ = backtest_pair(x1_test, x2_test, out_static["beta"], out_static["M"], z_window=80)
        df_dyn, _    = backtest_pair(x1_test, x2_test, out_dyn["beta"],    out_dyn["M"],    z_window=80)
        df_gated, _  = backtest_pair(x1_test, x2_test, out_dyn["beta"],    out_dyn["M"],    z_window=80, regime_mask=regime_mask)
        df_neural, _ = backtest_pair(x1_test, x2_test, beta_neural,        M_neural,        z_window=80)

        all_equity_data[p_name] = {
            "static": pd.Series(df_static["equity"].values, index=dates_trade),
            "dyn":    pd.Series(df_dyn["equity"].values, index=dates_trade),
            "gated":  pd.Series(df_gated["equity"].values, index=dates_trade),
            "neural": pd.Series(df_neural["equity"].values, index=dates_trade),
        }
        all_screening_records.append({
            "pair": p_name,
            "r2_mr": metrics["r2_mr"],
            "half_life": metrics["half_life"],
            "adf_pvalue": metrics["adf_pvalue"],
            "return_corr": metrics["return_corr"],
            "is_suitable": is_suitable,
        })

        results.append({
            "pair":     p_name,
            "static":   performance_summary(df_static),
            "dyn":      performance_summary(df_dyn),
            "gated":    performance_summary(df_gated),
            "neural":   performance_summary(df_neural),
            "suitable": is_suitable,
            "pass_strict": metrics.get("pass_strict", False),
            "pass_pci_aware": metrics.get("pass_pci_aware", False),
        })

    # Save benchmark text summary
    if save_txt:
        out_txt_path = os.path.join(save_dir, save_txt)
        print_and_save_results(results, model_names, out_txt_path)

    # Generate comparative plots
    if save_plots:
        print("\nRendering comparative visualizations...", flush=True)
        plot_benchmark_metrics(results, model_names, save_dir=save_dir)
        plot_multi_pair_equity_grid(all_equity_data, save_path=os.path.join(save_dir, "all_pairs_equity_grid.png"))
        plot_regime_screening_analysis(all_screening_records, save_path=os.path.join(save_dir, "regime_screening_analysis.png"))

    return results, model_names, all_equity_data, all_screening_records


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Dynamic-PCI Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run full benchmark across all 11 cross-asset pairs (default)")
    parser.add_argument("--canonical", action="store_true", help="Run only the 7 canonical equity/ETF pairs")
    parser.add_argument("--commodities", action="store_true", help="Run only the 4 commodity/energy pairs")
    parser.add_argument("--pairs", nargs="+", default=None, help="Specific pairs to benchmark (e.g. --pairs KO-PEP GLD-SLV)")
    parser.add_argument("--quick", action="store_true", help="Quick run on 2 pairs with 5 epochs")
    parser.add_argument("--epochs1", type=int, default=None, help="Stage 1 MSE epochs")
    parser.add_argument("--epochs2", type=int, default=None, help="Stage 2 Sharpe epochs")
    parser.add_argument("--out", default="results_comparison.txt", help="Output text report name")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot rendering")
    parser.add_argument("--paper", action="store_true", help="Run archived 3-state IEEE paper mode")
    args = parser.parse_args()

    if args.paper:
        from archive.paper_replication_3state import run_paper_mode
        print("Running archived 3-state IEEE paper replication mode...")
        run_paper_mode(out_txt=os.path.join(OUTPUTS_DIR, "results_paper.txt"))
        return

    if args.pairs:
        pairs = []
        for p in args.pairs:
            if "-" in p:
                t1, t2 = p.split("-")
                pairs.append((t1.strip().upper(), t2.strip().upper()))
            elif "_" in p:
                t1, t2 = p.split("_")
                pairs.append((t1.strip().upper(), t2.strip().upper()))
    elif args.canonical:
        pairs = CANONICAL_PAIRS
    elif args.commodities:
        pairs = COMMODITY_ENERGY_PAIRS
    elif args.quick:
        pairs = [("KO", "PEP"), ("AAP", "IBM")]
    else:
        pairs = ALL_PAIRS  # Runs all 11 pairs by default

    ep1 = args.epochs1 if args.epochs1 is not None else (5 if args.quick else 15)
    ep2 = args.epochs2 if args.epochs2 is not None else (5 if args.quick else 15)

    run_benchmark(
        pairs=pairs,
        epochs_stage1=ep1,
        epochs_stage2=ep2,
        save_dir=OUTPUTS_DIR,
        save_txt=args.out,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
