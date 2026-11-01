"""
generate_all_plots.py — PLOT ORCHESTRATOR: Master Visualization Generator
==========================================================================

Role: Coordinates the generation of ALL publication-quality figures by delegating
      to the appropriate runner modules. Called by main.py, or run standalone.
      Does NOT contain model logic — only orchestration.

Delegates to:
  - simulate.py       → Synthetic simulation & parameter recovery plots
  - run_neural_demo.py → Neural vs Classical comparison equity curve
  - run_demo.py        → Per-pair 4-section dashboards
  - run_all_pairs.py   → Full 4-model benchmark → bar charts, equity grid, screening

Generated Figures:
  1. simulation_results.png        — 6-panel synthetic DGP recovery diagnostic
  2. pair_<T1>_<T2>.png            — Per-pair dashboard (prices, β, M_t, equity)
  3. neural_vs_classical.png       — Neural KalmanBOT vs Classical equity curves
  4. benchmark_sharpe_comparison.png  — Cross-pair Sharpe ratio bar chart
  5. benchmark_returns_comparison.png — Cross-pair total return bar chart
  6. benchmark_drawdown_comparison.png — Cross-pair max drawdown bar chart
  7. benchmark_summary_dashboard.png  — Executive 3-in-1 multi-metric dashboard
  8. all_pairs_equity_grid.png     — Multi-pair equity curves grid
  9. regime_screening_analysis.png — 4-panel screening diagnostic

Usage:
  python generate_all_plots.py             # Generate all plots
  python generate_all_plots.py --simulation # Only simulation recovery
  python generate_all_plots.py --dashboards # Only single-pair dashboards
  python generate_all_plots.py --benchmark  # Only cross-pair benchmark
"""

import os
import argparse
import numpy as np
import pandas as pd

from data_loader import (
    fetch_pair_data,
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
from regime_filter import test_pair_suitability, compute_online_regime_mask
from plot_utils import (
    plot_detailed_simulation,
    plot_benchmark_metrics,
    plot_multi_pair_equity_grid,
    plot_regime_screening_analysis,
)
from simulate import simulate_dynamic_pci
from run_demo import run_pair_pipeline
from run_neural_demo import compare_models


OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def generate_simulation_plots(n: int = 1250, beta_path: str = "drift", seed: int = 42):
    """Run synthetic DGP, estimate via MLE, filter, and save detailed 6-panel diagnostic."""
    print("\n" + "=" * 70)
    print(" [1/4] GENERATING SYNTHETIC SIMULATION & STATE RECOVERY PLOTS")
    print("=" * 70)
    print(f"Simulating Dynamic-PCI DGP (n={n}, beta_path='{beta_path}', seed={seed})...")
    sim_data = simulate_dynamic_pci(n=n, beta_path=beta_path, seed=seed)
    x1, x2 = sim_data["x1"], sim_data["x2"]

    print("Fitting 4-State MLE Parameters...")
    fitted_params, _ = fit_dynamic_pci(x1, x2, allow_dynamic_beta=True)

    print("Running Kalman Filter forward pass...")
    kf = DynamicPCIKalmanFilter(fitted_params)
    filtered = kf.filter(x1, x2)

    save_path = os.path.join(OUTPUTS_DIR, "simulation_results.png")
    plot_detailed_simulation(sim_data, filtered, fitted_params, save_path=save_path)
    print(f"SUCCESS: Simulation diagnostic plot created at: {save_path}")


def generate_neural_demo_plot(ticker1="KO", ticker2="PEP"):
    """Run Neural KalmanBOT vs Classical Dynamic-PCI comparison."""
    print("\n" + "=" * 70)
    print(" [2/4] GENERATING NEURAL KALMANBOT VS CLASSICAL COMPARISON PLOT")
    print("=" * 70)
    save_path = os.path.join(OUTPUTS_DIR, "neural_vs_classical.png")
    compare_models(ticker1=ticker1, ticker2=ticker2, output_plot=save_path)


def generate_pair_dashboards(pairs=None):
    """Generate 4-section unified dashboard plots for target pairs."""
    if pairs is None:
        pairs = TARGET_PAIRS

    print("\n" + "=" * 70)
    print(f" [3/4] GENERATING UNIFIED PAIR DASHBOARDS FOR {len(pairs)} PAIRS")
    print("=" * 70)

    for t1, t2 in pairs:
        print(f"\nProcessing Pair: {t1} vs {t2}...")
        try:
            pdata = fetch_pair_data(t1, t2)
            out_img = os.path.join(OUTPUTS_DIR, f"pair_{t1}_{t2}.png")
            run_pair_pipeline(pdata, save_image_path=out_img)
        except Exception as e:
            print(f"Error processing {t1}-{t2}: {e}")


def generate_full_benchmark_plots(pairs=None, epochs_stage1: int = 15, epochs_stage2: int = 15):
    """Run full benchmark comparing Static, Dyn, Gated, Neural and generate cross-pair visual charts."""
    if pairs is None:
        pairs = TARGET_PAIRS

    print("\n" + "=" * 70, flush=True)
    print(f" [4/4] GENERATING FULL CROSS-PAIR BENCHMARK & REGIME PLOTS ({len(pairs)} PAIRS)", flush=True)
    print("=" * 70, flush=True)

    from run_all_pairs import run_benchmark
    results, model_names, all_equity_data, all_screening_records = run_benchmark(
        pairs=pairs,
        epochs_stage1=epochs_stage1,
        epochs_stage2=epochs_stage2,
        save_dir=OUTPUTS_DIR,
        save_txt="results_comparison.txt",
        save_plots=True,
    )
    print("\nSUCCESS: All benchmark charts and diagnostic figures generated in outputs/", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Master Plot Generation Utility for Dynamic-PCI")
    parser.add_argument("--all", action="store_true", help="Generate all simulation, demo, and benchmark plots across 11 pairs")
    parser.add_argument("--canonical", action="store_true", help="Generate plots for 7 canonical equity/ETF pairs only")
    parser.add_argument("--commodities", action="store_true", help="Generate plots for 4 commodity/energy pairs only")
    parser.add_argument("--quick", action="store_true", help="Fast validation mode on subset of pairs (reduced epochs)")
    parser.add_argument("--simulation", action="store_true", help="Generate synthetic simulation diagnostic plot")
    parser.add_argument("--dashboards", action="store_true", help="Generate single-pair dashboards")
    parser.add_argument("--neural", action="store_true", help="Generate neural vs classical comparison plot")
    parser.add_argument("--benchmark", action="store_true", help="Generate multi-pair benchmark comparison plots")
    args = parser.parse_args()

    # Default to all if no specific flag passed
    run_all = args.all or not (args.simulation or args.dashboards or args.neural or args.benchmark)

    if args.quick:
        bench_pairs = [("KO", "PEP"), ("AAP", "IBM")]
    elif args.canonical:
        bench_pairs = CANONICAL_PAIRS
    elif args.commodities:
        bench_pairs = COMMODITY_ENERGY_PAIRS
    else:
        bench_pairs = ALL_PAIRS  # Runs all 11 pairs by default

    ep1 = 5 if args.quick else 15
    ep2 = 5 if args.quick else 15

    if run_all or args.simulation:
        generate_simulation_plots()

    if run_all or args.neural:
        generate_neural_demo_plot()

    if run_all or args.dashboards:
        generate_pair_dashboards(pairs=bench_pairs if args.quick else None)

    if run_all or args.benchmark:
        generate_full_benchmark_plots(pairs=bench_pairs, epochs_stage1=ep1, epochs_stage2=ep2)

    print("\n" + "=" * 70, flush=True)
    print(" ALL PLOT GENERATION TASKS COMPLETED SUCCESSFULLY", flush=True)
    print(f" Figures saved in: {os.path.abspath(OUTPUTS_DIR)}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
