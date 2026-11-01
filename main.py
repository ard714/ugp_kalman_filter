"""
main.py — MASTER ORCHESTRATOR: End-to-End Pipeline for Dynamic-PCI Pairs Trading
=================================================================================

Role: Top-level entry point. Orchestrates the entire research pipeline by calling
      into dedicated helper modules. Does NOT contain model logic itself.

What This Project Does
----------------------
Implements and compares multiple state-space approaches to statistical arbitrage
(pairs trading) using Kalman filtering, combining ideas from:
  • Yu (2022): dynamic hedge ratio β_t tracked as a random walk via Kalman filter
  • Clegg & Krauss (2018): partial cointegration with mean-reverting + random-walk
    spread decomposition (M_t + R_t)
  • KalmanBOT (IEEE): GRU-based neural network that learns the Kalman gain matrix

Screening (What Pairs Are Suitable?)
-------------------------------------
Before trading, each pair undergoes a statistical suitability screen
(see regime_filter.py → test_pair_suitability):
  1. R²_MR ≥ 15% — at least 15% of spread variance must be mean-reverting
  2. ADF p-value ≤ 0.10 — Augmented Dickey-Fuller test rejects unit root in spread
  3. Half-life ≤ 252 days — mean reversion must occur within 1 trading year
  4. ρ < 0.999 — AR(1) coefficient must imply finite mean reversion speed
Pairs failing any criterion (e.g., AAP-IBM) are flagged as "REJECTED".

Models Compared
---------------
Four model variants are benchmarked on every pair:
  1. Static-PCI    : β fixed at MLE estimate from formation period (Clegg & Krauss baseline)
  2. Dynamic-PCI   : β evolves as random walk, tracked by 4-state Kalman filter (Yu extension)
  3. Regime-Gated  : Dynamic-PCI + online regime mask (rolling AR(1) half-life + vol ratio)
  4. Neural KalmanBOT : GRU replaces the analytic Kalman gain; two-stage training
                        (Stage 1: MSE on X2 reconstruction, Stage 2: differentiable Sharpe)

Pipeline Steps
--------------
  Step 1: Synthetic simulation → MLE parameter recovery → validate the filter works
  Step 2: Neural KalmanBOT vs Classical Dynamic-PCI side-by-side on one pair
  Step 3: Generate individual pair dashboards (prices, β tracking, M_t, equity curves)
  Step 4: Run full 4-model benchmark across all pairs → bar charts, equity grid, screening

Generated Outputs (in outputs/)
-------------------------------
  simulation_results.png       — Synthetic data recovery: ground truth vs filter estimates
  regime_screening_analysis.png — 4-panel diagnostic: R²_MR, half-life, ADF, correlation
  pair_<T1>_<T2>.png           — Per-pair dashboard (prices, beta, M_t, equity)
  neural_vs_classical.png      — Head-to-head equity curve: Neural KalmanBOT vs Classical
  benchmark_sharpe_comparison.png  — Sharpe ratio grouped bar chart across pairs
  benchmark_returns_comparison.png — Total return grouped bar chart
  benchmark_drawdown_comparison.png — Max drawdown grouped bar chart
  benchmark_summary_dashboard.png  — Executive 3-in-1 dashboard
  all_pairs_equity_grid.png    — Multi-pair equity curves in a grid
  results_comparison.txt       — Tab-formatted performance table
  pipeline.log                 — Full execution log with timestamps

Usage
-----
  python main.py                  # Full master pipeline across all canonical pairs
  python main.py --quick          # Fast validation mode (2 pairs, reduced epochs, ~30s)
  python main.py --pairs KO-PEP   # Run specific pair(s)

Module Dependency Map
---------------------
  main.py (orchestrator)
    ├── data_loader.py          (data ingestion & pair configuration)
    ├── generate_all_plots.py   (master plot generator, delegates to:)
    │     ├── simulate.py         (synthetic DGP)
    │     ├── run_demo.py         (per-pair pipeline runner)
    │     ├── run_neural_demo.py  (neural vs classical comparison)
    │     └── run_all_pairs.py    (multi-model benchmark runner)
    ├── dynamic_pci.py          (core 4-state Kalman filter & MLE)
    ├── neural_kalman_pci.py    (GRU-based KalmanNet gain + Sharpe loss)
    ├── backtest.py             (Z-score trading strategy & performance metrics)
    ├── regime_filter.py        (pair screening + online regime gating)
    └── plot_utils.py           (all plotting/visualization functions)
"""

import os
import sys
import time
import argparse
from datetime import datetime


class TeeLogger:
    """Simultaneously writes and immediately flushes output to console and log file."""
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.terminal = sys.stdout
        self.log_file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def print_banner(mode_str: str):
    banner = f"""
================================================================================
 DYNAMIC PARTIAL COINTEGRATION & NEURAL KALMANNET END-TO-END PIPELINE
 Mode: {mode_str} | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""
    print(banner, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Master Execution Pipeline for Dynamic-PCI")
    parser.add_argument("--all", action="store_true", help="Run full pipeline across all 11 cross-asset pairs (default)")
    parser.add_argument("--canonical", action="store_true", help="Run only the 7 canonical equity/ETF pairs")
    parser.add_argument("--commodities", action="store_true", help="Run only the 4 commodity/energy pairs")
    parser.add_argument("--quick", action="store_true", help="Fast validation mode on subset of pairs (reduced epochs)")
    parser.add_argument("--pairs", nargs="+", default=None, help="Specific pairs to evaluate (e.g. --pairs KO-PEP GLD-SLV)")
    parser.add_argument("--epochs1", type=int, default=None, help="Stage 1 MSE pre-training epochs")
    parser.add_argument("--epochs2", type=int, default=None, help="Stage 2 Sharpe fine-tuning epochs")
    parser.add_argument("--log", type=str, default="outputs/pipeline.log", help="Path to output execution log file")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    # Initialize Tee Logging
    tee = TeeLogger(args.log)
    sys.stdout = tee
    sys.stderr = tee

    start_time = time.time()

    try:
        from data_loader import (
            TARGET_PAIRS,
            CANONICAL_PAIRS,
            COMMODITY_ENERGY_PAIRS,
            ALL_PAIRS,
        )
        from generate_all_plots import (
            generate_simulation_plots,
            generate_neural_demo_plot,
            generate_pair_dashboards,
            generate_full_benchmark_plots,
        )

        # Determine target pairs (defaults to all 14 pairs: 11 benchmark + 3 passing)
        if args.pairs:
            selected_pairs = []
            for p in args.pairs:
                if "-" in p:
                    t1, t2 = p.split("-")
                    selected_pairs.append((t1.strip().upper(), t2.strip().upper()))
                elif "_" in p:
                    t1, t2 = p.split("_")
                    selected_pairs.append((t1.strip().upper(), t2.strip().upper()))
                else:
                    print(f"Warning: Skipping unrecognized pair format '{p}', expected 'T1-T2'.")
        elif args.canonical:
            selected_pairs = CANONICAL_PAIRS
        elif args.commodities:
            selected_pairs = COMMODITY_ENERGY_PAIRS
        elif args.quick:
            selected_pairs = [("KO", "PEP"), ("AAP", "IBM")]
        else:
            selected_pairs = ALL_PAIRS

        if args.quick:
            mode_str = "QUICK VALIDATION (2 Pairs, Fast Mode)"
        elif args.canonical:
            mode_str = "CANONICAL BENCHMARK (7 Equity/ETF Pairs)"
        elif args.commodities:
            mode_str = "COMMODITY BENCHMARK (4 Commodity/Energy Pairs)"
        else:
            mode_str = f"FULL CROSS-ASSET BENCHMARK ({len(selected_pairs)} Pairs)"

        print_banner(mode_str)

        # Epoch configuration
        ep1 = args.epochs1 if args.epochs1 is not None else (5 if args.quick else 15)
        ep2 = args.epochs2 if args.epochs2 is not None else (5 if args.quick else 15)

        print(f"Configuration:")
        print(f" - Evaluated Pairs : {[f'{t1}-{t2}' for t1, t2 in selected_pairs]}")
        print(f" - Neural Epochs   : Stage 1 = {ep1}, Stage 2 = {ep2}")
        print(f" - Transaction Cost: 10 bps per one-way trade")
        print(f" - Output Directory: {os.path.abspath(outputs_dir)}")
        print(f" - Execution Log   : {os.path.abspath(args.log)}\n", flush=True)

        # ----------------------------------------------------------------------
        # 1. Synthetic Simulation Recovery
        # ----------------------------------------------------------------------
        print(">>> STEP 1: Running Ground-Truth Synthetic Simulation & Parameter Recovery...", flush=True)
        generate_simulation_plots(n=1250, beta_path="drift", seed=42)

        # ----------------------------------------------------------------------
        # 2. Neural KalmanBOT vs Classical Side-by-Side Comparison
        # ----------------------------------------------------------------------
        demo_t1, demo_t2 = selected_pairs[0]
        print(f"\n>>> STEP 2: Running Neural KalmanBOT vs Classical Dynamic-PCI Comparison ({demo_t1} vs {demo_t2})...", flush=True)
        generate_neural_demo_plot(ticker1=demo_t1, ticker2=demo_t2)

        # ----------------------------------------------------------------------
        # 3. Individual Pair Unified Dashboards
        # ----------------------------------------------------------------------
        print("\n>>> STEP 3: Generating Unified Dashboards for Target Pairs...", flush=True)
        generate_pair_dashboards(pairs=selected_pairs)

        # ----------------------------------------------------------------------
        # 4. Multi-Model Benchmark, Screening & Comparative Visualizations
        # ----------------------------------------------------------------------
        print("\n>>> STEP 4: Running Multi-Model Benchmark Across Evaluated Pairs...", flush=True)
        generate_full_benchmark_plots(pairs=selected_pairs, epochs_stage1=ep1, epochs_stage2=ep2)

        elapsed = time.time() - start_time

        # ----------------------------------------------------------------------
        # 5. Final Report & Artifact Manifest
        # ----------------------------------------------------------------------
        print("\n" + "=" * 80, flush=True)
        print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY", flush=True)
        print(f" Total Elapsed Time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)", flush=True)
        print("=" * 80, flush=True)

        # Display performance table if generated
        res_file = os.path.join(outputs_dir, "results_comparison.txt")
        if os.path.exists(res_file):
            print("\n" + "-" * 80, flush=True)
            print(" FINAL PERFORMANCE BENCHMARK SUMMARY (from results_comparison.txt):", flush=True)
            print("-" * 80, flush=True)
            with open(res_file, "r", encoding="utf-8") as f:
                print(f.read().strip(), flush=True)
            print("-" * 80, flush=True)

        # Artifact Manifest
        print("\nGENERATED OUTPUT ARTIFACTS IN outputs/:", flush=True)
        out_files = sorted(os.listdir(outputs_dir))
        for fname in out_files:
            fpath = os.path.join(outputs_dir, fname)
            size_kb = os.path.getsize(fpath) / 1024.0
            print(f"  * {fname:<35} ({size_kb:>8.1f} KB)", flush=True)

        print("\nExecution complete. Log preserved at:", os.path.abspath(args.log), flush=True)

    except Exception as e:
        print(f"\nERROR: Pipeline execution encountered an unhandled exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout = tee.terminal
        sys.stderr = tee.terminal
        tee.flush()
        tee.close()


if __name__ == "__main__":
    main()
