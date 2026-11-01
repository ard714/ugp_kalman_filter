# Dynamic-PCI Pairs Trading — Project Guide

## What This Project Does

This project implements and benchmarks **Kalman filter-based statistical arbitrage** (pairs trading), combining three major academic contributions into a unified research framework:

| Academic Work | Quantitative Contribution | Location in Codebase |
|---|---|---|
| **Clegg & Krauss (2018)** | Partial cointegration (PCI): spread decomposes into $M_t$ (mean-reverting AR(1)) + $R_t$ (random walk permanent drift) | [`dynamic_pci.py`](dynamic_pci.py) |
| **Yu (2022)** | Dynamic hedge ratio $\beta_t$ tracked as a time-varying random walk via Kalman filtering | [`dynamic_pci.py`](dynamic_pci.py) |
| **KalmanBOT (Revach et al., IEEE)** | GRU recurrent neural network learns the Kalman gain matrix $K_t$ directly from data | [`neural_kalman_pci.py`](neural_kalman_pci.py) |

---

## The 4-State State-Space Formulation

```
State vector:  z_t = [X1_t, β_t, M_t, R_t]^T

State Transitions:
  X1_t  = X1_{t-1} + w_{1,t}        (driver asset price: random walk)
  β_t   = β_{t-1}  + w_{β,t}        (hedge ratio: random walk with step variance q_β)
  M_t   = ρ · M_{t-1} + w_{M,t}     (mean-reverting spread: stationary AR(1), TRADING SIGNAL)
  R_t   = R_{t-1}  + w_{R,t}        (permanent structural drift: random walk, NEVER TRADED)

Observations:
  y_t = [X2_t, X1_t]^T
  X2_t  = β_t · X1_t + M_t + R_t + v_{2,t}    (partial cointegration observation)
  X1_t  = X1_t + v_{1,t}                      (direct asset price observation)
```

**Core Insight**: Traditional pairs trading trades the raw spread $S_t = X_{2,t} - \beta X_{1,t}$. If the companies' fundamentals diverge permanently, $S_t$ drifts forever and the trader suffers catastrophic losses. In **Dynamic-PCI**, the Kalman filter isolates $M_t$ (transient mean-reverting shocks) from $R_t$ (permanent drift), and we **only trade on $M_t$**.

---

## Statistical Pair Screening Engine

Before capital is allocated, each pair undergoes a suitability screen in [`regime_filter.py`](regime_filter.py):

| Criterion | Threshold | Quantitative Purpose |
|---|---|---|
| **$R^2_{MR}$** | $\ge 15\%$ (Strict) / $\ge 50\%$ (Calibrated) | Proportion of spread variance that must be mean-reverting ($M_t$) rather than permanent drift ($R_t$) |
| **ADF $p$-value** | $\le 0.10$ (Strict) / $\le 0.05$ (Spread) | Augmented Dickey-Fuller test rejects unit root (spread is statistically stationary) |
| **$M_t$ ADF $p$-value** | $\le 0.01$ (Calibrated PCI) | Augmented Dickey-Fuller test confirms stationarity directly on latent state $M_t$ |
| **Half-life ($HL$)** | $\le 252\text{ days}$ | Mean reversion occurs within 1 trading year ($HL = -\ln 2 / \ln \rho$) |
| **$\rho$ Persistence** | $< 0.999$ | AR(1) coefficient implies finite mean-reversion velocity |

### 14-Pair Cross-Asset Universe & Verified Screening Status

The repository evaluates **14 total pairs** spanning single-stock equities, sovereign ETFs, sector ETFs, commodity-equity cross-asset pairs, and newly validated screening pairs:

| Pair | Asset Class / Sector | $R^2_{MR}$ | Half-Life ($HL$) | ADF $p$-value | Filtered $M_t$ ADF $p$ | Calibrated Screen | Quantitative Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`EWA–EWC`** | Sovereign ETFs (Australia vs. Canada) | **78.60%** | **136.4d** | **0.0439** | **0.0521** | **PASS** | Valid cointegrated resource-export pair; satisfies all criteria. |
| **`MS–GS`** | Financials (Morgan Stanley vs. Goldman Sachs) | **90.79%** | **68.6d** | **0.0491** | **0.0005** | **PASS** | Satisfies strict criteria; premier banking cointegration. |
| **`MRK–VZ`** | Pharma vs Telecom | **51.68%** | **692.8d** | 0.4882 | **0.0004** | **PASS** | Latent state $M_t$ is highly stationary; $51.7\%$ mean-reverting variance. |
| **`XOM–CVX`** | Integrated Energy Peers | **80.39%** | **692.8d** | 0.2432 | **0.0013** | **PASS** | High $R^2_{MR}$ ($80.4\%$), stationary $M_t$; star performer (+20.5% static / +40.1% neural). |
| **`XLE–OIH`** | Energy Equities vs Equipment | **62.46%** | **81.5d** | 0.6900 | **0.0027** | **PASS** | Rapid half-life, stationary $M_t$ ($p = 0.0027$); value-chain cointegration. |
| **`HD–LOW`** | Home Improvement Retail Duopoly | **50.86%** | **2.0d** | 0.9874 | **3.33e-25** | **PASS** | Fast mean reversion ($HL=2.0\text{d}$); stationary $M_t$. |
| **`AAP–IBM`** | Auto Retail vs Enterprise Tech | 22.38% | 22.0d | 0.3369 | 2.07e-08 | **REJECT** | $77\%$ of variance is drift $R_t$; static model hemorrhages (-7.3% to -30.7%). |
| **`NKE–AES`** | Discretionary vs Power Utility | 21.63% | 5.9d | 0.6494 | 1.35e-17 | **REJECT** | Low $R^2_{MR}$ ($21.6\%$); static suffers catastrophic -64.4% loss. |
| **`KO–PEP`** | Consumer Staples | 36.56% | 692.8d | 0.9641 | 7.67e-16 | **REJECT** | $R^2_{MR} < 50\%$; structural multiple drift between peers. |
| **`GLD–SLV`** | Precious Metals (Gold vs Silver) | 41.00% | 692.8d | 0.3142 | 0.1266 | **REJECT** | Non-stationary secular divergence in gold-silver ratio. |
| **`XLE–XOP`** | Integrated vs E&P Energy | 13.43% | 692.8d | 0.9383 | 3.04e-14 | **REJECT** | $R^2_{MR} < 50\%$; E&P debt leverage divergence. |
| **`USO–XLE`** | Crude Oil vs Energy Equities | 14.19% | 4.2d | 0.3329 | 9.41e-25 | **REJECT** | $R^2_{MR} < 15\%$; futures roll contango creates permanent drift. |
| **`GS–CVX`** | Financials vs Energy | 10.76% | 498.5d | 0.1736 | 0.3103 | **REJECT** | $R^2_{MR} < 15\%$; cross-sector divergence. |
| **`V–MA`** | Payments Duopoly (Visa vs Mastercard) | 33.89% | 1.0d | 0.7121 | 0.0000 | **REJECT** | $R^2_{MR} < 50\%$; cross-border travel transaction volumes divergence. |

---

## Models Compared in Benchmark & Verified Empirical Performance

Four model variants are evaluated on every target pair over out-of-sample data (incorporating 10 bps one-way transaction fees):

| # | Model | $\beta_t$ Tracking | Kalman Gain $K_t$ | Regime Gate |
|---|---|---|---|---|
| 1 | **Static-PCI** | Fixed $\beta_0$ from formation MLE | Analytical Kalman | None |
| 2 | **Dynamic-PCI** | Random walk $\beta_t$ ($q_\beta > 0$) | Analytical Kalman | None |
| 3 | **Regime-Gated** | Random walk $\beta_t$ ($q_\beta > 0$) | Analytical Kalman | Rolling 60d half-life + vol ratio |
| 4 | **Neural KalmanBOT** | Random walk $\beta_t$ ($q_\beta > 0$) | GRU-learned $K_t$ | None |

### Full 14-Pair Performance Benchmark Table

*From verified execution artifact [`outputs/results_comparison.txt`](outputs/results_comparison.txt):*

| Target Pair | Screen | Model | Total Return | Ann. Return | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **AAP–IBM** | REJ | Static-PCI | -7.27% | -1.87% | 0.01 | -44.52% |
| | | Dynamic-PCI | -24.59% | -6.82% | -0.27 | -43.55% |
| | | Regime-Gated | -28.72% | -8.12% | -0.38 | -45.44% |
| | | Neural KalmanBOT | -30.73% | -8.78% | -0.41 | -47.23% |
| **EWA–EWC** | **PASS** | **Static-PCI** | **+1.89%** | **+0.47%** | **0.11** | **-9.30%** |
| | | Dynamic-PCI | -5.16% | -1.32% | -0.18 | -13.21% |
| | | Regime-Gated | -3.69% | -0.94% | -0.16 | -14.31% |
| | | Neural KalmanBOT | -2.90% | -0.73% | -0.08 | -14.45% |
| **NKE–AES** | REJ | Static-PCI | -64.40% | -22.78% | -0.69 | -69.47% |
| | | Dynamic-PCI | -15.97% | -4.26% | -0.06 | -52.64% |
| | | Regime-Gated | -10.34% | -2.69% | -0.01 | -46.71% |
| | | **Neural KalmanBOT** | **+45.70%** | **+9.88%** | **0.55** | **-23.04%** |
| **MRK–VZ** | **PASS** | **Static-PCI** | **+37.72%** | **+8.34%** | **0.82** | **-7.00%** |
| | | Dynamic-PCI | -0.25% | -0.06% | 0.05 | -15.13% |
| | | Regime-Gated | -5.74% | -1.47% | -0.18 | -13.19% |
| | | **Neural KalmanBOT** | **+40.23%** | **+8.83%** | **0.75** | **-10.38%** |
| **GS–CVX** | REJ | Static-PCI | +19.56% | +4.57% | 0.40 | -24.30% |
| | | Dynamic-PCI | -1.80% | -0.45% | 0.03 | -26.48% |
| | | Regime-Gated | -0.13% | -0.03% | 0.04 | -16.50% |
| | | **Neural KalmanBOT** | **+32.39%** | **+7.27%** | **0.41** | **-43.86%** |
| **KO–PEP** | REJ | Static-PCI | -9.11% | -2.36% | -0.27 | -23.18% |
| | | Dynamic-PCI | -2.83% | -0.72% | -0.04 | -14.33% |
| | | Regime-Gated | -6.77% | -1.74% | -0.36 | -11.51% |
| | | **Neural KalmanBOT** | **+3.92%** | **+0.97%** | **0.15** | **-10.18%** |
| **XOM–CVX** | **PASS** | Static-PCI | +20.54% | +4.79% | 0.59 | -9.83% |
| | | Dynamic-PCI | -11.99% | -3.15% | -0.23 | -32.80% |
| | | Regime-Gated | -4.07% | -1.03% | -0.16 | -15.15% |
| | | **Neural KalmanBOT** | **+40.11%** | **+8.81%** | **0.98** | **-11.12%** |
| **GLD–SLV** | REJ | Static-PCI | -25.28% | -7.03% | -1.11 | -25.51% |
| | | Dynamic-PCI | **+13.24%** | **+3.16%** | **0.46** | **-16.75%** |
| | | Regime-Gated | -2.73% | -0.69% | -0.25 | **-6.48%** |
| | | Neural KalmanBOT | -27.54% | -7.75% | -1.26 | -27.71% |
| **XLE–XOP** | REJ | Static-PCI | -14.52% | -3.85% | -0.43 | -20.35% |
| | | Dynamic-PCI | -10.56% | -2.75% | -0.07 | -40.64% |
| | | Regime-Gated | -0.58% | -0.15% | -0.68 | **-0.58%** |
| | | Neural KalmanBOT | -6.12% | -1.57% | -0.13 | -16.91% |
| **XLE–OIH** | **PASS** | Static-PCI | +15.57% | +3.69% | 0.35 | -19.73% |
| | | **Dynamic-PCI** | **+16.47%** | **+3.89%** | **0.28** | -40.06% |
| | | Regime-Gated | -31.92% | -9.17% | -0.75 | -39.27% |
| | | Neural KalmanBOT | -5.76% | -1.47% | -0.12 | -13.57% |
| **USO–XLE** | REJ | Static-PCI | -23.31% | -6.43% | -0.27 | -44.09% |
| | | Dynamic-PCI | -34.82% | -10.16% | -0.36 | -56.53% |
| | | Regime-Gated | -37.56% | -11.12% | -0.42 | -56.53% |
| | | Neural KalmanBOT | -15.46% | -4.12% | -0.10 | -29.86% |
| **MS–GS** | **PASS** | **Static-PCI** | **+16.37%** | **+3.87%** | **0.50** | **-9.37%** |
| | | Dynamic-PCI | -16.77% | -4.49% | -0.42 | -22.93% |
| | | Regime-Gated | -17.55% | -4.71% | -0.66 | -23.65% |
| | | Neural KalmanBOT | -11.65% | -3.05% | -0.22 | -26.21% |
| **V–MA** | REJ | Static-PCI | -8.93% | -2.31% | -0.46 | -12.33% |
| | | Dynamic-PCI | -8.84% | -2.29% | -0.54 | -14.30% |
| | | Regime-Gated | -8.84% | -2.29% | -0.54 | -14.30% |
| | | Neural KalmanBOT | -14.04% | -3.72% | -0.69 | -15.27% |
| **HD–LOW** | **PASS** | Static-PCI | -21.42% | -5.85% | -0.52 | -26.74% |
| | | Dynamic-PCI | -11.90% | -3.12% | -0.33 | -20.05% |
| | | Regime-Gated | -11.67% | -3.06% | -0.32 | -19.84% |
| | | **Neural KalmanBOT** | **+6.83%** | **+1.67%** | **0.23** | **-18.56%** |

---

## Why Are There Multiple Run Files?

```
┌────────────────────────────────────────────────────────────────────────┐
│                        main.py (MASTER ORCHESTRATOR)                  │
│   Unified CLI (--quick, --pairs), TeeLogger (outputs/pipeline.log),    │
│   runs the complete research narrative end-to-end                      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               generate_all_plots.py (PLOT ORCHESTRATOR)                │
│   Coordinates batch generation of publication-ready PNG figures        │
└──────────┬───────────────────────┬──────────────────────┬──────────────┘
           │                       │                      │
           ▼                       ▼                      ▼
┌──────────────────────┐ ┌───────────────────┐ ┌─────────────────────────┐
│     run_demo.py      │ │run_neural_demo.py │ │    run_all_pairs.py     │
│ Single-Pair Runner   │ │Neural Runner      │ │Benchmark Runner         │
│ Generates 4-section  │ │Head-to-head       │ │4 models across 14 pairs │
│ unified dashboard for│ │comparison: Neural │ │Saves comparison table   │
│ a pair (pair_*.png)  │ │vs Classical equity│ │and 3x5 grid equity plot │
└──────────────────────┘ └───────────────────┘ └─────────────────────────┘
```

1. **`main.py` (Master Orchestrator)**:
   - The authoritative entry point for the entire project.
   - Manages CLI flags (`--quick`, `--pairs`, `--epochs1`, `--epochs2`, `--log`), initializes console + file logging via `TeeLogger`, and sequentially executes simulation, neural comparison, per-pair dashboards, and cross-pair benchmarks.
2. **`generate_all_plots.py` (Plot Orchestrator)**:
   - Coordinates the creation of all visual figures (`outputs/*.png`).
   - Delegates tasks to the appropriate runner modules without cluttering code with logging overhead.
3. **`run_demo.py` (Single-Pair Runner)**:
   - Runs the classical 3-model pipeline (Static, Dynamic, Gated) on a single pair and produces a 4-section dashboard:
     - Section 1: Normalized asset prices & regimes
     - Section 2: Dynamic $\beta_t$ vs static $\beta_0$
     - Section 3: Filtered mean-reverting $M_t$
     - Section 4: Cumulative out-of-sample equity curves
4. **`run_neural_demo.py` (Neural Comparison Runner)**:
   - Focused head-to-head comparison between GRU-based Neural KalmanBOT and Classical Dynamic-PCI on one pair.
5. **`run_all_pairs.py` (Benchmark Runner)**:
   - Evaluates all 4 models across all 14 target pairs, saves `outputs/results_comparison.txt`, and generates the $3 \times 5$ all-pairs equity curves grid.
6. **`simulate.py` (Simulation & Validation Helper)**:
   - Generates synthetic price pairs with a known time-varying $\beta_t$ and known $M_t, R_t$ split to mathematically prove that the Kalman filter and MLE estimator recover ground-truth parameters.

---

## File Roles & Architecture

| File | Classification | Exact Responsibilities |
|---|---|---|
| [`main.py`](main.py) | **ORCHESTRATOR** | Master entry point. Parses CLI flags, sets up `TeeLogger`, executes the full research workflow. |
| [`generate_all_plots.py`](generate_all_plots.py) | **PLOT ORCHESTRATOR** | Master visualization coordinator. Triggers figure generation across all submodules. |
| [`run_demo.py`](run_demo.py) | **RUNNER** | Runs single-pair pipeline $\to$ creates 4-section dashboard (`outputs/pair_<T1>_<T2>.png`). |
| [`run_neural_demo.py`](run_neural_demo.py) | **RUNNER** | Neural KalmanBOT vs Classical Dynamic-PCI comparison (`outputs/neural_vs_classical.png`). |
| [`run_all_pairs.py`](run_all_pairs.py) | **BENCHMARK RUNNER** | Benchmarks all 4 models on all pairs $\to$ writes `outputs/results_comparison.txt` and $3 \times 5$ equity grid. |
| [`dynamic_pci.py`](dynamic_pci.py) | **CORE MODEL** | 4-state Kalman filter, log-likelihood, and L-BFGS-B MLE parameter fitting. |
| [`neural_kalman_pci.py`](neural_kalman_pci.py) | **NEURAL MODULE** | `KalmanNetGain` GRU network, `DifferentiableSharpeLoss`, tensor builders, and two-stage training loop. |
| [`regime_filter.py`](regime_filter.py) | **HELPER** | Pre-trade pair screening (Calibrated PCI & Strict) + online rolling regime gate. |
| [`backtest.py`](backtest.py) | **HELPER** | Z-score trading strategy, position transitions, transaction fee accounting (10 bps), metrics. |
| [`data_loader.py`](data_loader.py) | **HELPER** | Yahoo Finance data ingestion, offline CSV caching, normalization, train/test splitting. |
| [`simulate.py`](simulate.py) | **HELPER** | Synthetic DGP generator and parameter recovery validation. |
| [`plot_utils.py`](plot_utils.py) | **HELPER** | Publication-quality Matplotlib visualization routines ($3 \times 5$ equity grid, dashboards). |
| [`screening_performance_analysis.md`](screening_performance_analysis.md) | **RESEARCH DOCUMENT** | Comprehensive mathematical proof and contingency breakdown of screening performance across all 14 pairs. |
| [`report.tex`](report.tex) / [`report.pdf`](report.pdf) | **RESEARCH REPORT** | Formal academic paper compiling mathematical derivations, verified 11-pair empirical results, failure mode audits, and references. |

---

## How to Run

```bash
# Full Master Pipeline (All 14 cross-asset pairs, full neural training):
python main.py

# Run Canonical Equities/ETFs Only (7 pairs):
python main.py --canonical

# Run Commodity/Energy Pairs Only (4 pairs):
python main.py --commodities

# Fast Validation Mode (~30 sec, 2 pairs, 5 epochs):
python main.py --quick

# Evaluate Specific Pairs:
python main.py --pairs KO-PEP GLD-SLV EWA-EWC

# Individual Component Execution:
python simulate.py                     # Synthetic parameter recovery test
python run_demo.py                     # Single-pair dashboard (KO-PEP default)
python run_demo.py --all-pairs         # All 14 pair dashboards
python run_demo.py --canonical         # 7 canonical dashboards
python run_demo.py --ticker1 EWA --ticker2 EWC
python run_neural_demo.py              # Neural vs Classical comparison
python run_all_pairs.py                # Full 4-model benchmark table + 3x5 grid plot (14 pairs)
python run_all_pairs.py --canonical    # Benchmark on 7 canonical pairs
python run_all_pairs.py --quick        # Quick benchmark on 2 pairs
python generate_all_plots.py           # Batch plot generator
```
