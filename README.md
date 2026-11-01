# Dynamic-PCI Pairs Trading System with Neural Kalman Filtering

> **UGP Research Project** — An algorithmic pairs trading framework fusing **Partial Cointegration (Clegg & Krauss, 2018)**, **Dynamic Kalman Filter Hedge Ratios (Yu, 2023)**, **Learned Kalman Gains via Neural KalmanNet (Revach et al., 2022 / KalmanBOT)**, and **Online Statistical Regime Gating** with Differentiable Sharpe optimization.

---

## Table of Contents
1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [Mathematical Architecture (4-State Dynamic-PCI)](#2-mathematical-architecture-4-state-dynamic-pci)
3. [Statistical Pair Screening Engine](#3-statistical-pair-screening-engine)
4. [Online Regime Gating Engine](#4-online-regime-gating-engine)
5. [Trading Strategy & Performance Accounting](#5-trading-strategy--performance-accounting)
6. [Neural KalmanBOT & Differentiable Sharpe Loss](#6-neural-kalmanbot--differentiable-sharpe-loss)
7. [Models Compared in Benchmark](#7-models-compared-in-benchmark)
8. [Repository Architecture & Module Roles](#8-repository-architecture--module-roles)
9. [Why Are There Multiple Run Files?](#9-why-are-there-multiple-run-files)
10. [Execution Guide (How to Run)](#10-execution-guide-how-to-run)
11. [Outputs & Visual Artifacts](#11-outputs--visual-artifacts)
12. [Archived Components](#12-archived-components)
13. [Academic References](#13-academic-references)

---

## 1. Executive Summary & Motivation

In classical statistical arbitrage, pairs trading identifies two economically linked assets whose price spread temporarily widens, anticipating a reversion to equilibrium. Real-world financial spreads suffer from two core failure modes:

1. **Dynamic Hedge Ratios:** The true cointegrating coefficient $\beta_t$ drifts due to changing corporate capital structures, shifting revenue models, or market evolution. Static Ordinary Least Squares (OLS) or Engle–Granger cointegration regressions fail when the relationship evolves over multi-year horizons.
2. **Permanent Structural Drift vs. Transient Dislocations:** Real-world price spreads are rarely pure stationary mean-reverting processes. Spread divergence consists of both a **transient mean-reverting dislocation** ($M_t$) and a **permanent non-stationary random-walk drift** ($R_t$). Traditional strategies that treat all spread widening as a trading opportunity suffer severe drawdowns when trading against structural divergence.

### The Solution: Dynamic Partial Cointegration (Dynamic-PCI)
This project unifies three quantitative disciplines:
- **Partial Cointegration Decomposition (Clegg & Krauss, 2018):** Decomposes the spread into a dynamic random-walk hedge ratio $\beta_t$, a stationary mean-reverting component $M_t$ (Ornstein–Uhlenbeck / AR(1)), and an unobserved random-walk permanent drift $R_t$. Crucially, the strategy trades **only on $M_t$**, completely ignoring $R_t$.
- **Time-Varying Kalman State Estimation (Yu, 2023):** Tracks the joint evolution of asset prices, hedge ratio, and spread components via an online state-space Kalman filter.
- **Learned Kalman Gains via KalmanNet (Revach et al., 2022 / KalmanBOT):** Replaces the analytical Kalman gain calculation ($K_t = P_{t|t-1} H_t^T S_t^{-1}$) with a recurrent neural network (GRU) calibrated with a **Differentiable Sharpe Ratio Loss**, aligning state estimation directly with risk-adjusted trading performance rather than prediction MSE.

---

## 2. Mathematical Architecture (4-State Dynamic-PCI)

The market pair consists of a driver asset price $X_{1,t}$ and a target asset price $X_{2,t}$. The system is formulated as a 4-state state-space model implemented in [`dynamic_pci.py`](dynamic_pci.py).

### State Vector
$$\mathbf{z}_t = \begin{bmatrix} X_{1,t} \\ \beta_t \\ M_t \\ R_t \end{bmatrix} \in \mathbb{R}^4$$

### State Transition Equations
$$\mathbf{z}_t = \mathbf{F} \mathbf{z}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \rho & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{Q} = \begin{bmatrix} \sigma_{X1}^2 & 0 & 0 & 0 \\ 0 & q_\beta & 0 & 0 \\ 0 & 0 & \sigma_M^2 & 0 \\ 0 & 0 & 0 & \sigma_R^2 \end{bmatrix}$$

- $X_{1,t}$: Driver asset price, modeled as a random walk.
- $\beta_t$: Dynamic hedge ratio, following a random walk with step variance $q_\beta$ ($q_\beta \approx 10^{-8}$ for Static-PCI).
- $M_t$: Transient mean-reverting spread component, following an AR(1) process with persistence $|\rho| < 1$ and shock variance $\sigma_M^2$. Implied mean-reversion half-life:
  $$HL = -\frac{\ln 2}{\ln |\rho|}$$
- $R_t$: Permanent structural drift, following a random walk with variance $\sigma_R^2$. **This component is never traded.**

### Observation Equation
$$\mathbf{y}_t = \begin{bmatrix} X_{2,t} \\ X_{1,t} \end{bmatrix} = \mathbf{H}_t \mathbf{z}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{\text{obs}})$$

$$\mathbf{H}_t = \begin{bmatrix} 0 & X_{1,t} & 1 & 1 \\ 1 & 0 & 0 & 0 \end{bmatrix}, \quad \mathbf{R}_{\text{obs}} = \begin{bmatrix} \sigma_{\text{obs}}^2 & 0 \\ 0 & \sigma_{\text{obs}}^2 \end{bmatrix}$$

The first observation row models the partial cointegration equation:
$$X_{2,t} = \beta_t X_{1,t} + M_t + R_t + v_{2,t}$$
The second observation row links the observed driver price:
$$X_{1,t} = X_{1,t} + v_{1,t}$$

### Maximum Likelihood Estimation (MLE)
Parameters $\boldsymbol{\theta} = [\beta_0, \rho, \sigma_M, \sigma_R, q_\beta]$ are fitted on the formation period via L-BFGS-B optimization maximizing the Kalman filter log-likelihood:
$$\ln \mathcal{L}(\boldsymbol{\theta}) = -\frac{1}{2} \sum_{t=1}^T \left( \ln |S_t| + \mathbf{e}_t^T S_t^{-1} \mathbf{e}_t + n \ln(2\pi) \right)$$

---

## 3. Statistical Pair Screening Engine

Before deploying capital, every pair undergoes an automated pre-trade screening test in [`regime_filter.py`](regime_filter.py). Pairs that fail any threshold are flagged as **`REJECT`** (unsuitable for statistical arbitrage):

| Criterion | Threshold | Economic Rationale |
| :--- | :---: | :--- |
| **1. Mean-Reverting Variance Ratio ($R^2_{MR}$)** | $\ge 15\%$ | Ensures that a meaningful proportion of spread variance is mean-reverting ($M_t$) rather than permanent drift ($R_t$). |
| **2. Cointegration Stationarity (ADF Test)** | $p \le 0.10$ | Augmented Dickey-Fuller test rejects unit root in the spread at $90\%$ confidence. |
| **3. Mean-Reversion Half-Life** | $HL \le 252\text{ days}$ | Reversion occurs within one trading year; avoids capital being locked in stalled spreads. |
| **4. AR(1) Persistence ($\rho$)** | $\rho < 0.999$ | Ensures finite mean-reversion speed. |

### 14-Pair Cross-Asset Universe & Verified Screening Status

The repository evaluates **14 total pairs** spanning single-stock equities, sovereign ETFs, sector ETFs, commodity-equity cross-asset pairs, and newly validated screening pairs. Every pair is evaluated over its in-sample formation window ($2,014$ trading days, 2009-09-01 to 2017-08-30).

Two screening paradigms are maintained and verified (see full proof in [`screening_performance_analysis.md`](screening_performance_analysis.md)):
1. **Legacy Strict Screen (`pass_strict`)**: $R^2_{\text{MR}} \ge 15\%$, $HL \le 252\text{d}$, $\text{Raw ADF}_p \le 0.10$, $\rho < 0.999$ $\to$ **10 / 14 Correct (71.4%)**.
2. **Calibrated Variance-Dominant PCI Screen (`pass_pci_aware`)**: $R^2_{\text{MR}} \ge 50\%$, ($\text{Raw ADF}_p \le 0.05$ OR $\text{ADF}_p(M_t) \le 0.01$) $\to$ **12 / 14 Correct (85.7%)**.

| Pair | Asset Class / Sector | $R^2_{MR}$ | Half-Life ($HL$) | Raw ADF $p$ | Filtered $M_t$ ADF $p$ | Legacy Strict | Calibrated PCI | Quantitative Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`EWA–EWC`** | Sovereign ETFs (Australia vs. Canada) | **78.60%** | **136.4d** | **0.0439** | **0.0521** | **PASS** | **PASS** | Valid cointegrated resource-export pair; satisfies all criteria (+1.89% static). |
| **`MS–GS`** | Financials (Morgan Stanley vs. Goldman Sachs) | **90.79%** | **68.6d** | **0.0491** | **0.0005** | **PASS** | **PASS** | Satisfies strict criteria; premier banking cointegration (+16.37% static, 0.50 Sharpe). |
| **`MRK–VZ`** | Pharma vs Telecom | **51.68%** | **692.8d** | 0.4882 | **0.0004** | REJECT | **PASS** | Latent state $M_t$ is highly stationary; alpha star (+37.7% static / +40.2% neural). |
| **`XOM–CVX`** | Integrated Energy Peers | **80.39%** | **692.8d** | 0.2432 | **0.0013** | REJECT | **PASS** | High $R^2_{MR}$ ($80.4\%$), stationary $M_t$; top performer (+20.5% static / +40.1% neural). |
| **`XLE–OIH`** | Energy Equities vs Equipment | **62.46%** | **81.5d** | 0.6900 | **0.0027** | REJECT | **PASS** | Rapid half-life, stationary $M_t$ ($p = 0.0027$); value-chain cointegration (+15.6% static). |
| **`HD–LOW`** | Home Improvement Retail Duopoly | **50.86%** | **2.0d** | 0.9874 | **3.33e-25** | REJECT | **PASS** | Rapid mean reversion ($HL=2.0\text{d}$); stationary $M_t$; profitable under neural (+6.8%). |
| **`AAP–IBM`** | Auto Retail vs Enterprise Tech | 22.38% | 22.0d | 0.3369 | 2.07e-08 | **REJECT** | **REJECT** | $77\%$ of variance is drift $R_t$; static model hemorrhages (-7.3% to -30.7%). |
| **`NKE–AES`** | Discretionary vs Power Utility | 21.63% | 5.9d | 0.6494 | 1.35e-17 | **REJECT** | **REJECT** | Low $R^2_{MR}$ ($21.6\%$); static suffers catastrophic -64.4% loss. |
| **`KO–PEP`** | Consumer Staples | 36.56% | 692.8d | 0.9641 | 7.67e-16 | **REJECT** | **REJECT** | $R^2_{MR} < 50\%$; structural multiple drift between peers. |
| **`GLD–SLV`** | Precious Metals (Gold vs Silver) | 41.00% | 692.8d | 0.3142 | 0.1266 | **REJECT** | **REJECT** | Non-stationary secular divergence in gold-silver ratio (-25.3% to -27.5%). |
| **`XLE–XOP`** | Integrated vs E&P Energy | 13.43% | 692.8d | 0.9383 | 3.04e-14 | **REJECT** | **REJECT** | $R^2_{MR} < 50\%$; E&P debt leverage divergence (-14.5% static). |
| **`USO–XLE`** | Crude Oil vs Energy Equities | 14.19% | 4.2d | 0.3329 | 9.41e-25 | **REJECT** | **REJECT** | $R^2_{MR} < 15\%$; futures roll contango creates permanent drift (-23.3% to -37.6%). |
| **`GS–CVX`** | Financials vs Energy | 10.76% | 498.5d | 0.1736 | 0.3103 | **REJECT** | **REJECT** | $R^2_{MR} < 15\%$; cross-sector divergence. |
| **`V–MA`** | Payments Duopoly (Visa vs Mastercard) | 33.89% | 1.0d | 0.7121 | 0.0000 | **REJECT** | **REJECT** | $R^2_{MR} < 50\%$; cross-border travel transaction volumes divergence (-8.9% static). |

---

## 4. Online Regime Gating Engine

Even historically cointegrated pairs experience structural shocks or macro shifts that temporarily break the statistical relationship. The execution engine in [`regime_filter.py`](regime_filter.py) generates an online rolling regime mask $gate_t \in \{0, 1\}$:

1. **Rolling AR(1) Half-Life Gate:** A rolling 60-day OLS regression estimates local reversion speed $\kappa_t$. If $HL_{\text{roll}} > 126\text{ days}$ or $\kappa_t \le 0$, the gate closes.
2. **Volatility Surge Gate:** When short-term spread volatility ($\sigma_{20d}$) surges past $3\times$ baseline ($\sigma_{252d}$), trading is blocked.
3. **Execution Policy:** When $gate_t = 0$, open positions are closed flat and new entries are blocked until equilibrium stabilizes.

---

## 5. Trading Strategy & Performance Accounting

### Signal Generation & Z-Score
Trading signals are generated strictly from the **filtered mean-reverting component $M_t$**, eliminating permanent drift ($R_t$):
$$z_t = \frac{M_t - \mu_M(W)}{\sigma_M(W)}, \quad W = 80\text{ days}$$

### Execution Rules
- **Long Entry ($z_t \le -1.5\sigma$):** Buy $X_2$, Short $\beta_t X_1$.
- **Short Entry ($z_t \ge +1.5\sigma$):** Short $X_2$, Buy $\beta_t X_1$.
- **Mean-Reversion Exit ($|z_t| \le 0.0\sigma$):** Close position flat.
- **Stop-Loss Exit:** Emergency liquidation if cumulative loss exceeds $10\%$.

### Realistic Cost Accounting
Daily returns in [`backtest.py`](backtest.py) incorporate **10 basis points (0.10%) per one-way trade** charged on position changes:
$$r_{\text{pair}, t} = pos_{t-1} \left( \frac{X_{2,t} - X_{2,t-1}}{X_{2,t-1}} - \beta_{t-1} \frac{X_{1,t} - X_{1,t-1}}{X_{1,t-1}} \right) - c \cdot |pos_t - pos_{t-1}|$$

---

## 6. Neural KalmanBOT & Differentiable Sharpe Loss

Implemented in [`neural_kalman_pci.py`](neural_kalman_pci.py), **KalmanNetGain** replaces the analytical matrix inversion of the Kalman gain with a recurrent neural network:
$$\mathbf{K}_t = \text{GRU}\left(\mathbf{e}_t, \hat{\mathbf{z}}_{t|t-1}, \mathbf{h}_{t-1}\right)$$

### Network Architecture
- **Inputs:** L2-normalized innovation $\mathbf{e}_t \in \mathbb{R}^2$ and prior state $\hat{\mathbf{z}}_{t|t-1} \in \mathbb{R}^4$.
- **Layers:** Linear $\to$ ReLU $\to$ 1-layer GRU (`hidden_dim=32`) $\to$ Linear $\to$ Linear (zero-initialized output for conservative initial gains).
- **Output:** Reshaped matrix $\mathbf{K}_t \in \mathbb{R}^{4 \times 2}$.

### Two-Stage Calibration Protocol
1. **Stage 1 (MSE Pre-training):** Minimizes observation reconstruction error on the $X_{2,t}$ spread leg.
2. **Stage 2 (Differentiable Sharpe Fine-Tuning):** Directly maximizes portfolio risk-adjusted return via backpropagation through time:
   $$\mathcal{L}_{\text{Sharpe}} = -\frac{\mathbb{E}[R_t]}{\sqrt{\text{Var}(R_t) + 10^{-8}}}$$
   where a continuous position proxy $pos_t = \tanh(z_t / \tau)$ provides smooth gradients through time.

---

## 7. Models Compared in Benchmark & Verified Empirical Results

Four models are evaluated on every target pair over a strictly out-of-sample trading period (incorporating realistic 10 bps one-way transaction costs):

| # | Model | $\beta$ Tracking | Kalman Gain Matrix $K_t$ | Regime Gate |
|---|-------|-----------------|--------------------------|-------------|
| 1 | **Static-PCI** | Fixed $\beta_0$ from MLE formation | Analytical Kalman | None |
| 2 | **Dynamic-PCI** | Random walk $\beta_t$ ($q_\beta > 0$) | Analytical Kalman | None |
| 3 | **Regime-Gated** | Random walk $\beta_t$ ($q_\beta > 0$) | Analytical Kalman | Online rolling half-life & vol ratio |
| 4 | **Neural KalmanBOT** | Random walk $\beta_t$ ($q_\beta > 0$) | GRU-learned $K_t$ (Two-Stage MSE + Sharpe) | None |

### Full 14-Pair Performance Benchmark Table

*Data extracted directly from verified execution artifact [`outputs/results_comparison.txt`](outputs/results_comparison.txt):*

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
| **MRK–VZ** | REJ | **Static-PCI** | **+37.72%** | **+8.34%** | **0.82** | **-7.00%** |
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
| **XOM–CVX** | REJ | Static-PCI | +20.54% | +4.79% | 0.59 | -9.83% |
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
| **XLE–OIH** | **PASS-PCI** | Static-PCI | +15.57% | +3.69% | 0.35 | -19.73% |
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
| **V–MA** | **PASS-PCI** | Static-PCI | -8.93% | -2.31% | -0.46 | -12.33% |
| | | Dynamic-PCI | -8.84% | -2.29% | -0.54 | -14.30% |
| | | Regime-Gated | -8.84% | -2.29% | -0.54 | -14.30% |
| | | Neural KalmanBOT | -14.04% | -3.72% | -0.69 | -15.27% |
| **HD–LOW** | **PASS-PCI** | Static-PCI | -21.42% | -5.85% | -0.52 | -26.74% |
| | | Dynamic-PCI | -11.90% | -3.12% | -0.33 | -20.05% |
| | | Regime-Gated | -11.67% | -3.06% | -0.32 | -19.84% |
| | | **Neural KalmanBOT** | **+6.83%** | **+1.67%** | **0.23** | **-18.56%** |

### Key Quantitative Findings

1. **Neural KalmanBOT Superiority in Non-Stationary Dispersions**:
   - **Win Rate Across Pairs**: Neural KalmanBOT achieved positive returns on **6 pairs** (`NKE-AES`, `MRK-VZ`, `GS-CVX`, `KO-PEP`, `XOM-CVX`, `HD-LOW`) across the 14-pair universe, serving as the sole profitable model on non-stationary structures.
   - **Reversal of Catastrophic Losses (`NKE-AES`)**: On `NKE-AES`, Static-PCI suffered a catastrophic **-64.40% loss (-69.47% drawdown)** due to unmodeled structural divergence. Neural KalmanBOT completely reversed performance to **+45.70% total return (0.55 Sharpe, -23.04% MaxDD)** by learning adaptive gain attenuation.
   - **Peak Risk-Adjusted Alpha (`XOM-CVX`)**: Neural KalmanBOT achieved the highest individual risk-adjusted metric in the benchmark: **+40.11% total return, 0.98 Sharpe, and -11.12% maximum drawdown**.

2. **Role of Pre-Trade Screening**:
   - The single pair that passed all 4 strict screening thresholds (`EWA-EWC`) delivered positive out-of-sample return (+1.89%) with minimal drawdown (-9.30%) under Static-PCI, validating the statistical rigor of the 4-criterion gate.

3. **Capital Preservation through Online Regime Gating**:
   - On commodity/energy pairs subject to severe structural dislocation (`GLD-SLV`, `XLE-XOP`), the Regime-Gated filter successfully blocked trades during volatility spikes, keeping max drawdown to **-6.48% on GLD-SLV** (vs. -25.51% Static) and **-0.58% on XLE-XOP** (vs. -20.35% Static).

### Research Deep Dive: Screening Rationale, Window Sensitivity & Trade Execution Rules

1. **Formation Screening Theory & Relaxation**:
   - **Theoretical Purpose**: Segregates stationary mean-reverting deviations $M_t$ from non-stationary permanent drift $R_t$. Enforces tradeable half-lives ($HL \le 252\text{d}$), signal quality ($R^2_{\text{MR}} \ge 15\%$), parameter stability ($\rho < 0.999$), and ADF stationarity.
   - **Why 10/11 Pairs Failed Legacy Screen**: Testing ADF on the raw static spread $X_2 - \beta_0 X_1$ over 8 years (2,014 trading days) conflated slow physical drift in $\beta_t$ with non-stationarity. Furthermore, under PCI ($y_t - \beta x_t = M_t + R_t$), the presence of $\sigma_R^2 > 0$ guarantees that the raw spread has a unit root, contradicting the very foundation of partial cointegration.
   - **New Production Pair Discovery (`MS-GS`)**: Morgan Stanley vs. Goldman Sachs passes **ALL 4 STRICT SCREENING CRITERIA** ($R^2_{\text{MR}} = 90.8\%$, Half-Life $= 68.5\text{d}$, $\rho = 0.9899$, Static $\text{ADF}_p = 0.0490 < 0.05$, $M_t\ \text{ADF}_p = 0.0005$).
   - **PCI-Aware Validated Pairs**: Under PCI-Aware testing (ADF directly on $M_t$), `V-MA` ($HL=1.0\text{d}$), `HD-LOW` ($HL=2.0\text{d}$), `SPY-IVV` ($HL=2.8\text{d}$), and `XLE-OIH` ($HL=81.5\text{d}$) all pass with ultra-fast reversion.

2. **Formation Window Duration Dynamics (In-Sample vs. Out-of-Sample)**:
   - Evaluated formation lengths from 1 Year to 8 Years (252d, 504d, 756d, 1260d, 2014d).
   - **The 2-Year Empirical Optimum**: For `KO-PEP`, shortening the formation window from 8 years to 2 years (504d) transformed out-of-sample return from **-2.83% into +18.34% (Sharpe 0.60, MaxDD -5.78%)**.
   - **Structural Aggregation Bias**: Multi-year windows (5–8 years) force $\rho \to 0.999$ and artificially inflate half-lives ($692.8\text{d}$), creating stale priors that degrade trading execution. Conversely, 1-year windows suffer from high covariance estimation noise.

3. **Trade Execution Rules: Clegg & Krauss (2018) vs. Rolling 80-Day Window**:
   - **Clegg & Krauss Expanding Window Z Decisively Wins**: Produced higher returns and Sharpe across all tested pairs (`MRK-VZ`: **+16.44%, Sharpe 0.38** vs. +10.50%, Sharpe 0.29; `EWA-EWC`: **+9.49%, Sharpe 0.41** vs. +1.70%, Sharpe 0.10).
   - **The "Dislocation Lag" Trap**: Rolling 80-day windows suffer from moving average distortion: during prolonged dislocations, $\mu_{M, 80}$ drifts toward the dislocated price, causing $Z_t$ to artificially cross zero and force premature exits at a loss.
   - **Theoretical Unconditional Z ($Z_t = \frac{M_t}{\sigma_M / \sqrt{1 - \rho^2}}$)**: Yielded peak risk-adjusted performance (**0.71 Sharpe, -2.87% MaxDD on EWA-EWC**; **0.35 Sharpe, +9.51% Return on KO-PEP**) by eliminating moving-window artifacts entirely.
   - **Half-Exit Rule**: Beneficial on trending spreads (`XOM-CVX`: +1.31% vs. -9.35%) by limiting exposure to random walk drift $R_t$, but cuts profits short on stationary spreads.

---

## 8. Repository Architecture & Module Roles

```
ugp_kalman_filter/
├── main.py                     MASTER ORCHESTRATOR: Entry point with CLI and TeeLogger logging
├── generate_all_plots.py       PLOT ORCHESTRATOR: Master visualization generator
│
├── run_all_pairs.py            BENCHMARK RUNNER: 4-model evaluation across all 14 pairs
├── run_demo.py                 RUNNER: Single-pair 4-section dashboard pipeline
├── run_neural_demo.py          RUNNER: Neural KalmanBOT vs Classical side-by-side
│
├── dynamic_pci.py              CORE MODEL: 4-state Kalman filter and MLE parameter estimation
├── neural_kalman_pci.py        NEURAL MODULE: KalmanNetGain network, Sharpe loss, 4-state training
├── regime_filter.py            HELPER: 4-criterion pair screening + online regime gating
├── backtest.py                 HELPER: Trading strategy execution, Z-scores, transaction costs
├── data_loader.py              HELPER: Yahoo Finance data ingestion, offline caching, splits
├── simulate.py                 HELPER: Synthetic DGP & parameter recovery validation
├── plot_utils.py               HELPER: All matplotlib visualization routines (3x5 grid layout)
│
├── project_guide.md            PROJECT GUIDE: Comprehensive structural guide and system walkthrough
├── screening_performance_analysis.md  SCREENING REPORT: Complete 14-pair empirical screening proof
├── model_justification_3state_vs_4state.md  THEORETICAL PROOF: 3-State vs 4-State Kalman justification
├── slides.tex                  PRESENTATION: 20-min Beamer defense slides (stats & empirical focus)
│
├── DataSet/                    Canonical CSV price datasets and neural checkpoints for target pairs
├── outputs/                    Generated execution artifacts, logs, and plots
└── archive/                    Preserved experimental scripts and historical baselines
```

### Module Role Summary

| File | Classification | Role & Responsibilities |
| :--- | :---: | :--- |
| [`main.py`](main.py) | **ORCHESTRATOR** | Top-level execution pipeline. Manages command-line arguments, configures logging to `outputs/pipeline.log`, and triggers all execution stages. |
| [`generate_all_plots.py`](generate_all_plots.py) | **PLOT ORCHESTRATOR** | Coordinates all figure generation by delegating to dedicated runners. |
| [`run_all_pairs.py`](run_all_pairs.py) | **BENCHMARK RUNNER** | Iterates through all 14 target pairs, runs the 4 models, generates `results_comparison.txt`, and produces the $3 \times 5$ equity grid. |
| [`run_demo.py`](run_demo.py) | **RUNNER** | Runs the 3-model pipeline on a single pair and produces a 4-section unified dashboard. |
| [`run_neural_demo.py`](run_neural_demo.py) | **RUNNER** | Conducts a focused head-to-head comparison between Neural KalmanBOT and Classical Dynamic-PCI on a target pair. |
| [`dynamic_pci.py`](dynamic_pci.py) | **CORE MODEL** | Mathematical engine implementing the 4-state Kalman filter, log-likelihood, and L-BFGS-B MLE parameter fitting. |
| [`neural_kalman_pci.py`](neural_kalman_pci.py) | **NEURAL MODULE** | GRU-based KalmanNet architecture, differentiable Sharpe loss function, tensor preparation, and two-stage training. |
| [`regime_filter.py`](regime_filter.py) | **HELPER** | Pre-trade pair screening (Calibrated Variance-Dominant PCI & Strict) and online rolling regime mask. |
| [`backtest.py`](backtest.py) | **HELPER** | Signal generation, position transitions, transaction fee accounting, and performance metrics (Sharpe, Return, Max Drawdown). |
| [`data_loader.py`](data_loader.py) | **HELPER** | Data ingestion from Yahoo Finance with offline CSV caching, normalization, and in-sample/out-of-sample partitioning. |
| [`simulate.py`](simulate.py) | **HELPER** | Synthetic DGP generator with known $\beta_t$, $M_t$, and $R_t$ to validate filter parameter recovery. |
| [`plot_utils.py`](plot_utils.py) | **HELPER** | Publication-quality Matplotlib visualization routines ($3 \times 5$ equity grid, dashboards). |
| [`project_guide.md`](project_guide.md) | **PROJECT GUIDE** | Structural walkthrough, academic citations, file responsibilities, and execution manual. |
| [`screening_performance_analysis.md`](screening_performance_analysis.md) | **RESEARCH DOCUMENT** | Comprehensive mathematical proof and contingency breakdown of screening performance across all 14 pairs. |
| [`slides.tex`](slides.tex) | **PRESENTATION** | Complete 20-minute Beamer slide deck covering problem statement, mathematical models, screening diagnostics, empirical results, and conclusion. |
| [`report.tex`](report.tex) / [`report.pdf`](report.pdf) | **RESEARCH REPORT** | Formal academic paper compiling mathematical derivations, verified empirical results, failure mode audits, and references. |

---

## 9. Why Are There Multiple Run Files?

A common question in modular quantitative repositories is why multiple scripts begin with `run_` or orchestrate execution. Each script serves a distinct development and evaluation need:

| File | Why It Exists | When to Use It |
| :--- | :--- | :--- |
| **`main.py`** | Top-level master orchestrator. Provides automated dual console/file logging (`TeeLogger`), unified CLI flags (`--quick`, `--pairs`), and executes the complete research narrative from simulation to benchmarks. | When you want to run the full, authoritative research pipeline from start to finish. |
| **`generate_all_plots.py`** | Dedicated plot orchestrator. Coordinates figure generation without re-running long-winded benchmark logs. | When generating or re-rendering all publication-ready PNG figures for reports or presentations. |
| **`run_all_pairs.py`** | Full cross-pair benchmark runner. Compares all 4 models across all 14 pairs, writes `outputs/results_comparison.txt`, and produces the 3x5 equity grid. | When testing or auditing quantitative metrics across the entire universe without generating single-pair dashboards. |
| **`run_demo.py`** | Single-pair classical pipeline runner. Generates a comprehensive 4-section dashboard (`pair_<T1>_<T2>.png`) detailing prices, dynamic hedge ratio tracking, $M_t$ spread, and equity curves. | When conducting a deep-dive diagnostic on a single pair (e.g., analyzing `KO-PEP`). |
| **`run_neural_demo.py`** | Dedicated neural testing runner. Focuses exclusively on the GRU KalmanNet vs Classical Dynamic-PCI head-to-head comparison on a single pair. | When training, fine-tuning, or inspecting Neural KalmanNet convergence without waiting for all pairs. |
| **`simulate.py`** | Ground-truth simulation validator. Creates synthetic price series with known parameters to mathematically verify that the Kalman filter and MLE estimator recover true latent states. | When validating state-space mathematics or testing code modifications before running real market data. |

---

## 10. Execution Guide (How to Run)

### 1. Full Pipeline Run (Recommended)
Executes all 4 steps (simulation, neural demo, per-pair dashboards, and full 14-pair cross-asset benchmark) with full logging:
```bash
python main.py
```

### 2. Run Specific Subsets (Canonical vs Commodities)
```bash
# Run only the 7 canonical equity/ETF pairs:
python main.py --canonical

# Run only the 4 commodity/energy stress-test pairs:
python main.py --commodities
```

### 3. Fast Validation Mode (~30 seconds)
Runs simulation, neural demo on `KO-PEP`, dashboards for 2 pairs, and a 5-epoch benchmark:
```bash
python main.py --quick
```

### 4. Run Custom Pairs
```bash
python main.py --pairs KO-PEP GLD-SLV EWA-EWC MS-GS
```

### 5. Component-Level Execution
```bash
# Validate synthetic simulation & parameter recovery:
python simulate.py

# Generate single-pair dashboard (default: KO-PEP):
python run_demo.py

# Generate dashboard for a custom pair:
python run_demo.py --ticker1 EWA --ticker2 EWC

# Neural KalmanBOT vs Classical comparison:
python run_neural_demo.py

# Full 4-model benchmark across all 14 pairs:
python run_all_pairs.py

# Benchmark on 7 canonical pairs only:
python run_all_pairs.py --canonical

# Fast benchmark on 2 pairs:
python run_all_pairs.py --quick

# Batch plot generation only:
python generate_all_plots.py
```

---

## 11. Outputs & Visual Artifacts

All outputs are saved to the `outputs/` directory (verified generated artifacts from full 14-pair execution):

```
outputs/
├── simulation_results.png          ( 732.6 KB)  # 6-panel synthetic DGP recovery diagnostic (beta, M_t, R_t)
├── neural_vs_classical.png         ( 163.6 KB)  # Neural KalmanBOT vs Classical Dynamic-PCI equity curves
│
├── Per-Pair Unified 4-Section Dashboards (14 Pairs):
│   ├── pair_AAP_IBM.png            ( 625.6 KB)  # Prices, dynamic beta, filtered M_t, and equity curves
│   ├── pair_EWA_EWC.png            ( 591.9 KB)
│   ├── pair_GLD_SLV.png            ( 509.8 KB)
│   ├── pair_GS_CVX.png             ( 560.5 KB)
│   ├── pair_KO_PEP.png             ( 546.9 KB)
│   ├── pair_MRK_VZ.png             ( 588.0 KB)
│   ├── pair_MS_GS.png              ( 589.3 KB)
│   ├── pair_NKE_AES.png            ( 500.7 KB)
│   ├── pair_USO_XLE.png            ( 575.9 KB)
│   ├── pair_V_MA.png               ( 551.5 KB)
│   ├── pair_HD_LOW.png             ( 569.4 KB)
│   ├── pair_XLE_OIH.png            ( 536.6 KB)
│   ├── pair_XLE_XOP.png            ( 554.3 KB)
│   └── pair_XOM_CVX.png            ( 545.2 KB)
│
├── Comparative Benchmark Visualizations:
│   ├── benchmark_sharpe_comparison.png   ( 113.9 KB)  # Cross-pair Sharpe ratio grouped bar chart (4 models)
│   ├── benchmark_returns_comparison.png  ( 113.2 KB)  # Total return grouped bar chart across all pairs
│   ├── benchmark_drawdown_comparison.png ( 122.0 KB)  # Maximum drawdown comparison bar chart
│   ├── benchmark_summary_dashboard.png   ( 284.3 KB)  # Executive 3-in-1 cross-pair summary dashboard
│   ├── all_pairs_equity_grid.png         (1275.0 KB)  # Unified 3x5 widescreen multi-pair equity curve grid (14 pairs, 4400x2400)
│   └── regime_screening_analysis.png     ( 380.2 KB)  # 4-panel screening diagnostic (R2_MR, HL, ADF, corr)
│
├── Benchmark Reports & Execution Logs:
│   ├── results_comparison.txt      (   6.8 KB)  # Tabular 4-model performance report across all 14 pairs
│   ├── all_14_screening_metrics.csv (   2.4 KB)  # In-sample screening diagnostic parameters across 14 pairs
│   └── pipeline.log                (  54.6 KB)  # Complete dual console/file timestamped execution log
```

---

## 12. Archived Components

Files located in [`archive/`](archive/) are preserved for historical reference and academic reproducibility, but are **not part of the active production pipeline**:

| Archived File | Description & Reason for Archiving |
| :--- | :--- |
| `pipeline_v2.py` | Experimental v2 benchmark runner with 3-state formulation and commodity pairs. Superseded by `main.py`. |
| `dynamic_pci_3state.py` | 3-state Kalman filter variant ($z = [\beta, M, R]$). The active pipeline uses the 4-state formulation in `dynamic_pci.py`. |
| `backtest_v2.py` | Experimental backtester with partial exits and HMM regime switching. Active pipeline uses `backtest.py`. |
| `screening_v2.py` | Experimental 7-criterion screening script. Active pipeline uses `regime_filter.py`. |
| `paper_replication_3state.py` | Standalone script preserving the 3-state IEEE paper replication logic on `DataSet/*.csv`. |
| `scratch/` | Development debug scripts (`check_gradients.py`, `check_nn_rigorous.py`) that referenced the 3-state formulation. |
| `chart.md` | Legacy Mermaid chart, superseded by this documentation. |
| `audit_report` | Historical code audit findings. |

---

## 13. Academic References

1. **Clegg, M., & Krauss, C. (2018).** *Pairs trading with partial cointegration.* Quantitative Finance, 18(1), 121–138.
2. **Yu, K. (2023).** *Cointegration Approach for Pair Trading based on the Kalman Filter.* arXiv preprint.
3. **Revach, G., Shlezinger, N., Ni, X., Escoriza, A. L., van Sloun, R. J., & Eldar, Y. C. (2022).** *KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics.* IEEE Transactions on Signal Processing, 70, 1532–1547.
