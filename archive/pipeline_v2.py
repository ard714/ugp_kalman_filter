"""
pipeline_v2.py — Master Benchmark Runner (v2)
==============================================

Models compared per pair:
  A. Static-PCI          : 3-state Kalman, q_beta frozen (fixed hedge ratio)
  B. Dynamic-PCI (KF)    : 3-state Kalman, beta as random walk (no neural net)
  C. Gated (Vol+AR1)     : Dynamic-PCI spread gated by volatility + AR(1) regime
  D. Neural-Gated        : GRU-based Kalman gain, gated by combined mask

All four models use the SAME:
  • data split     : in-sample = 2014 days, out-of-sample = 1007 days
  • return series  : pair_return from backtest_v2 (fully reconciled)
  • entry logic    : tested at tau_open ∈ {1.0, 1.5, 1.75} σ
  • screening      : strengthened 7-criterion screen (screening_v2)

Usage:
  python pipeline_v2.py [--pairs all|equity|commodity] [--out results_v2.txt]
  python pipeline_v2.py --fast           # single tau=1.5 run (fastest)
  python pipeline_v2.py --tau tau1.5     # specific tau only

Speed note: MLE fits are cached per pair; neural training uses 10+10 epochs.
"""

import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import TARGET_PAIRS, COMMODITY_ENERGY_PAIRS, fetch_pair_data
from dynamic_pci_3state import fit_pci3, PCI3KalmanFilter
from backtest_v2 import (
    backtest_v2, performance_summary_v2,
    compute_vol_gate, compute_regime_mask,
    compute_hmm_regime_filter,
)
from screening_v2 import screen_pair, describe_screening
from neural_kalman_pci import KalmanNetGain, DifferentiableSharpeLoss

import yfinance as yf

OUTPUTS_DIR = os.path.join("outputs", "v2")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Safe unbuffered stdout
import functools
_print = print
def print(*args, **kwargs):
    try:
        _print(*args, **kwargs)
    except (UnicodeEncodeError, UnicodeError):
        safe_args = [str(a).encode("ascii", errors="replace").decode("ascii") for a in args]
        _print(*safe_args, **kwargs)
    try:
        sys.stdout.flush()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration (Canonical 2009-2021 Window across all pairs)
# ---------------------------------------------------------------------------
IN_SAMPLE_DAYS  = 2014   # 2009-09-01 to 2017-08-30 (~8 years formation)
OUT_SAMPLE_DAYS = 1007   # 2017-08-31 to 2021-08-31 (~4 years trading)

ALL_PAIRS = TARGET_PAIRS + COMMODITY_ENERGY_PAIRS

TAU_CONFIGS = {
    "tau1.0": dict(tau_open=1.0,  tau_close=0.0, tau_partial=None),
    "tau1.5": dict(tau_open=1.5,  tau_close=0.0, tau_partial=0.5),
    "tau1.75":dict(tau_open=1.75, tau_close=0.0, tau_partial=0.5),
}
DEFAULT_TAU = "tau1.5"   # primary comparison threshold


# ---------------------------------------------------------------------------
# Data fetching (by day-count, not calendar dates)
# ---------------------------------------------------------------------------

def fetch_pair_by_count(t1: str, t2: str, total_days: int = 3500) -> pd.DataFrame:
    """
    Download adjusted-close prices for (t1, t2), return last `total_days` rows.
    Uses group_by='ticker' workaround for yfinance multi-ticker MultiIndex bug.
    Falls back to downloading tickers one by one if batch fails.
    """
    def _download_one(ticker):
        raw = yf.download(ticker, period="max", auto_adjust=False, progress=False)
        if raw.empty:
            return pd.Series(dtype=float, name=ticker)
        if "Adj Close" in raw.columns:
            s = raw["Adj Close"]
        elif "Close" in raw.columns:
            s = raw["Close"]
        else:
            s = raw.iloc[:, 0]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.name = ticker
        return s

    try:
        s1 = _download_one(t1)
        s2 = _download_one(t2)
        if s1.empty or s2.empty:
            return pd.DataFrame()
        df = pd.concat([s1, s2], axis=1)
        df.columns = [t1, t2]
        df = df.ffill().bfill().dropna()
        if len(df) > total_days:
            df = df.iloc[-total_days:]
        return df
    except Exception as e:
        print(f"  [fetch error] {t1}-{t2}: {e}")
        return pd.DataFrame()



def split_pair_data(df: pd.DataFrame, t1: str, t2: str, in_sample_days: int, out_sample_days: int):
    """
    Split into formation (in-sample) and trading (out-of-sample), normalised to base 1.0.
    """
    total = in_sample_days + out_sample_days
    if len(df) < total:
        raise ValueError(f"Not enough data: {len(df)} rows, need {total}")

    df = df.iloc[-total:]
    df_form  = df.iloc[:in_sample_days]
    df_trade = df.iloc[in_sample_days: in_sample_days + out_sample_days]

    x1_form  = df_form[t1].values.astype(float)
    x2_form  = df_form[t2].values.astype(float)
    x1_trade = df_trade[t1].values.astype(float)
    x2_trade = df_trade[t2].values.astype(float)
    dates_trade = df_trade.index

    # Normalise to base 1.0 (formation period and test period separately)
    x1_form  = x1_form  / x1_form[0]
    x2_form  = x2_form  / x2_form[0]
    x1_trade = x1_trade / x1_trade[0]
    x2_trade = x2_trade / x2_trade[0]

    return x1_form, x2_form, x1_trade, x2_trade, dates_trade


# ---------------------------------------------------------------------------
# Neural 3-state KalmanNet training
# ---------------------------------------------------------------------------

def _build_F(rho: float) -> torch.Tensor:
    return torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, rho, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)


def train_gated_neural(
    model: KalmanNetGain,
    x1_form: np.ndarray,
    x2_form: np.ndarray,
    params,
    *,
    epochs_stage1: int = 10,
    epochs_stage2: int = 10,
    lr: float = 1e-3,
    verbose: bool = True,
):
    """
    Two-stage training for 3-state Neural KalmanNet on formation data.
    Stage 1: MSE on X2 reconstruction.
    Stage 2: Differentiable Sharpe fine-tuning.
    """
    optimizer   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse_loss    = nn.MSELoss()
    sharpe_loss = DifferentiableSharpeLoss()
    F_mat = _build_F(params.rho)

    T = len(x1_form)
    batch_y = torch.tensor(
        np.stack([x1_form, x2_form], axis=1), dtype=torch.float32
    ).unsqueeze(0)  # (1, T, 2)

    # Build time-varying H: shape (T, 1, 3),  H_t = [x1[t], 1, 1]
    batch_H = torch.zeros(1, T, 1, 3, dtype=torch.float32)
    for t in range(T):
        batch_H[0, t, 0] = torch.tensor([x1_form[t], 1.0, 1.0])

    # Returns for Sharpe objective: delta-X2 minus beta0 * delta-X1
    returns = np.zeros(T)
    for t in range(1, T):
        returns[t] = ((x2_form[t] - x2_form[t - 1])
                      - params.beta0 * (x1_form[t] - x1_form[t - 1]))
    batch_ret = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)

    def _init_state(B):
        r0 = float(x2_form[0]) - params.beta0 * float(x1_form[0])
        z  = torch.zeros(B, 3)
        z[:, 0] = params.beta0
        z[:, 1] = 0.0
        z[:, 2] = r0
        return z

    # ---- Stage 1: MSE ----
    if verbose:
        print("    Stage 1: MSE pre-training...")
    for ep in range(epochs_stage1):
        B = 1
        optimizer.zero_grad()
        h_prev = torch.zeros(B, model.hidden_dim)
        x_upd  = _init_state(B)
        loss   = torch.tensor(0.0)
        for t in range(1, T):
            x_prior = torch.matmul(x_upd, F_mat.T)
            H_t     = batch_H[:, t]                          # (1, 1, 3)
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)  # (1, 1)
            e_t     = batch_y[:, t, 1:2] - y_prior          # innovation on X2
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr    = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_upd   = torch.clamp(x_prior + corr, -100.0, 100.0)
            loss    = loss + mse_loss(y_prior[:, 0], batch_y[:, t, 1])
        (loss / T).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"      Epoch {ep+1}/{epochs_stage1}  MSE={loss.item()/T:.6f}")

    # ---- Stage 2: Sharpe ----
    if verbose:
        print("    Stage 2: Differentiable Sharpe fine-tuning...")
    for ep in range(epochs_stage2):
        B = 1
        optimizer.zero_grad()
        h_prev = torch.zeros(B, model.hidden_dim)
        x_upd  = _init_state(B)
        e_traj = []
        for t in range(1, T):
            x_prior = torch.matmul(x_upd, F_mat.T)
            H_t     = batch_H[:, t]
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            e_t     = batch_y[:, t, 1:2] - y_prior
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr    = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_upd   = torch.clamp(x_prior + corr, -100.0, 100.0)
            e_traj.append(e_t[:, 0])
        e_traj = torch.stack(e_traj, dim=1)           # (1, T-1)
        sigma  = torch.std(e_traj, dim=1, keepdim=True).expand_as(e_traj)
        sl = sharpe_loss(e_traj, sigma, batch_ret[:, 1:])
        sl.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"      Epoch {ep+1}/{epochs_stage2}  neg-Sharpe={sl.item():.4f}")

    return model


def run_neural_filter_3state(
    model: KalmanNetGain,
    x1_test: np.ndarray,
    x2_test: np.ndarray,
    params,
):
    """Out-of-sample pass of the trained 3-state neural KalmanNet."""
    T     = len(x1_test)
    F_mat = _build_F(params.rho)

    beta_traj = []
    M_traj    = []
    innov_var_traj = []

    model.eval()
    with torch.no_grad():
        r0    = float(x2_test[0]) - params.beta0 * float(x1_test[0])
        x_upd = torch.zeros(1, 3)
        x_upd[0, 0] = params.beta0
        x_upd[0, 1] = 0.0
        x_upd[0, 2] = r0
        h_prev = torch.zeros(1, model.hidden_dim)

        # Record t=0 state
        beta_traj.append(float(x_upd[0, 0]))
        M_traj.append(float(x_upd[0, 1]))
        innov_var_traj.append(np.nan)

        for t in range(1, T):
            x_prior = torch.matmul(x_upd, F_mat.T)
            x1t = float(x1_test[t])
            H_t = torch.tensor([[[x1t, 1.0, 1.0]]], dtype=torch.float32)  # (1,1,3)
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)   # (1,1)
            e_t = torch.tensor([[float(x2_test[t]) - float(y_prior[0, 0])]],
                               dtype=torch.float32)
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr    = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_upd   = torch.clamp(x_prior + corr, -100.0, 100.0)

            beta_traj.append(float(x_upd[0, 0]))
            M_traj.append(float(x_upd[0, 1]))
            innov_var_traj.append(float(e_t[0, 0] ** 2))   # proxy

    return np.array(beta_traj), np.array(M_traj), np.array(innov_var_traj)


# ---------------------------------------------------------------------------
# Run one pair — returns dict of model results
# ---------------------------------------------------------------------------

def prepare_pair(
    t1: str, t2: str,
    total_days: int = None,
    verbose: bool = True,
) -> dict:
    """
    Fetch data, fit models ONCE, compute gates and train neural net ONCE.
    Returns a cached dict that can be backtested for multiple tau configs.
    """
    total_days = total_days or (IN_SAMPLE_DAYS + OUT_SAMPLE_DAYS + 200)
    pair_name  = f"{t1}-{t2}"

    print(f"\n{'='*20} {pair_name} {'='*20}")
    sys.stdout.flush()

    # ---- 1. Fetch (Canonical 2009-2021 window) ----
    print(f"  Fetching data (2009-2021 window)..."); sys.stdout.flush()
    try:
        p = fetch_pair_data(t1, t2)
        x1_form = p.x1_formation / p.x1_formation[0]
        x2_form = p.x2_formation / p.x2_formation[0]
        x1_test = p.x1_trading   / p.x1_trading[0]
        x2_test = p.x2_trading   / p.x2_trading[0]
        dates_test = p.dates_trading
    except Exception as e:
        print(f"  [SKIP] fetch failed for {pair_name}: {e}")
        return {}

    # ---- 2. Fit 3-state MLE (once per pair) ----
    print(f"  Fitting Static PCI (MLE)..."); sys.stdout.flush()
    params_static,  _ = fit_pci3(x1_form, x2_form, allow_dynamic_beta=False, verbose=False)
    print(f"  Fitting Dynamic PCI (MLE)..."); sys.stdout.flush()
    params_dynamic, _ = fit_pci3(x1_form, x2_form, allow_dynamic_beta=True,  verbose=False)
    print(f"  MLE done. rho={params_dynamic.rho:.4f}, HL={params_dynamic.half_life:.1f}d, "
          f"R2_MR={params_dynamic.r2_mr:.4f}")

    # ---- 3. Screening ----
    is_suitable, screen_metrics = screen_pair(x1_form, x2_form, params_dynamic)
    print(describe_screening(pair_name, is_suitable, screen_metrics))

    # ---- 4. Kalman filter passes (once per pair) ----
    print(f"  Running Kalman filters..."); sys.stdout.flush()
    kf_static  = PCI3KalmanFilter(params_static)
    kf_dynamic = PCI3KalmanFilter(params_dynamic)
    out_static  = kf_static.filter(x1_test, x2_test)
    out_dynamic = kf_dynamic.filter(x1_test, x2_test)

    # ---- 5. Gates (once per pair) ----
    vol_gate  = compute_vol_gate(out_dynamic["M"], short_window=20, vol_cap_ratio=3.0)
    ar1_gate  = compute_regime_mask(out_dynamic["M"], rolling_window=60, max_half_life=126.0)
    combined  = (vol_gate * ar1_gate).astype(float)
    active_pct = combined.mean() * 100
    print(f"  Dual Gate active {active_pct:.1f}% of trading days")

    hmm_weights, hmm_probs = compute_hmm_regime_filter(out_dynamic["M"], rolling_window=60)
    hmm_pct = (hmm_weights > 0).mean() * 100
    print(f"  HMM Gate active {hmm_pct:.1f}% of trading days")

    # ---- 6. Neural KalmanNet (train once per pair) ----
    print(f"  Training Neural KalmanNet..."); sys.stdout.flush()
    neural_model = KalmanNetGain(state_dim=3, obs_dim=1, hidden_dim=32)
    neural_model = train_gated_neural(
        neural_model, x1_form, x2_form, params_dynamic, verbose=verbose
    )
    beta_neural, M_neural, iv_neural = run_neural_filter_3state(
        neural_model, x1_test, x2_test, params_dynamic
    )
    print(f"  Neural filter done.")

    return {
        "pair": pair_name, "t1": t1, "t2": t2,
        "suitable": is_suitable, "screen": screen_metrics,
        "x1_test": x1_test, "x2_test": x2_test,
        "dates_test": dates_test,
        "out_static": out_static, "out_dynamic": out_dynamic,
        "combined_gate": combined,
        "hmm_weights": hmm_weights, "hmm_probs": hmm_probs,
        "beta_neural": beta_neural, "M_neural": M_neural, "iv_neural": iv_neural,
        "params_static": params_static, "params_dynamic": params_dynamic,
    }


def run_pair(
    t1: str, t2: str,
    total_days: int = None,
    tau_key: str = DEFAULT_TAU,
    verbose: bool = True,
    cached: dict = None,
    include_dynamic: bool = False,
) -> dict:
    """Run backtest for one pair at one tau config. Uses cached fit if provided."""
    c = cached if cached else prepare_pair(t1, t2, total_days, verbose)
    if not c:
        return {}

    pair_name   = c["pair"]
    x1_test     = c["x1_test"]; x2_test = c["x2_test"]
    dates_test  = c["dates_test"]
    out_static  = c["out_static"]; out_dynamic = c["out_dynamic"]
    combined    = c["combined_gate"]
    hmm_weights = c["hmm_weights"]
    beta_neural = c["beta_neural"]; M_neural = c["M_neural"]; iv_neural = c["iv_neural"]

    tau_cfg = TAU_CONFIGS[tau_key]
    results = {}

    model_configs = [
        ("static",       out_static["beta"],  out_static["M"],  None,         out_static["innovation_var"],  False),
        ("gated",        out_dynamic["beta"], out_dynamic["M"], combined,     out_dynamic["innovation_var"], True),
        ("hmm_gated",    out_dynamic["beta"], out_dynamic["M"], hmm_weights,  out_dynamic["innovation_var"], True),
        ("neural_gated", beta_neural,         M_neural,         combined,     iv_neural,                     True),
    ]
    if include_dynamic:
        model_configs.insert(1, ("dynamic", out_dynamic["beta"], out_dynamic["M"], None, out_dynamic["innovation_var"], False))

    for model_label, beta_t, M_t, gate, innov_var, use_usizing in model_configs:
        df_bt, trade_log = backtest_v2(
            x1_test, x2_test, beta_t, M_t,
            tau_open=tau_cfg["tau_open"],
            tau_close=tau_cfg["tau_close"],
            tau_partial=tau_cfg["tau_partial"],
            stop_loss=0.10, z_window=80, transaction_cost=0.001,
            regime_mask=gate, innov_var=innov_var, uncertainty_sizing=use_usizing,
        )
        perf = performance_summary_v2(df_bt)
        results[model_label] = {"perf": perf, "df": df_bt, "trades": trade_log, "dates": dates_test}
        print(
            f"  [{tau_key}|{model_label:15s}] "
            f"TotRet={perf['total_return']*100:+.2f}%  "
            f"Ann={perf['ann_return']*100:+.2f}%  "
            f"Sharpe={perf['sharpe']:+.2f}  "
            f"MaxDD={perf['max_drawdown']*100:.2f}%  "
            f"Calmar={perf.get('calmar', float('nan')):+.2f}  "
            f"Trades={perf['n_position_changes']}"
        )

    return {
        "pair": pair_name,
        "suitable": c["suitable"], "screen": c["screen"],
        "models": results,
        "params_static": c["params_static"], "params_dynamic": c["params_dynamic"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(all_results: list, tau_key: str, out_path: str, model_names: list = None):
    MODEL_NAMES = model_names or ["static", "dynamic", "gated", "hmm_gated", "neural_gated"]
    header = (f"{'Pair':<14} | {'Model':<15} | "
              f"{'TotRet':>9} | {'AnnRet':>9} | {'Sharpe':>7} | "
              f"{'MaxDD':>8} | {'Calmar':>7} | {'Trades':>7}")
    sep = "-" * len(header)
    lines = [
        f"PCI PAIRS TRADING -- v2 RESULTS  [threshold: {tau_key}]",
        "=" * len(header), header, sep,
    ]
    for r in all_results:
        if not r:
            continue
        pair = r["pair"]
        suit = " [PASS]" if r["suitable"] else " [REJ]"
        for m in MODEL_NAMES:
            if m not in r.get("models", {}):
                continue
            perf = r["models"][m]["perf"]
            def _f(v, pct=False, dp=2):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "N/A"
                return f"{v*100:.{dp}f}%" if pct else f"{v:.{dp}f}"
            lines.append(
                f"{pair+suit:<14} | {m:<15} | "
                f"{_f(perf['total_return'], pct=True):>9} | "
                f"{_f(perf['ann_return'],   pct=True):>9} | "
                f"{_f(perf['sharpe']):>7} | "
                f"{_f(perf['max_drawdown'], pct=True):>8} | "
                f"{_f(perf.get('calmar')):>7} | "
                f"{perf['n_position_changes']:>7}"
            )
        lines.append(sep)

    text = "\n".join(lines)
    print("\n" + text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nResults saved -> {out_path}")
    return text


def save_equity_csv(all_results: list, out_dir: str):
    """Save per-pair equity curves as CSV for plotting."""
    for r in all_results:
        if not r:
            continue
        pair = r["pair"].replace("-", "_")
        frames = {}
        for m, mdata in r.get("models", {}).items():
            s = pd.Series(
                mdata["df"]["equity"].values,
                index=mdata["dates"],
                name=m,
            )
            frames[m] = s
        if frames:
            df_out = pd.DataFrame(frames)
            csv_path = os.path.join(out_dir, f"equity_{pair}.csv")
            df_out.to_csv(csv_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _safe_plot(all_results, model_names, tau_key, out_dir):
    """Generate summary plots if plot_utils available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Equity grid
        n_pairs = sum(1 for r in all_results if r and r.get("models"))
        cols = 4
        rows = max(1, (n_pairs + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), squeeze=False)
        fig.suptitle(f"Equity Curves -- {tau_key}", fontsize=14, fontweight="bold")

        ax_flat = axes.flatten()
        ai = 0
        for r in all_results:
            if not r or not r.get("models"):
                continue
            ax = ax_flat[ai]; ai += 1
            colors = {
                "static": "#888888", "dynamic": "#4a90d9",
                "gated": "#f5a623", "hmm_gated": "#9013fe", "neural_gated": "#7ed321",
            }
            for m in model_names:
                if m in r["models"]:
                    eq = r["models"][m]["df"]["equity"].values
                    ax.plot(eq, label=m, color=colors.get(m, None), linewidth=0.9)
            ax.axhline(1.0, color="red", lw=0.5, linestyle="--")
            suit = " [PASS]" if r["suitable"] else " [REJ]"
            ax.set_title(f"{r['pair']}{suit}", fontsize=9)
            ax.set_xlabel(""); ax.set_ylabel("Equity")
            ax.legend(fontsize=6, ncol=2)

        for j in range(ai, len(ax_flat)):
            ax_flat[j].set_visible(False)

        plt.tight_layout()
        path_eq = os.path.join(out_dir, f"equity_grid_{tau_key}.png")
        plt.savefig(path_eq, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Equity grid saved -> {path_eq}")

        # 2. Sharpe bar chart
        pairs   = [r["pair"] for r in all_results if r and r.get("models")]
        x = np.arange(len(pairs))
        width = 0.8 / max(1, len(model_names))
        fig2, ax2 = plt.subplots(figsize=(max(10, len(pairs) * 1.3), 5))
        for i, m in enumerate(model_names):
            sharpes = []
            for r in all_results:
                if r and r.get("models") and m in r["models"]:
                    v = r["models"][m]["perf"]["sharpe"]
                    sharpes.append(v if np.isfinite(v) else 0.0)
                else:
                    sharpes.append(0.0)
            ax2.bar(x + i * width, sharpes, width, label=m, color=colors.get(m, None))
        ax2.axhline(0, color="black", lw=0.5)
        ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax2.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
        ax2.set_title(f"Out-of-Sample Sharpe by Model -- {tau_key}")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.legend()
        plt.tight_layout()
        path_sh = os.path.join(out_dir, f"sharpe_bar_{tau_key}.png")
        plt.savefig(path_sh, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Sharpe bar chart saved -> {path_sh}")

        # 3. Screening heatmap
        scr_rows = []
        for r in all_results:
            if r and r.get("screen"):
                c = r["screen"]["criteria"]
                row = {"pair": r["pair"], **{k: int(v) for k, v in c.items()}}
                scr_rows.append(row)
        if scr_rows:
            scr_df = pd.DataFrame(scr_rows).set_index("pair")
            fig3, ax3 = plt.subplots(figsize=(len(scr_df.columns) * 1.2, len(scr_df) * 0.5 + 1))
            im = ax3.imshow(scr_df.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax3.set_xticks(range(len(scr_df.columns)))
            ax3.set_xticklabels(scr_df.columns, rotation=45, ha="right", fontsize=8)
            ax3.set_yticks(range(len(scr_df)))
            ax3.set_yticklabels(scr_df.index, fontsize=8)
            ax3.set_title("Pair Screening Criteria (green=pass)")
            plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.04)
            plt.tight_layout()
            path_sc = os.path.join(out_dir, f"screening_heatmap.png")
            plt.savefig(path_sc, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  Screening heatmap saved -> {path_sc}")

    except Exception as e:
        print(f"  [Plot skipped: {e}]")


# ---------------------------------------------------------------------------
# Multi-tau sweep
# ---------------------------------------------------------------------------

def run_tau_sweep(all_results_by_tau: dict, out_dir: str):
    """Compare best tau for each pair on Sharpe."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows_data = []
        for tau_key, results in all_results_by_tau.items():
            for r in results:
                if not r or not r.get("models"):
                    continue
                pair = r["pair"]
                for model_label, mdata in r["models"].items():
                    sharpe = mdata["perf"]["sharpe"]
                    rows_data.append({"tau": tau_key, "pair": pair, "model": model_label, "sharpe": sharpe})

        df_sweep = pd.DataFrame(rows_data)
        if df_sweep.empty:
            return

        order = ["static", "dynamic", "gated", "hmm_gated", "neural_gated"]
        models_in_df = [m for m in order if m in df_sweep["model"].unique()]
        fig, axes = plt.subplots(1, len(models_in_df), figsize=(max(15, len(models_in_df) * 4.5), 5), sharey=False)
        if len(models_in_df) == 1:
            axes = [axes]
        for i, model_label in enumerate(models_in_df):
            sub = df_sweep[df_sweep["model"] == model_label].pivot(index="pair", columns="tau", values="sharpe")
            if sub.empty:
                continue
            sub.plot(kind="bar", ax=axes[i], rot=45)
            axes[i].set_title(f"Sharpe by tau -- {model_label}")
            axes[i].axhline(0, color="black", lw=0.5)
            axes[i].legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, "tau_sweep_sharpe.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Tau sweep plot saved -> {path}")
    except Exception as e:
        print(f"  [Tau sweep plot failed: {e}]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCI Pairs Trading Pipeline v2")
    parser.add_argument("--pairs", choices=["all", "equity", "commodity"], default="all")
    parser.add_argument("--tau",   default=None,
                        help="tau1.0 / tau1.5 / tau1.75  (default: run all three)")
    parser.add_argument("--fast",  action="store_true",
                        help="Run only tau=1.5 (fastest single-pass mode)")
    parser.add_argument("--include-dynamic", action="store_true",
                        help="Include raw un-gated dynamic PCI model (default: static, gated, hmm_gated, neural_gated)")
    parser.add_argument("--out",   default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.pairs == "equity":
        pairs = TARGET_PAIRS
    elif args.pairs == "commodity":
        pairs = COMMODITY_ENERGY_PAIRS
    else:
        pairs = ALL_PAIRS

    if args.fast:
        tau_keys = ["tau1.5"]
    elif args.tau:
        tau_keys = [args.tau]
    else:
        tau_keys = list(TAU_CONFIGS.keys())

    model_names = ["static", "dynamic", "gated", "hmm_gated", "neural_gated"] if args.include_dynamic else ["static", "gated", "hmm_gated", "neural_gated"]

    print(f"\n{'#'*60}")
    print(f"# PCI PAIRS TRADING PIPELINE v2")
    print(f"# In-Sample  : {IN_SAMPLE_DAYS} days (2009-09-01 to 2017-08-30)")
    print(f"# Out-Sample : {OUT_SAMPLE_DAYS} days (2017-08-31 to 2021-08-31)")
    print(f"# Pairs      : {len(pairs)}  ({args.pairs})")
    print(f"# Models     : {model_names}")
    print(f"# Tau keys   : {tau_keys}")
    print(f"# Mode       : {'FAST (tau1.5 only)' if args.fast else 'FULL SWEEP'}")
    print(f"{'#'*60}\n")
    sys.stdout.flush()

    all_results_by_tau = {}
    t_global = time.time()

    # -----------------------------------------------------------------------
    # PHASE 1: Fetch + Fit + Train ONCE per pair (expensive part)
    # -----------------------------------------------------------------------
    print("\nPHASE 1: Fitting models for all pairs...\n")
    pair_cache = {}
    for i, (t1, t2) in enumerate(pairs):
        t0 = time.time()
        print(f"[{i+1}/{len(pairs)}] Processing {t1}-{t2}")
        sys.stdout.flush()
        cache = prepare_pair(t1, t2, verbose=verbose)
        pair_cache[(t1, t2)] = cache
        elapsed = time.time() - t0
        print(f"  >> Done in {elapsed:.1f}s")
        sys.stdout.flush()

    # -----------------------------------------------------------------------
    # PHASE 2: Sweep tau thresholds (fast — just backtests, no refitting)
    # -----------------------------------------------------------------------
    print(f"\nPHASE 2: Backtesting across {len(tau_keys)} tau threshold(s)...\n")
    for tau_key in tau_keys:
        print(f"\n{'*'*50}")
        print(f"*  RUNNING  tau = {tau_key}")
        print(f"{'*'*50}")
        sys.stdout.flush()
        results = []
        for t1, t2 in pairs:
            cache = pair_cache.get((t1, t2), {})
            if not cache:
                results.append({})
                continue
            res = run_pair(t1, t2, tau_key=tau_key, verbose=verbose, cached=cache, include_dynamic=args.include_dynamic)
            results.append(res)

        out_path = args.out or os.path.join(OUTPUTS_DIR, f"results_v2_{tau_key}.txt")
        print_report(results, tau_key, out_path, model_names)
        save_equity_csv(results, OUTPUTS_DIR)
        _safe_plot(results, model_names, tau_key, OUTPUTS_DIR)
        all_results_by_tau[tau_key] = results

    # Cross-tau sweep
    if len(tau_keys) > 1:
        run_tau_sweep(all_results_by_tau, OUTPUTS_DIR)

    # Sync primary conclusive files to outputs root
    try:
        import shutil, glob
        root_out = "outputs"
        eq_dir = os.path.join(root_out, "equity_curves")
        sweep_dir = os.path.join(root_out, "threshold_sweeps")
        os.makedirs(eq_dir, exist_ok=True)
        os.makedirs(sweep_dir, exist_ok=True)

        for f in glob.glob(os.path.join(OUTPUTS_DIR, "equity_*.csv")):
            shutil.copy2(f, os.path.join(eq_dir, os.path.basename(f)))
        for src_name, dst_name in [
            ("results_v2_tau1.5.txt", "benchmark_results_primary.txt"),
            ("equity_grid_tau1.5.png", "equity_grid_primary.png"),
            ("sharpe_bar_tau1.5.png", "sharpe_comparison_primary.png"),
            ("tau_sweep_sharpe.png", "tau_sweep_comparison.png"),
            ("screening_heatmap.png", "screening_criteria_heatmap.png"),
        ]:
            sp = os.path.join(OUTPUTS_DIR, src_name)
            if os.path.exists(sp):
                shutil.copy2(sp, os.path.join(root_out, dst_name))
        for sweep_f in glob.glob(os.path.join(OUTPUTS_DIR, "*tau*")):
            shutil.copy2(sweep_f, os.path.join(sweep_dir, os.path.basename(sweep_f)))

        # Clean staging directory
        if os.path.exists(OUTPUTS_DIR):
            shutil.rmtree(OUTPUTS_DIR, ignore_errors=True)
    except Exception as e:
        pass

    total_elapsed = time.time() - t_global
    print(f"\nAll outputs organized in: outputs/")
    print(f"Total elapsed: {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
