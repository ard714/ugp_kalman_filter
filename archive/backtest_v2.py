"""
backtest_v2.py — Reconciled Backtest Engine (v2)
=================================================

Key improvements over backtest.py:
  1. Single return series — all metrics computed from the SAME daily return
     vector `pair_return`.  Equity = cumprod(1 + pair_return). No
     inconsistency between `equity` and performance metrics.
  2. Configurable entry thresholds (tau_open) — supports 1.0, 1.5, 1.75σ.
  3. Partial exits — on 50% z-score reversion, unwind half the position.
  4. Uncertainty-aware position sizing — scale position by 1 / sqrt(innovation_var)
     if the Kalman filter provides it (prevents over-sizing when spread variance
     is inflated).
  5. Volatility gate (deterministic) — suspends trading when rolling 20-day
     realized vol of the spread exceeds a cap (default 3× long-run vol).
     This is computed causally (only past data).
  6. Stop-loss on total pair P&L since entry.
  7. Transaction costs on BOTH legs at every position change.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------

def _zscore_series(series: np.ndarray, window: int = None) -> np.ndarray:
    """Expanding (window=None) or rolling z-score of the series."""
    s = pd.Series(series)
    if window is None:
        mu  = s.expanding().mean()
        sig = s.expanding().std()
    else:
        mu  = s.rolling(window).mean()
        sig = s.rolling(window).std()
    z = (s - mu) / sig.clip(lower=1e-12)
    return z.values


# ---------------------------------------------------------------------------
# Volatility Gate
# ---------------------------------------------------------------------------

def compute_vol_gate(
    M_t: np.ndarray,
    short_window: int = 20,
    long_window: int = 252,
    vol_cap_ratio: float = 3.0,
) -> np.ndarray:
    """
    Deterministic (causal) volatility gate on the filtered spread M_t.

    Returns binary array: 1 = trade allowed, 0 = suspended.
    Gate is 0 when the short-window realized vol exceeds vol_cap_ratio × long-run vol.
    """
    n = len(M_t)
    mask = np.ones(n, dtype=float)
    s = pd.Series(M_t)
    short_vol = s.rolling(short_window, min_periods=5).std()
    long_vol  = s.expanding(min_periods=50).std()

    for t in range(n):
        sv = short_vol.iloc[t]
        lv = long_vol.iloc[t]
        if pd.isna(sv) or pd.isna(lv) or lv < 1e-12:
            continue
        if sv / lv > vol_cap_ratio:
            mask[t] = 0.0

    return mask


# ---------------------------------------------------------------------------
# Regime mask (AR1 half-life rolling gate — same as compute_online_regime_mask)
# ---------------------------------------------------------------------------

def compute_regime_mask(
    M_t: np.ndarray,
    rolling_window: int = 60,
    max_half_life: float = 126.0,
    z_vol_cap: float = 3.0,
) -> np.ndarray:
    """Rolling AR(1)-based regime mask. Combined with vol gate in gated backtest."""
    n = len(M_t)
    mask = np.ones(n, dtype=float)
    if n < rolling_window:
        return mask

    s_M    = pd.Series(M_t)
    r_std  = s_M.rolling(rolling_window, min_periods=20).std()
    l_std  = s_M.expanding(min_periods=20).std()

    for t in range(rolling_window, n):
        w   = M_t[t - rolling_window: t]
        y   = w[1:]; x = w[:-1]
        den = np.dot(x, x)
        phi = float(np.dot(x, y) / den) if den > 1e-12 else 1.0
        hl  = (-np.log(2.0) / np.log(phi)) if 0 < phi < 0.995 else 999.0

        rv = r_std.iloc[t]; lv = l_std.iloc[t]
        vol_ratio = (rv / (lv + 1e-8)) if not (pd.isna(rv) or pd.isna(lv)) else 1.0

        if hl > max_half_life or vol_ratio > z_vol_cap:
            mask[t] = 0.0

    return mask


# ---------------------------------------------------------------------------
# 3-State Hidden Markov Model (Hamilton Filter) Regime Filter
# ---------------------------------------------------------------------------

def compute_hmm_regime_filter(
    M_t: np.ndarray,
    rolling_window: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Online Causal 3-State Hidden Markov Model (Hamilton Filter) on spread dynamics.

    States:
      S = 0: Stationary Mean-Reverting (Low Vol, finite half-life) -> Full position (1.0x)
      S = 1: High-Volatility / Transient Shock                   -> Scaled position (0.5x)
      S = 2: Structural Drift / Divergence (Infinite half-life)  -> Flat / No trade (0.0x)

    Returns:
      weights: array of size n with values in {1.0, 0.5, 0.0}
      probs: (n, 3) causal filtered probabilities P(S_t = s | Y_{1:t})
    """
    n = len(M_t)
    weights = np.ones(n, dtype=float)
    probs = np.zeros((n, 3), dtype=float)
    if n < rolling_window:
        return weights, probs

    # Transition matrix A: high persistence within regime states
    A = np.array([
        [0.96, 0.03, 0.01],
        [0.05, 0.90, 0.05],
        [0.02, 0.03, 0.95],
    ])
    # Composite dislocation metric D = 0.5*(hl/60) + 0.5*vr
    means = np.array([1.0, 2.5, 5.0])
    stds  = np.array([0.5, 1.0, 2.0])

    s_M = pd.Series(M_t)
    r_std = s_M.rolling(rolling_window, min_periods=20).std()
    l_std = s_M.expanding(min_periods=20).std()

    alpha = np.array([0.80, 0.15, 0.05])

    for t in range(rolling_window, n):
        w = M_t[t - rolling_window: t]
        y = w[1:]; x = w[:-1]
        den = np.dot(x, x)
        phi = float(np.dot(x, y) / den) if den > 1e-12 else 1.0
        hl = (-np.log(2.0) / np.log(phi)) if 0 < phi < 0.995 else 999.0
        hl_metric = min(hl / 60.0, 10.0)

        rv = r_std.iloc[t]; lv = l_std.iloc[t]
        vr = (rv / (lv + 1e-8)) if not (pd.isna(rv) or pd.isna(lv)) else 1.0
        vr_metric = min(vr, 10.0)

        obs = 0.5 * hl_metric + 0.5 * vr_metric

        lik = np.zeros(3)
        for s in range(3):
            lik[s] = norm.pdf(obs, loc=means[s], scale=stds[s]) + 1e-12

        prior = alpha @ A
        post = prior * lik
        alpha = post / np.sum(post)
        probs[t] = alpha

        # Decision rule:
        if alpha[2] > 0.40:
            weights[t] = 0.0   # No trade / flat (divergent state)
        elif alpha[1] > 0.40:
            weights[t] = 0.5   # Scaled down 50% (turbulent state)
        else:
            weights[t] = 1.0   # Full trading (stationary state)

    return weights, probs


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def backtest_v2(
    x1: np.ndarray,
    x2: np.ndarray,
    beta_t: np.ndarray,
    M_t: np.ndarray,
    *,
    tau_open: float = 1.0,         # entry z-score threshold
    tau_close: float = 0.0,        # exit z-score threshold (0 = mean cross)
    tau_partial: float = None,     # partial-exit threshold (None = off)
    stop_loss: float = 0.10,       # fraction of capital; -1 to disable
    z_window: int = 80,            # None = expanding, int = rolling
    transaction_cost: float = 0.001,  # per half-turn (10 bps)
    regime_mask: np.ndarray = None,   # external 0/1 gate
    innov_var: np.ndarray = None,     # Kalman innovation variance for sizing
    uncertainty_sizing: bool = False,
):
    """
    Simulate pair trading on the filtered spread M_t.

    A "long spread" position (pos=+1) means:
        +w2 in X2,  -w1 in X1  where w2 = 1/(1+|beta|), w1 = beta/(1+|beta|)

    Returns
    -------
    df : pd.DataFrame
        Columns: z, position, pair_return, equity, cummax, drawdown
    trade_log : list of (entry_idx, exit_idx) tuples
    """
    n = len(x1)
    z = _zscore_series(M_t, window=z_window)

    # Baseline uncertainty weight: 1.0 (no adjustment) unless innov_var given
    if uncertainty_sizing and innov_var is not None:
        # Weight inversely proportional to sqrt(innovation variance); clamp [0.1, 2.0]
        iv = np.array(innov_var, dtype=float)
        iv = np.where(np.isfinite(iv) & (iv > 0), iv, np.nanmedian(iv[iv > 0]) if np.any(iv > 0) else 1.0)
        size_scale = np.clip(1.0 / np.sqrt(iv / (np.nanmedian(iv) + 1e-12)), 0.1, 2.0)
    else:
        size_scale = np.ones(n)

    position    = np.zeros(n)       # raw: +1 / -1 / 0
    pos_size    = np.zeros(n)       # effective size (can be fractional after partial exit)
    pair_return = np.zeros(n)
    trade_log   = []

    entry_value = None              # equity at trade open (for stop-loss tracking)
    entry_idx   = None
    partial_done = False            # has partial exit already fired for this trade?

    for t in range(1, n):
        if np.isnan(z[t]):
            pos_size[t] = pos_size[t - 1]
            position[t] = position[t - 1]
            continue

        zt = float(z[t])
        pos_prev  = position[t - 1]
        size_prev = pos_size[t - 1]

        # ---- Gate check ----
        gated_off = (regime_mask is not None and regime_mask[t] == 0)

        if gated_off and pos_prev != 0:
            # Forced close by gate
            position[t] = 0.0
            pos_size[t] = 0.0
            trade_log.append((entry_idx, t, "gate"))
            entry_idx    = None
            partial_done = False
        elif pos_prev == 0 and not gated_off:
            # Look for new entry
            scale_mult = float(regime_mask[t]) if regime_mask is not None else 1.0
            if zt < -tau_open:
                position[t] = 1.0
                pos_size[t] = size_scale[t] * scale_mult
                entry_idx   = t
                entry_value = 1.0  # normalised equity reference
                partial_done = False
            elif zt > tau_open:
                position[t] = -1.0
                pos_size[t] = size_scale[t] * scale_mult
                entry_idx   = t
                entry_value = 1.0
                partial_done = False
            else:
                position[t] = 0.0
                pos_size[t] = 0.0
        elif pos_prev != 0:
            # Currently holding — check exits
            exit_now = False

            # Normal z-cross exit
            if pos_prev == 1.0 and zt > tau_close:
                exit_now = True
            elif pos_prev == -1.0 and zt < -tau_close:
                exit_now = True

            # Stop-loss
            if stop_loss > 0 and entry_value is not None:
                # equity tracked within the trade (approximated)
                cumret_since_entry = pair_return[entry_idx:t].sum()
                if cumret_since_entry < -stop_loss:
                    exit_now = True

            if exit_now:
                position[t] = 0.0
                pos_size[t] = 0.0
                trade_log.append((entry_idx, t, "normal"))
                entry_idx    = None
                partial_done = False
            else:
                # Partial exit: if halfway to mean, unwind 50%
                if (tau_partial is not None
                        and not partial_done
                        and size_prev > 0.0):
                    if pos_prev == 1.0 and zt > -tau_open / 2.0:
                        pos_size[t] = size_prev * 0.5
                        partial_done = True
                    elif pos_prev == -1.0 and zt < tau_open / 2.0:
                        pos_size[t] = size_prev * 0.5
                        partial_done = True
                    else:
                        pos_size[t] = size_prev
                else:
                    pos_size[t] = size_prev

                position[t] = pos_prev
        else:
            # gated_off and pos_prev == 0: stay flat
            position[t] = 0.0
            pos_size[t] = 0.0

        # ---- Daily P&L ----
        if pos_prev != 0 and size_prev > 0:
            r_x1 = float(x1[t]) / float(x1[t - 1]) - 1.0
            r_x2 = float(x2[t]) / float(x2[t - 1]) - 1.0
            b    = float(beta_t[t - 1])
            denom = 1.0 + abs(b)
            w2 = 1.0 / denom
            w1 = b / denom
            raw_ret = pos_prev * (w2 * r_x2 - w1 * r_x1)
            pair_return[t] = raw_ret * size_prev  # scaled by effective position size

        # Transaction costs on leg changes
        if position[t] != pos_prev:
            pair_return[t] -= transaction_cost * 2

    equity   = np.cumprod(1.0 + pair_return)
    cummax   = np.maximum.accumulate(equity)
    drawdown = equity / cummax - 1.0

    df = pd.DataFrame({
        "x1": x1, "x2": x2, "beta": beta_t, "M": M_t, "z": z,
        "position": position, "pos_size": pos_size,
        "pair_return": pair_return, "equity": equity,
        "cummax": cummax, "drawdown": drawdown,
    })
    return df, trade_log


# ---------------------------------------------------------------------------
# Performance summary — ALL metrics from pair_return
# ---------------------------------------------------------------------------

def performance_summary_v2(df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    Compute performance metrics *entirely* from df['pair_return'].

    equity is recomputed here to guarantee consistency — no divergence
    between reported return series and equity curve.
    """
    r = df["pair_return"].values
    # Recompute equity from the canonical return series
    equity       = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    n            = len(r)

    ann_return = float((1.0 + total_return) ** (periods_per_year / n) - 1.0)
    ann_vol    = float(np.std(r) * np.sqrt(periods_per_year))
    sharpe     = float((np.mean(r) * periods_per_year) / ann_vol) if ann_vol > 1e-12 else np.nan

    running_max = np.maximum.accumulate(equity)
    max_dd      = float((equity / running_max - 1.0).min())

    # Win rate (per-day; only days with non-zero return)
    active = r[r != 0.0]
    win_rate = float(np.mean(active > 0)) if len(active) > 0 else np.nan

    # Calmar ratio
    calmar = (ann_return / abs(max_dd)) if max_dd < -1e-4 else np.nan

    # Average trade count from position changes
    n_trades = int((pd.Series(df["position"]).diff().abs() > 0).sum())

    return dict(
        total_return=total_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        n_position_changes=n_trades,
    )
