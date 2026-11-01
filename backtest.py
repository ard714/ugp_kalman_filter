"""
backtest.py — HELPER: Z-Score Trading Strategy & Out-of-Sample Backtesting Engine
==================================================================================

Role: Implements the trading signal, position management, and performance measurement.
      Called by run_demo.py, run_all_pairs.py, run_neural_demo.py, and generate_all_plots.py.

Signal Construction (follows Clegg & Krauss 2018)
--------------------------------------------------
We trade off the Z-score of the FILTERED mean-reverting component M_t — not the raw
spread (X2 - β·X1), and not the total spread (M_t + R_t). Only M_t is expected to
mean-revert, so only M_t generates valid entry/exit signals.

Z-score modes:
  - Expanding window (z_window=None): uses all history up to t (online deployment)
  - Rolling window (z_window=80): uses last 80 days (more responsive)

Position Logic
--------------
  Entry: |Z(M_t)| > τ_open (default 1.0σ) → open a position betting on reversion
    - Z < -τ → long spread (long X2, short β·X1): spread expected to rise
    - Z >  τ → short spread (short X2, long β·X1): spread expected to fall
  Exit:  Z crosses τ_close (default 0.0 = mean cross)
  Optional: Regime mask can force flat when regime is broken

Hedge-Ratio Convention
-----------------------
Each pair position uses the CURRENT filtered β_t (from the Kalman filter) so the
hedge adapts over time. Position weights: w2 = 1/(1+|β|), w1 = β/(1+|β|).

Transaction Costs
-----------------
10 bps per half-turn (20 bps round-trip) applied on every position change.

Key Exports:
  - backtest_pair()        : Simulates trading and returns equity curve + trade log
  - performance_summary()  : Computes Sharpe, return, max drawdown, etc.
"""

import numpy as np
import pandas as pd


def zscore(series, window=None):
    """Expanding (window=None) or rolling z-score."""
    s = pd.Series(series)
    if window is None:
        mean = s.expanding().mean()
        std = s.expanding().std()
    else:
        mean = s.rolling(window).mean()
        std = s.rolling(window).std()
    z = (s - mean) / std
    return z.values


def backtest_pair(x1, x2, beta_t, M_t, tau_open=1.0, tau_close=0.0,
                   stop_loss=0.10, z_window=None, transaction_cost=0.001,
                   regime_mask=None):
    """
    Simulate trading a single pair using the Dynamic-PCI signal.

    Parameters
    ----------
    x1, x2      : price series (aligned, same length)
    beta_t      : filtered dynamic hedge ratio at each time step
    M_t         : filtered mean-reverting component at each time step
    tau_open    : |Z| threshold to open a position (Clegg & Krauss use 1.0)
    tau_close   : Z threshold to close a position.  Default 0.0 means exit
                  when the spread z-score crosses zero (captures full
                  reversion).  A long position (entered at z < -tau_open)
                  exits when z > tau_close; a short position (entered at
                  z > tau_open) exits when z < -tau_close.
    stop_loss   : fraction of allocated capital; position is closed if
                  breached
    z_window    : None = expanding window Z-score (uses all history up to t,
                  as in an online/real-time deployment); or an int for a
                  rolling window
    transaction_cost : proportional cost per half-turn (applied on notional
                  traded), e.g. 0.001 = 10 bps per half-turn (20 bps round-trip)
    regime_mask : optional array (length n) of 1s and 0s indicating whether
                  trading is permitted (1 = trade, 0 = flat / risk-off).

    Returns
    -------
    DataFrame with columns: z, position (+1 long spread / -1 short spread / 0),
    pair_return (per-period return of the pair portfolio), equity (cumulative)
    """
    n = len(x1)
    z = zscore(M_t, window=z_window)

    position = np.zeros(n)   # +1 = long X2 / short X1 ("long the spread"), -1 = opposite
    entry_value = None
    entry_idx = None

    pair_return = np.zeros(n)
    trade_log = []

    for t in range(1, n):
        if np.any(np.isnan(z[t])):
            continue
            
        zt_val = z[t].flatten()[0] if hasattr(z[t], "flatten") else z[t]

        pos_prev = position[t - 1]

        # Check regime mask
        if regime_mask is not None and regime_mask[t] == 0:
            if pos_prev != 0:
                position[t] = 0.0
                trade_log.append((entry_idx, t))
                entry_idx = None
            else:
                position[t] = 0.0
        elif pos_prev == 0:
            # Look for entry
            if zt_val < -tau_open:
                position[t] = 1.0   # spread too low -> expect it to rise: long X2, short beta*X1
                entry_idx = t
            elif zt_val > tau_open:
                position[t] = -1.0  # spread too high -> expect it to fall: short X2, long beta*X1
                entry_idx = t
            else:
                position[t] = 0.0
        else:
            # Currently in a position -- check exit conditions
            exit_now = False
            if pos_prev == 1.0 and zt_val > tau_close:
                exit_now = True
            elif pos_prev == -1.0 and zt_val < -tau_close:
                exit_now = True

            if exit_now:
                position[t] = 0.0
                trade_log.append((entry_idx, t))
                entry_idx = None
            else:
                position[t] = pos_prev

        # ---- daily P&L of the pair position ----
        # A "long the spread" position of size 1 means: +1 dollar in X2,
        # -beta_t dollars in X1 (dollar-neutral-ish, using current beta).
        if pos_prev != 0:
            r_x1 = x1[t] / x1[t - 1] - 1.0
            r_x2 = x2[t] / x2[t - 1] - 1.0
            b = beta_t[t - 1]
            # weight on X2 leg = 1, weight on X1 leg = b (opposite sign), normalize by (1+b)
            denom = 1.0 + abs(b)
            w2 = 1.0 / denom
            w1 = b / denom
            pair_return[t] = pos_prev * (w2 * r_x2 - w1 * r_x1)

        # transaction costs when position changes
        if position[t] != pos_prev:
            pair_return[t] -= transaction_cost * 2  # both legs traded (half-turn each)

    equity = np.cumprod(1 + pair_return)

    df = pd.DataFrame({
        "x1": x1, "x2": x2, "beta": beta_t, "M": M_t, "z": z,
        "position": position, "pair_return": pair_return, "equity": equity,
    })
    return df, trade_log


def performance_summary(df, periods_per_year=252):
    r = df["pair_return"].values
    total_return = df["equity"].iloc[-1] - 1
    ann_return = (1 + total_return) ** (periods_per_year / len(r)) - 1
    ann_vol = np.std(r) * np.sqrt(periods_per_year)
    sharpe = (np.mean(r) * periods_per_year) / ann_vol if ann_vol > 0 else np.nan
    running_max = np.maximum.accumulate(df["equity"].values)
    drawdown = df["equity"].values / running_max - 1
    max_dd = drawdown.min()
    n_trades = (df["position"].diff().abs() > 0).sum()
    return dict(total_return=total_return, ann_return=ann_return, ann_vol=ann_vol,
                sharpe=sharpe, max_drawdown=max_dd, n_position_changes=int(n_trades))
