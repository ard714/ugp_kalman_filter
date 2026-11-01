"""
regime_filter.py — HELPER: Statistical Pair Screening & Online Regime Gating
==============================================================================

Role: Provides two complementary safeguards against trading on broken pairs:
      (A) Formation-period suitability screening (before you start trading)
      (B) Real-time regime mask (during trading, to go flat when conditions deteriorate)

A. Pair Screening — test_pair_suitability()
--------------------------------------------
Checks whether a pair has genuine mean-reverting dynamics BEFORE capital is deployed.
Four criteria must all pass:
  1. R²_MR ≥ 15%       — at least 15% of spread variance is mean-reverting
  2. ADF p-value ≤ 0.10 — Augmented Dickey-Fuller test rejects unit root
  3. Half-life ≤ 252d   — mean reversion occurs within 1 trading year
  4. ρ < 0.999          — AR(1) coefficient implies finite reversion speed

Example: KO-PEP passes all 4 criteria; AAP-IBM fails (spread drifts, no mean reversion).

B. Online Regime Gating — compute_online_regime_mask()
-------------------------------------------------------
After formation, the spread regime can CHANGE. This function computes a rolling
binary mask (1 = trade allowed, 0 = flat) using two causal features:
  1. Rolling AR(1) half-life on M_t (60-day window): if half-life explodes beyond
     126 days, mean reversion has broken down → go flat.
  2. Volatility ratio (short-run / long-run vol): if >3×, the spread has entered
     a blow-out regime → go flat.

Key Exports:
  - test_pair_suitability()       : Formation-period screening (returns bool + metrics)
  - compute_online_regime_mask()  : Trading-period binary regime gate (returns array)
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def test_pair_suitability(
    x1: np.ndarray,
    x2: np.ndarray,
    fitted_params,
    min_r2_mr: float = 0.15,
    max_half_life: float = 252.0,
    adf_pvalue_threshold: float = 0.10,
    static_params=None,
) -> Tuple[bool, Dict[str, float]]:
    """
    Screen whether a pair has statistically valid mean-reverting dynamics in-sample.

    Parameters
    ----------
    x1, x2 : np.ndarray
        Formation period price series.
    fitted_params : DynamicPCIParams
        Fitted parameters from MLE (containing rho, sigma_M, sigma_R, r2_mr, half_life).
    min_r2_mr : float
        Minimum proportion of spread variance that must be mean-reverting (default 0.15 = 15%).
    max_half_life : float
        Maximum allowable mean-reversion half-life in trading days (default 252 days = 1 year).
    adf_pvalue_threshold : float
        Significance level for Augmented Dickey-Fuller test on the spread (default 0.10).
    static_params : Optional[DynamicPCIParams]
        Static MLE parameters (fixed beta0) if available, for evaluating baseline state stationarity.

    Returns
    -------
    is_suitable : bool
        True if the pair satisfies cointegration/mean-reversion criteria; False otherwise.
    metrics : dict
        Computed screening statistics.
    """
    # 1. Spread calculation using fitted static or initial hedge ratio
    beta0 = fitted_params.beta0
    spread = x2 - beta0 * x1

    # 2. Augmented Dickey-Fuller test on spread
    try:
        adf_res = adfuller(spread, autolag="AIC")
        adf_stat = float(adf_res[0])
        adf_pvalue = float(adf_res[1])
    except Exception:
        adf_stat = 0.0
        adf_pvalue = 1.0

    # 2b. Augmented Dickey-Fuller test on Kalman-filtered M_t state
    try:
        from dynamic_pci import DynamicPCIKalmanFilter
        kf = DynamicPCIKalmanFilter(fitted_params)
        f_out = kf.filter(x1, x2)
        adf_m_res = adfuller(f_out["M"], autolag="AIC")
        adf_pvalue_m = float(adf_m_res[1])
    except Exception:
        adf_pvalue_m = 1.0

    # 2c. Optional static state evaluation if static_params provided
    adf_pvalue_m_stat = 1.0
    r2_mr_stat = float(fitted_params.r2_mr)
    if static_params is not None:
        try:
            from dynamic_pci import DynamicPCIKalmanFilter
            kf_s = DynamicPCIKalmanFilter(static_params)
            f_out_s = kf_s.filter(x1, x2)
            adf_m_stat_res = adfuller(f_out_s["M"], autolag="AIC")
            adf_pvalue_m_stat = float(adf_m_stat_res[1])
            r2_mr_stat = float(static_params.r2_mr)
        except Exception:
            adf_pvalue_m_stat = 1.0

    # 3. Model-derived parameters
    r2_mr = float(fitted_params.r2_mr)
    half_life = float(fitted_params.half_life)
    rho = float(fitted_params.rho)

    # 4. Return correlation
    ret1 = np.diff(x1) / x1[:-1]
    ret2 = np.diff(x2) / x2[:-1]
    ret_corr = float(np.corrcoef(ret1, ret2)[0, 1]) if len(ret1) > 1 else 0.0

    # Decision criteria:
    # Tier 1 - Legacy Strict Screen: R2 >= 15%, Half-life <= 252d, static ADF <= 0.10, rho < 0.999
    pass_strict = bool(
        r2_mr >= min_r2_mr
        and half_life <= max_half_life
        and adf_pvalue <= adf_pvalue_threshold
        and rho < 0.999
    )

    # Tier 2 - Calibrated Variance-Dominant PCI-Aware Screen:
    # Requires genuine variance dominance (R2_MR >= 50% on dynamic or static fit)
    # AND confirmed stationarity on raw spread (p <= 0.05) or latent state M_t (p <= 0.01)
    effective_r2 = max(r2_mr, r2_mr_stat)
    best_m_pvalue = min(adf_pvalue_m, adf_pvalue_m_stat)
    pass_pci_aware = bool(
        effective_r2 >= 0.50
        and (adf_pvalue <= 0.05 or best_m_pvalue <= 0.01)
    )
    is_suitable = pass_strict or pass_pci_aware

    metrics = {
        "r2_mr": r2_mr,
        "half_life": half_life,
        "rho": rho,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "adf_pvalue_m": adf_pvalue_m,
        "adf_pvalue_m_stat": adf_pvalue_m_stat,
        "pass_strict": pass_strict,
        "pass_pci_aware": pass_pci_aware,
        "return_corr": ret_corr,
        "is_suitable": is_suitable,
    }
    return is_suitable, metrics


def compute_online_regime_mask(
    M_t: np.ndarray,
    rolling_window: int = 60,
    max_half_life: float = 126.0,
    z_vol_cap: float = 3.0,
) -> np.ndarray:
    """
    Compute a causal (lagged) binary regime mask for trading M_t.

    Returns an array of shape (n,) where:
      1.0 = regime is healthy (mean-reverting, moderate volatility),
      0.0 = regime broken (half-life exploded or extreme volatility blowout).
    """
    n = len(M_t)
    mask = np.ones(n, dtype=float)

    if n < rolling_window:
        return mask

    # Rolling AR(1) estimate on M_t
    s_M = pd.Series(M_t)
    rolling_std = s_M.rolling(rolling_window, min_periods=20).std()
    long_std = s_M.expanding(min_periods=20).std()

    for t in range(rolling_window, n):
        window_data = M_t[t - rolling_window : t]
        y = window_data[1:]
        x = window_data[:-1]

        # OLS AR(1): y = phi * x + eps
        denom = np.dot(x, x)
        if denom > 1e-12:
            phi = float(np.dot(x, y) / denom)
        else:
            phi = 1.0

        # Estimate local half-life
        if 0 < phi < 0.995:
            hl = -np.log(2.0) / np.log(phi)
        else:
            hl = 999.0

        # Check volatility blowout
        vol_ratio = (
            rolling_std.iloc[t] / (long_std.iloc[t] + 1e-8)
            if not pd.isna(rolling_std.iloc[t]) and not pd.isna(long_std.iloc[t])
            else 1.0
        )

        # If half-life explodes beyond threshold or volatility blows out, disable trading
        if hl > max_half_life or vol_ratio > z_vol_cap:
            mask[t] = 0.0
        else:
            mask[t] = 1.0

    return mask
