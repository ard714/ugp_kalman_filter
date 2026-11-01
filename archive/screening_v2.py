"""
screening_v2.py — Strengthened Pair Screening & Permanent-Drift Rejection
==========================================================================

Improvements over regime_filter.py:
  1. Stricter thresholds: min_r2_mr=0.20, max_half_life=126d, adf_p<0.05.
  2. Transient-component strength: PCI index must exceed a minimum (0.20).
  3. Structural stability: rolling ADF on sub-windows (requires >50% pass rate).
  4. Permanent-drift rejection: R_t variance must NOT dominate excessively.
  5. All results are fully transparent for logging/audit.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------------
# Formation-period suitability screen
# ---------------------------------------------------------------------------

def screen_pair(
    x1: np.ndarray,
    x2: np.ndarray,
    params,                          # PCI3Params from fit_pci3
    min_r2_mr: float = 0.20,         # minimum MR fraction
    max_half_life: float = 126.0,    # max half-life in trading days (6 months)
    adf_pvalue_threshold: float = 0.05,
    min_pci_index: float = 0.20,     # minimum PCI (transient component strength)
    max_r_dominance: float = 0.85,   # reject if R_t variance fraction > this
    stability_windows: int = 4,      # number of sub-windows for rolling ADF
    stability_min_pass_rate: float = 0.50,  # fraction of sub-windows that must pass ADF
) -> Tuple[bool, Dict]:
    """
    Screen whether a pair has statistically valid mean-reverting dynamics.

    Parameters
    ----------
    x1, x2       : Formation-period price series.
    params        : Fitted PCI3Params (from fit_pci3).
    min_r2_mr     : Minimum fraction of spread variance that must be MR.
    max_half_life : Maximum half-life (trading days).
    adf_pvalue_threshold : ADF test significance level.
    min_pci_index : Minimum PCI (transient) index [0,1].
    max_r_dominance : Reject if R_t variance fraction too large (permanent drift).
    stability_windows : Number of sub-windows for rolling ADF stability check.
    stability_min_pass_rate : Required fraction of stable sub-windows.

    Returns
    -------
    is_suitable : bool
    metrics     : dict with all computed screening statistics.
    """
    beta0 = params.beta0
    spread = x2 - beta0 * x1

    # ---- 1. ADF on full formation spread ----
    try:
        adf_res    = adfuller(spread, autolag="AIC")
        adf_stat   = float(adf_res[0])
        adf_pvalue = float(adf_res[1])
    except Exception:
        adf_stat   = 0.0
        adf_pvalue = 1.0

    # ---- 2. Model-derived stats ----
    r2_mr     = float(params.r2_mr)
    half_life = float(params.half_life)
    rho       = float(params.rho)
    pci_index = float(params.pci_index)

    # Variance fractions: sigma_M^2 vs sigma_R^2
    var_M = params.sigma_M ** 2
    var_R = params.sigma_R ** 2
    total_spread_var = var_M + var_R
    r_dominance = var_R / (total_spread_var + 1e-12)

    # ---- 3. Return correlation ----
    ret1 = np.diff(x1) / np.abs(x1[:-1])
    ret2 = np.diff(x2) / np.abs(x2[:-1])
    ret_corr = float(np.corrcoef(ret1, ret2)[0, 1]) if len(ret1) > 1 else 0.0

    # ---- 4. Rolling ADF stability check ----
    n = len(spread)
    window_size = n // stability_windows
    adf_passes = 0
    adf_pvals_rolling = []
    for i in range(stability_windows):
        w = spread[i * window_size: (i + 1) * window_size]
        if len(w) < 20:
            continue
        try:
            p = float(adfuller(w, autolag="AIC")[1])
        except Exception:
            p = 1.0
        adf_pvals_rolling.append(p)
        if p <= adf_pvalue_threshold * 2:   # slightly relaxed for sub-windows
            adf_passes += 1

    stability_pass_rate = adf_passes / max(len(adf_pvals_rolling), 1)

    # ---- 5. Decision ----
    criteria = {
        "r2_mr_ok"       : r2_mr >= min_r2_mr,
        "half_life_ok"   : half_life <= max_half_life,
        "adf_ok"         : adf_pvalue <= adf_pvalue_threshold,
        "rho_ok"         : rho < 0.999,
        "pci_ok"         : pci_index >= min_pci_index,
        "drift_ok"       : r_dominance <= max_r_dominance,
        "stability_ok"   : stability_pass_rate >= stability_min_pass_rate,
    }
    is_suitable = all(criteria.values())

    metrics = dict(
        r2_mr=r2_mr,
        half_life=half_life,
        rho=rho,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        pci_index=pci_index,
        r_dominance=r_dominance,
        return_corr=ret_corr,
        stability_pass_rate=stability_pass_rate,
        adf_pvals_rolling=adf_pvals_rolling,
        is_suitable=is_suitable,
        criteria=criteria,
    )
    return is_suitable, metrics


def describe_screening(pair_name: str, is_suitable: bool, metrics: dict) -> str:
    """Return a human-readable screening summary line."""
    c = metrics["criteria"]
    passed = [k for k, v in c.items() if v]
    failed = [k for k, v in c.items() if not v]
    status = "PASS" if is_suitable else "REJECT"
    return (
        f"  [{status}] {pair_name}: "
        f"R2_MR={metrics['r2_mr']:.3f}, HL={metrics['half_life']:.1f}d, "
        f"ADF_p={metrics['adf_pvalue']:.3f}, PCI={metrics['pci_index']:.3f}, "
        f"R_dom={metrics['r_dominance']:.3f}, Stab={metrics['stability_pass_rate']:.2f}"
        + (f" | FAILED: {failed}" if failed else "")
    )
