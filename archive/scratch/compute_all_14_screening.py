import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

from data_loader import (
    fetch_all_target_pairs,
    ALL_PAIRS,
    DEFAULT_FORMATION_START,
    DEFAULT_FORMATION_END,
)
from dynamic_pci import fit_dynamic_pci, DynamicPCIKalmanFilter
from regime_filter import test_pair_suitability

print("Fetching formation data for all 14 pairs...")
all_data = fetch_all_target_pairs(
    pairs=ALL_PAIRS,
    formation_start=DEFAULT_FORMATION_START,
    formation_end=DEFAULT_FORMATION_END,
)

rows = []
for (t1, t2), pdata in all_data.items():
    p_name = f"{t1}-{t2}"
    s1, s2 = float(pdata.x1_all[0]), float(pdata.x2_all[0])
    x1_norm = pdata.x1_all / s1
    x2_norm = pdata.x2_all / s2
    form_n = len(pdata.x1_formation)

    x1_form = x1_norm[:form_n]
    x2_form = x2_norm[:form_n]

    # Fit static and dynamic PCI
    dyn_params, _ = fit_dynamic_pci(x1_form, x2_form, allow_dynamic_beta=True, maxiter=100)
    static_params, _ = fit_dynamic_pci(x1_form, x2_form, allow_dynamic_beta=False, maxiter=100)
    
    # Run suitability test with dyn_params
    is_suitable, metrics = test_pair_suitability(x1_form, x2_form, dyn_params)
    
    # Also calculate adf on static spread:
    spread_static = x2_form - static_params.beta0 * x1_form
    try:
        adf_stat_static, adf_p_static = adfuller(spread_static, autolag="AIC")[:2]
    except Exception:
        adf_p_static = 1.0

    # Filtered M_t from dyn_pci
    dyn_kf = DynamicPCIKalmanFilter(dyn_params)
    out_dyn = dyn_kf.filter(x1_form, x2_form)
    try:
        adf_stat_M_dyn, adf_p_M_dyn = adfuller(out_dyn["M"], autolag="AIC")[:2]
    except Exception:
        adf_p_M_dyn = 1.0

    # Filtered M_t from static_pci
    static_kf = DynamicPCIKalmanFilter(static_params)
    out_static = static_kf.filter(x1_form, x2_form)
    try:
        adf_stat_M_stat, adf_p_M_stat = adfuller(out_static["M"], autolag="AIC")[:2]
    except Exception:
        adf_p_M_stat = 1.0

    rows.append({
        "pair": p_name,
        "r2_mr": metrics["r2_mr"],
        "half_life": metrics["half_life"],
        "rho": metrics["rho"],
        "adf_p_spread": metrics["adf_pvalue"],
        "adf_p_spread_static": adf_p_static,
        "adf_p_M_dyn": adf_p_M_dyn,
        "adf_p_M_stat": adf_p_M_stat,
        "pass_strict": metrics["pass_strict"],
        "pass_pci_aware": metrics["pass_pci_aware"],
        "is_suitable": is_suitable,
    })

df = pd.DataFrame(rows)
print("\n=== SCREENING METRICS FOR ALL 14 PAIRS ===")
print(df.to_string(index=False))
df.to_csv("outputs/all_14_screening_metrics.csv", index=False)
