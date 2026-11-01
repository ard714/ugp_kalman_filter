import os
import pandas as pd
import numpy as np

# Load actual performance results
res_lines = """
AAP-IBM | static | -7.27% | 0.01 | -44.52% | neural | -30.73% | -0.41 | -47.23%
EWA-EWC | static | 1.89% | 0.11 | -9.30% | neural | -2.90% | -0.08 | -14.45%
NKE-AES | static | -64.40% | -0.69 | -69.47% | neural | 45.70% | 0.55 | -23.04%
MRK-VZ | static | 37.72% | 0.82 | -7.00% | neural | 40.23% | 0.75 | -10.38%
GS-CVX | static | 19.56% | 0.40 | -24.30% | neural | 32.39% | 0.41 | -43.86%
KO-PEP | static | -9.11% | -0.27 | -23.18% | neural | 3.92% | 0.15 | -10.18%
XOM-CVX | static | 20.54% | 0.59 | -9.83% | neural | 40.11% | 0.98 | -11.12%
GLD-SLV | static | -25.28% | -1.11 | -25.51% | neural | -27.54% | -1.26 | -27.71%
XLE-XOP | static | -14.52% | -0.43 | -20.35% | neural | -6.12% | -0.13 | -16.91%
XLE-OIH | static | 15.57% | 0.35 | -19.73% | neural | -5.76% | -0.12 | -13.57%
USO-XLE | static | -23.31% | -0.27 | -44.09% | neural | -15.46% | -0.10 | -29.86%
MS-GS | static | 16.37% | 0.50 | -9.37% | neural | -11.65% | -0.22 | -26.21%
V-MA | static | -8.93% | -0.46 | -12.33% | neural | -14.04% | -0.69 | -15.27%
HD-LOW | static | -21.42% | -0.52 | -26.74% | neural | 6.83% | 0.23 | -18.56%
"""

# Parse performance data
perf_data = {}
for line in res_lines.strip().split("\n"):
    parts = [p.strip() for p in line.split("|")]
    pair = parts[0]
    s_ret = float(parts[2].replace("%", ""))
    s_shp = float(parts[3])
    s_dd = float(parts[4].replace("%", ""))
    n_ret = float(parts[6].replace("%", ""))
    n_shp = float(parts[7])
    n_dd = float(parts[8].replace("%", ""))
    perf_data[pair] = {
        "static_ret": s_ret,
        "static_sharpe": s_shp,
        "static_dd": s_dd,
        "neural_ret": n_ret,
        "neural_sharpe": n_shp,
        "neural_dd": n_dd,
        # A pair is an OOS success if static is profitable OR neural is profitable
        "static_success": s_ret > 0 and s_shp > 0 and s_dd >= -25.0,
        "neural_success": n_ret > 0 and n_shp > 0,
        "any_success": (s_ret > 0 and s_shp > 0) or (n_ret > 0 and n_shp > 0),
    }

# Load screening metrics
scr_df = pd.read_csv("outputs/all_14_screening_metrics.csv")

# Merge
for k, v in perf_data.items():
    scr_df.loc[scr_df["pair"] == k, "static_ret"] = v["static_ret"]
    scr_df.loc[scr_df["pair"] == k, "static_sharpe"] = v["static_sharpe"]
    scr_df.loc[scr_df["pair"] == k, "static_success"] = v["static_success"]
    scr_df.loc[scr_df["pair"] == k, "neural_ret"] = v["neural_ret"]
    scr_df.loc[scr_df["pair"] == k, "neural_sharpe"] = v["neural_sharpe"]
    scr_df.loc[scr_df["pair"] == k, "neural_success"] = v["neural_success"]
    scr_df.loc[scr_df["pair"] == k, "any_success"] = v["any_success"]

print("=== SCREENING AND OUT-OF-SAMPLE PERFORMANCE DATA ===")
cols_show = ["pair", "r2_mr", "half_life", "rho", "adf_p_spread", "adf_p_M_dyn", "adf_p_M_stat", "static_ret", "static_sharpe", "neural_ret", "neural_sharpe"]
print(scr_df[cols_show].to_string(index=False))

# Evaluate Rule 1: Legacy Strict Screen
# pass_strict: r2_mr >= 0.15, half_life <= 252, adf_p_spread <= 0.10, rho < 0.999
scr_df["pred_strict"] = (
    (scr_df["r2_mr"] >= 0.15) &
    (scr_df["half_life"] <= 252) &
    (scr_df["adf_p_spread"] <= 0.10) &
    (scr_df["rho"] < 0.999)
)

# Evaluate Rule 2: Current PCI-Aware in regime_filter.py
# pass_pci_aware: r2_mr >= 0.10, half_life <= 365, (adf_p_spread <= 0.25 or adf_p_M_dyn <= 0.05), rho < 0.999
scr_df["pred_pci_current"] = (
    (scr_df["r2_mr"] >= 0.10) &
    (scr_df["half_life"] <= 365.0) &
    ((scr_df["adf_p_spread"] <= 0.25) | (scr_df["adf_p_M_dyn"] <= 0.05)) &
    (scr_df["rho"] < 0.999)
)

# Evaluate Rule 3: Static-Calibrated Rule (For Static-PCI)
# Testing stationarity on raw spread OR static M_t with strict half-life
scr_df["pred_static_calib"] = (
    (scr_df["r2_mr"] >= 0.50) &
    ((scr_df["adf_p_spread"] <= 0.05) | (scr_df["adf_p_M_stat"] <= 0.01))
)

def score_predictions(df, pred_col, target_col, name=""):
    tp = int(((df[pred_col] == True) & (df[target_col] == True)).sum())
    tn = int(((df[pred_col] == False) & (df[target_col] == False)).sum())
    fp = int(((df[pred_col] == True) & (df[target_col] == False)).sum())
    fn = int(((df[pred_col] == False) & (df[target_col] == True)).sum())
    total = len(df)
    correct = tp + tn
    acc = correct / total * 100
    print(f"\n--- {name} (Target: {target_col}) ---")
    print(f"Total Correct: {correct} / {total} ({acc:.1f}%)")
    print(f"True Positives (Passed & Profitable): {tp}")
    print(f"True Negatives (Rejected & Unprofitable): {tn}")
    print(f"False Positives (Passed but Lost money): {fp}")
    print(f"False Negatives (Rejected but Profitable): {fn}")
    for _, row in df.iterrows():
        p = row["pair"]
        pred = "PASS" if row[pred_col] else "REJ "
        actual = "WIN " if row[target_col] else "LOSS"
        status = "CORRECT" if row[pred_col] == row[target_col] else "WRONG"
        print(f"  {p:<8} | Pred: {pred} | Actual: {actual} | Status: {status}")
    return correct, acc

score_predictions(scr_df, "pred_strict", "static_success", "Legacy Strict Screen vs Static-PCI")
score_predictions(scr_df, "pred_pci_current", "static_success", "Current pass_pci_aware vs Static-PCI")
score_predictions(scr_df, "pred_pci_current", "neural_success", "Current pass_pci_aware vs Neural KalmanBOT")
score_predictions(scr_df, "pred_pci_current", "any_success", "Current pass_pci_aware vs Any Profitable Model")
