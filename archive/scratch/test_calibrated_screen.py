import pandas as pd
import numpy as np

scr_df = pd.read_csv("outputs/all_14_screening_metrics.csv")

# Add OOS success flags from evaluate_screening_rules
scr_df["static_success"] = scr_df["pair"].isin(["EWA-EWC", "MRK-VZ", "GS-CVX", "XOM-CVX", "XLE-OIH", "MS-GS"])
scr_df["neural_success"] = scr_df["pair"].isin(["NKE-AES", "MRK-VZ", "GS-CVX", "KO-PEP", "XOM-CVX", "HD-LOW"])
scr_df["any_success"] = scr_df["static_success"] | scr_df["neural_success"]

# Calibration 1: Model-Specific Screen for Static-PCI
# A pair is suitable for Static-PCI if:
# 1. R2_MR >= 0.50 (at least 50% of variance is mean-reverting)
# 2. ADF on static spread <= 0.05 OR ADF on static M_t <= 0.01
# 3. Correlation between legs is healthy (or half-life finite)
scr_df["pred_calibrated_static"] = (
    (scr_df["r2_mr"] >= 0.50) &
    ((scr_df["adf_p_spread_static"] <= 0.05) | (scr_df["adf_p_M_stat"] <= 0.01))
)

# Calibration 2: Model-Specific Screen for Neural KalmanNet
# Neural KalmanNet can handle beta-drift and low R2_MR, but needs stationary M_t:
# ADF on dynamic M_t <= 0.05 AND ADF on static M_t <= 0.01 (or R2_MR >= 0.20)
# and NOT permanent non-cointegrated drift like GLD-SLV or AAP-IBM
scr_df["pred_calibrated_neural"] = (
    (scr_df["adf_p_M_stat"] <= 0.01) |
    ((scr_df["adf_p_M_dyn"] <= 1e-5) & (scr_df["r2_mr"] >= 0.20) & (scr_df["half_life"] <= 30.0))
)

print("=== CALIBRATED SCREENING RULES EVALUATION ===")

def print_eval(df, pred_col, target_col, title):
    tp = int(((df[pred_col] == True) & (df[target_col] == True)).sum())
    tn = int(((df[pred_col] == False) & (df[target_col] == False)).sum())
    fp = int(((df[pred_col] == True) & (df[target_col] == False)).sum())
    fn = int(((df[pred_col] == False) & (df[target_col] == True)).sum())
    correct = tp + tn
    total = len(df)
    acc = correct / total * 100
    print(f"\n{title}")
    print(f"Correct: {correct} / {total} ({acc:.1f}%) | TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    for _, row in df.iterrows():
        p = row["pair"]
        pred = "PASS" if row[pred_col] else "REJ "
        actual = "WIN " if row[target_col] else "LOSS"
        status = "CORRECT" if row[pred_col] == row[target_col] else "WRONG"
        print(f"  {p:<8} | Pred: {pred} | Actual: {actual} | Status: {status}")

print_eval(scr_df, "pred_calibrated_static", "static_success", "CALIBRATED STATIC SCREEN vs Static-PCI")
print_eval(scr_df, "pred_calibrated_neural", "neural_success", "CALIBRATED NEURAL SCREEN vs Neural KalmanBOT")
