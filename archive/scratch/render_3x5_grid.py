import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_loader import (
    fetch_all_target_pairs,
    ALL_PAIRS,
    DEFAULT_FORMATION_START,
    DEFAULT_FORMATION_END,
    DEFAULT_TRADING_START,
    DEFAULT_TRADING_END,
)
from dynamic_pci import fit_dynamic_pci, DynamicPCIKalmanFilter
from backtest import backtest_pair
from regime_filter import compute_online_regime_mask
from neural_kalman_pci import (
    KalmanNetGain,
    prepare_tensors_4state,
    train_neural_kalman_4state,
    run_neural_filter_4state,
)
from plot_utils import plot_multi_pair_equity_grid

print("Fetching data for all 14 pairs...")
all_data = fetch_all_target_pairs(
    pairs=ALL_PAIRS,
    formation_start=DEFAULT_FORMATION_START,
    formation_end=DEFAULT_FORMATION_END,
    trading_start=DEFAULT_TRADING_START,
    trading_end=DEFAULT_TRADING_END,
)

all_equity_data = {}

for idx, ((t1, t2), pdata) in enumerate(all_data.items()):
    p_name = f"{t1}-{t2}"
    print(f"[{idx+1}/14] Processing equity curves for {p_name}...", flush=True)

    s1, s2 = float(pdata.x1_all[0]), float(pdata.x2_all[0])
    x1_norm = pdata.x1_all / s1
    x2_norm = pdata.x2_all / s2
    form_n = len(pdata.x1_formation)

    x1_form = x1_norm[:form_n]
    x2_form = x2_norm[:form_n]
    x1_test = x1_norm[form_n:]
    x2_test = x2_norm[form_n:]
    dates_trade = pdata.dates_trading

    dyn_params, _ = fit_dynamic_pci(x1_form, x2_form, allow_dynamic_beta=True, maxiter=30)
    static_params, _ = fit_dynamic_pci(x1_form, x2_form, allow_dynamic_beta=False, maxiter=30)

    dyn_pci = DynamicPCIKalmanFilter(dyn_params)
    static_pci = DynamicPCIKalmanFilter(static_params)

    out_dyn = dyn_pci.filter(x1_test, x2_test)
    out_static = static_pci.filter(x1_test, x2_test)
    regime_mask = compute_online_regime_mask(out_dyn["M"], rolling_window=60)

    # Neural filter (fast 2+2 epoch setup for visualization)
    batch_y, batch_returns, batch_H = prepare_tensors_4state(x1_form, x2_form, dyn_pci)
    dataset = TensorDataset(batch_y, batch_returns, batch_H)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = KalmanNetGain(state_dim=4, obs_dim=2, hidden_dim=32)
    model = train_neural_kalman_4state(
        model, dataloader, dyn_pci, epochs_stage1=2, epochs_stage2=2, verbose=False
    )
    beta_neural, M_neural = run_neural_filter_4state(model, x1_test, x2_test, dyn_pci)

    df_static, _ = backtest_pair(x1_test, x2_test, out_static["beta"], out_static["M"], z_window=80)
    df_dyn, _    = backtest_pair(x1_test, x2_test, out_dyn["beta"],    out_dyn["M"],    z_window=80)
    df_gated, _  = backtest_pair(x1_test, x2_test, out_dyn["beta"],    out_dyn["M"],    z_window=80, regime_mask=regime_mask)
    df_neural, _ = backtest_pair(x1_test, x2_test, beta_neural,        M_neural,        z_window=80)

    all_equity_data[p_name] = {
        "static": pd.Series(df_static["equity"].values, index=dates_trade),
        "dyn":    pd.Series(df_dyn["equity"].values, index=dates_trade),
        "gated":  pd.Series(df_gated["equity"].values, index=dates_trade),
        "neural": pd.Series(df_neural["equity"].values, index=dates_trade),
    }

print("\nRendering 3x5 all pairs equity grid plot...")
out_path = os.path.join("outputs", "all_pairs_equity_grid.png")
plot_multi_pair_equity_grid(all_equity_data, save_path=out_path)
print(f"SUCCESS: Rendered 3x5 equity grid to {out_path}")
