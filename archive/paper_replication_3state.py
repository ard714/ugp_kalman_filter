"""
paper_replication_3state.py — ARCHIVED: 3-State IEEE Paper Replication Pipeline
================================================================================

Role: Preserved historical baseline for replicating the original 3-state KalmanBOT
      IEEE paper results on DataSet/*.csv files.
      
State Formulation (3-State):
  z_t = [beta_t, mu_t, m_t]
  Observation: X2_t = beta_t * X1_t + mu_t + m_t

Note: This formulation is preserved for academic replication only. The active production
      pipeline uses the 4-state Dynamic-PCI model in `main.py` and `dynamic_pci.py`.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dynamic_pci import fit_dynamic_pci, DynamicPCIKalmanFilter
from backtest import backtest_pair, performance_summary
from neural_kalman_pci import KalmanNetGain, DifferentiableSharpeLoss

PAPER_TRAIN_SIZE = 2000
PAPER_TEST_SIZE = 1500

PAPER_PAIRS = [
    ("AAP", "IBM"),
    ("EWA", "EWC"),
    ("GS",  "CVX"),
    ("KO",  "PEP"),
    ("MRK", "VZ"),
    ("NKE", "AES"),
    ("XOM", "CVX"),
]


def fetch_data_paper(t1, t2):
    """Load and normalise a pair CSV from DataSet/ (paper mode uses raw Open prices)."""
    pair_name = f"{t1}_{t2}"
    file_path = os.path.join("DataSet", f"{pair_name}.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path, index_col="Date")
    df = df.dropna(subset=[f"{t1} Open", f"{t2} Open"])
    df_out = df[[f"{t1} Open", f"{t2} Open"]].copy()
    df_out.columns = [t1, t2]
    return df_out / df_out.iloc[0]


def prepare_tensors_3state(df_train, fitted_dyn):
    """Build tensors for 3-state model [beta, mu, m] (paper architecture)."""
    y_train = df_train.values
    T_train = len(y_train)
    batch_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(0)

    batch_H = torch.zeros((1, T_train, 1, 3), dtype=torch.float32)
    batch_S = torch.zeros((1, T_train, 1, 1), dtype=torch.float32)

    returns = np.zeros(T_train)
    for t in range(1, T_train):
        returns[t] = (
            (y_train[t, 0] - y_train[t - 1, 0])
            - fitted_dyn.p.beta0 * (y_train[t, 1] - y_train[t - 1, 1])
        )
    batch_returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)

    for t in range(T_train):
        batch_H[0, t] = torch.tensor([[y_train[t, 0], 1.0, 1.0]], dtype=torch.float32)
        batch_S[0, t] = torch.eye(1) * 0.01

    return batch_y, batch_S, batch_returns, batch_H


def train_neural_kalman_3state(model, dataloader, fitted_dyn, epochs_stage1=20, epochs_stage2=15):
    """Two-stage training for the 3-state paper model."""
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    sharpe_loss = DifferentiableSharpeLoss()

    F_mat = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, fitted_dyn.p.rho]],
        dtype=torch.float32,
    )

    # --- Stage 1 ---
    print("  Stage 1: MSE pre-training...")
    for epoch in range(epochs_stage1):
        for batch_y, batch_S, batch_returns, batch_H in dataloader:
            B, T, _ = batch_y.shape
            optimizer.zero_grad()
            h_prev = torch.zeros(1, B, model.hidden_dim)
            correction = torch.zeros(B, model.state_dim)
            x_upd = torch.zeros(B, model.state_dim)
            x_upd[:, 0] = fitted_dyn.p.beta0
            x_upd[:, 1] = batch_y[:, 0, 1] - fitted_dyn.p.beta0 * batch_y[:, 0, 0]

            loss = 0
            for t in range(1, T):
                x_prior = torch.matmul(x_upd, F_mat.T)
                H_t = batch_H[:, t]
                y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
                dm1y = batch_y[:, t, 1:2] - y_prior
                loss += mse_loss(y_prior, batch_y[:, t, 1:2])
                K_t, h_prev = model(dm1y, correction, h_prev)
                correction = torch.bmm(K_t, dm1y.unsqueeze(-1)).squeeze(-1)
                x_upd = torch.clamp(x_prior + correction, -100.0, 100.0)

            (loss / T).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs_stage1}, loss={loss.item():.6f}")

    # --- Stage 2 ---
    print("  Stage 2: Sharpe fine-tuning...")
    for epoch in range(epochs_stage2):
        for batch_y, batch_S, batch_returns, batch_H in dataloader:
            B, T, _ = batch_y.shape
            optimizer.zero_grad()
            h_prev = torch.zeros(1, B, model.hidden_dim)
            correction = torch.zeros(B, model.state_dim)
            x_upd = torch.zeros(B, model.state_dim)
            x_upd[:, 0] = fitted_dyn.p.beta0
            x_upd[:, 1] = batch_y[:, 0, 1] - fitted_dyn.p.beta0 * batch_y[:, 0, 0]

            e_t_traj = []
            for t in range(1, T):
                x_prior = torch.matmul(x_upd, F_mat.T)
                H_t = batch_H[:, t]
                y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
                dm1y = batch_y[:, t, 1:2] - y_prior
                K_t, h_prev = model(dm1y, correction, h_prev)
                correction = torch.bmm(K_t, dm1y.unsqueeze(-1)).squeeze(-1)
                x_upd = torch.clamp(x_prior + correction, -100.0, 100.0)
                e_t_traj.append(dm1y[:, 0])

            e_t_traj = torch.stack(e_t_traj, dim=1)
            sigma = torch.std(e_t_traj, dim=1, keepdim=True).expand_as(e_t_traj)
            loss = sharpe_loss(e_t_traj, sigma, batch_returns[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs_stage2}, neg-Sharpe={loss.item():.4f}")

    return model


def run_neural_filter_3state(model, y_data, fitted_dyn):
    """Run trained 3-state model out-of-sample; returns (beta_traj, e_t_traj)."""
    T = len(y_data)
    F_mat = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, fitted_dyn.p.rho]],
        dtype=torch.float32,
    )
    beta_traj, e_t_traj = [], []
    model.eval()
    with torch.no_grad():
        x_upd = torch.zeros(1, model.state_dim)
        x_upd[0, 0] = fitted_dyn.p.beta0
        x_upd[0, 1] = y_data[0, 1] - fitted_dyn.p.beta0 * y_data[0, 0]
        h_prev = torch.zeros(1, 1, model.hidden_dim)
        correction = torch.zeros(1, model.state_dim)
        batch_y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(0)

        for t in range(1, T):
            x_prior = torch.matmul(x_upd, F_mat.T)
            H_t = torch.tensor([[[y_data[t, 0], 1.0, 1.0]]], dtype=torch.float32)
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            dm1y = batch_y[:, t, 1:2] - y_prior
            K_t, h_prev = model(dm1y, correction, h_prev)
            correction = torch.bmm(K_t, dm1y.unsqueeze(-1)).squeeze(-1)
            x_upd = torch.clamp(x_prior + correction, -100.0, 100.0)
            beta_traj.append(x_upd[0, 0].item())
            e_t_traj.append(dm1y[0, 0].item())

    return (
        np.array([fitted_dyn.p.beta0] + beta_traj),
        np.array([0.0] + e_t_traj),
    )


def run_paper_mode(out_txt="outputs/results_paper.txt"):
    """3-state paper-replication: classical vs neural on PAPER_PAIRS."""
    results = []
    for t1, t2 in PAPER_PAIRS:
        print(f"\n{'='*16} PAIR: {t1} vs {t2} {'='*16}")
        df = fetch_data_paper(t1, t2)
        if df.empty:
            continue

        train_end = min(PAPER_TRAIN_SIZE, len(df))
        test_end = min(train_end + PAPER_TEST_SIZE, len(df))
        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:test_end]

        x1_train, x2_train = df_train.values[:, 0], df_train.values[:, 1]
        x1_test, x2_test = df_test.values[:, 0], df_test.values[:, 1]

        dyn_params, _ = fit_dynamic_pci(x1_train, x2_train)
        dyn_pci = DynamicPCIKalmanFilter(dyn_params)

        print("  Preparing Neural Dataloaders...")
        tensors = prepare_tensors_3state(df_train, dyn_pci)
        model = KalmanNetGain(state_dim=3, obs_dim=1, hidden_dim=32)
        print("  Training...")
        model = train_neural_kalman_3state(model, [tensors], dyn_pci)

        out_classic = dyn_pci.filter(x1_test, x2_test)
        beta_classic = np.array(out_classic["beta"]).flatten()
        M_classic = np.array(out_classic["M"]).flatten()
        beta_neural, e_t_neural = run_neural_filter_3state(model, df_test.values, dyn_pci)

        df_classic_bt, _ = backtest_pair(x1_test, x2_test, beta_classic, M_classic, z_window=80)
        df_neural_bt, _ = backtest_pair(x1_test, x2_test, beta_neural.flatten(), e_t_neural.flatten(), z_window=80)

        results.append({
            "pair": f"{t1}-{t2}",
            "classic": performance_summary(df_classic_bt),
            "neural": performance_summary(df_neural_bt),
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_txt)), exist_ok=True)
    header = f"{'Pair':<12} | {'Model':<15} | {'Tot Return':<12} | {'Ann Return':<12} | {'Sharpe':<10} | {'Max DD':<10}"
    sep = "-" * len(header)
    lines = ["3-STATE IEEE PAPER REPLICATION REPORT", "=" * len(header), header, sep]
    for r in results:
        pair = r["pair"]
        for m_name in ["classic", "neural"]:
            perf = r[m_name]
            tot_ret = f"{perf['total_return']*100:.2f}%" if not pd.isna(perf["total_return"]) else "NaN"
            ann_ret = f"{perf['ann_return']*100:.2f}%" if not pd.isna(perf["ann_return"]) else "NaN"
            sharpe = f"{perf['sharpe']:.2f}" if not pd.isna(perf["sharpe"]) else "NaN"
            max_dd = f"{perf['max_drawdown']*100:.2f}%" if not pd.isna(perf["max_drawdown"]) else "NaN"
            lines.append(f"{pair:<12} | {m_name:<15} | {tot_ret:<12} | {ann_ret:<12} | {sharpe:<10} | {max_dd:<10}")
        lines.append(sep)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(out_txt, "w") as f:
        f.write(report_text)
    print(f"\nSaved paper replication results to: {out_txt}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3-State IEEE Paper Replication")
    parser.add_argument("--out", default="outputs/results_paper.txt", help="Output text file path")
    args = parser.parse_args()
    run_paper_mode(out_txt=args.out)
