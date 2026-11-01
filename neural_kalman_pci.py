"""
neural_kalman_pci.py — NEURAL MODULE: GRU-Based Learned Kalman Gain (KalmanBOT)
================================================================================

Role: Defines the neural network architecture that REPLACES the analytic Kalman gain
      matrix K_t with a learned function. Also provides the differentiable Sharpe loss
      used for Stage 2 fine-tuning.

Architecture: KalmanNetGain
-------------------------------
Instead of computing K_t = P_pred @ H^T @ S^{-1} analytically, a GRU-based network
learns to map (innovation e_t, prior state x_pred) → Kalman gain K_t.

  Input:  [L2-normalized innovation || L2-normalized prior state]  (dim = obs_dim + state_dim)
  Layer 1: Linear → ReLU                                          (dim → H1)
  GRU:     Recurrent layer with hidden state h_t                   (H1 → hidden_dim)
  Layer 2: Linear → ReLU                                          (hidden_dim → H2)
  Layer 3: Linear (zero-initialized for conservative initial gain) (H2 → state_dim × obs_dim)
  Output:  Reshaped to (batch, state_dim, obs_dim) = K_t

The zero initialization of the output layer means the network starts with K_t ≈ 0
(i.e., "trust the prior"), and learns to correct it during training.

Training: Two-Stage Protocol
-----------------------------
  Stage 1 (MSE):    Minimize reconstruction error on X2 observations
  Stage 2 (Sharpe): Maximize differentiable Sharpe ratio of the implied trading strategy

DifferentiableSharpeLoss
-------------------------
Computes soft positions via tanh(z_score / threshold), applies turnover penalty,
and returns negative Sharpe for gradient descent.

Key Exports:
  - KalmanNetGain           : The GRU-based Kalman gain network (nn.Module)
  - DifferentiableSharpeLoss : Differentiable Sharpe ratio loss (nn.Module)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



class KalmanNetGain(nn.Module):
    def __init__(self, state_dim, obs_dim, hidden_dim=32):
        super(KalmanNetGain, self).__init__()
        self.m = state_dim
        self.n = obs_dim

        # Expose clean aliases so training code can use model.state_dim / model.obs_dim
        self.state_dim = state_dim
        self.obs_dim   = obs_dim

        # Configurable capacity for financial time-series (prevents severe overfitting)
        D_in = self.m + self.n
        H1 = min(48, (self.m + self.n) * 8)
        H2 = min(24, max(8, (self.m * self.n) * 2))
        D_out = self.m * self.n

        self.KG_l1 = nn.Linear(D_in, H1, bias=True)
        self.KG_relu1 = nn.ReLU()

        self.input_dim = H1
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        self.KG_l2 = nn.Linear(self.hidden_dim, H2, bias=True)
        self.KG_relu2 = nn.ReLU()

        self.KG_l3 = nn.Linear(H2, D_out, bias=True)
        # Small weight initialization so initial Kalman gain is conservative
        nn.init.zeros_(self.KG_l3.weight)
        nn.init.zeros_(self.KG_l3.bias)
        self.fc_out = self.KG_l3
        
    def forward(self, dm1y, dm1x, h_prev):
        # We assume dm1y and dm1x are already normalized by the caller, 
        # or we can normalize them here as per paper:
        # func.normalize(x, p=2, dim=-1, eps=1e-12)
        
        dm1y_norm = func.normalize(dm1y, p=2, dim=-1, eps=1e-12)
        dm1x_norm = func.normalize(dm1x, p=2, dim=-1, eps=1e-12)
        
        # Concat along features
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=-1)
        
        # Layer 1
        x = self.KG_l1(KGainNet_in)
        x = self.KG_relu1(x)
        
        # GRU expects sequence length dimension: (seq_len, batch, input_size)
        # We process step by step, so seq_len = 1
        # Input to GRU should be (1, batch, H1)
        x = x.unsqueeze(0)
        
        # h_prev should be (n_layers, batch, hidden_dim)
        if h_prev.dim() == 2:
            h_prev = h_prev.unsqueeze(0)
            
        x, h_next = self.rnn_GRU(x, h_prev.detach())
        
        # Remove seq_len dim
        x = x.squeeze(0)
        h_next = h_next.squeeze(0)
        
        # Layer 2
        x = self.KG_l2(x)
        x = self.KG_relu2(x)
        
        # Layer 3
        KG = self.KG_l3(x)
        
        # Reshape to (batch, m, n)
        batch_size = KG.shape[0]
        KG = KG.view(batch_size, self.m, self.n)
        
        return KG, h_next

class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0, turnover_penalty=0.001):
        super(DifferentiableSharpeLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.turnover_penalty = turnover_penalty
        
    def forward(self, spread_t, sigma_t, returns_t, z_threshold=0.3):
        z_scores = spread_t / (sigma_t + 1e-8)
        
        # Soft positions using tanh
        positions = torch.tanh(z_scores / z_threshold)
        
        # Strategy returns (batch, time)
        strategy_returns = positions * returns_t
        if self.turnover_penalty > 0 and positions.shape[1] > 1:
            diff_pos = torch.abs(positions[:, 1:] - positions[:, :-1])
            strategy_returns = strategy_returns.clone()
            strategy_returns[:, 1:] = strategy_returns[:, 1:] - self.turnover_penalty * diff_pos
        
        mean_return = torch.mean(strategy_returns, dim=1)
        std_return = torch.std(strategy_returns, dim=1) + 1e-8
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        # Maximize sharpe = minimize negative sharpe
        return -torch.mean(sharpe)


# ===========================================================================
# 4-State Neural KalmanNet Routines (Data Prep, Training, Inference)
# ===========================================================================

def prepare_tensors_4state(x1_train, x2_train, fitted_dyn):
    """Build PyTorch tensors for the 4-state model [X1, beta, M, R].

    State vector: z_t = [X1_t, beta_t, M_t, R_t].
    Observation : y_t = [X2_t, X1_t].
    Time-varying observation matrix H_t:
        Row 0 (X2): [0,  x1[t], 1, 1]  =>  X2 = 0*X1 + beta*x1[t] + M + R
        Row 1 (X1): [1,  0,     0, 0]  =>  X1 = X1
    """
    T_train = len(x1_train)
    y_train = np.stack([x2_train, x1_train], axis=1)  # obs order: [X2, X1]

    batch_y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(0)
    batch_H = torch.zeros((1, T_train, 2, 4), dtype=torch.float32)

    returns = np.zeros(T_train)
    for t in range(1, T_train):
        returns[t] = (
            (y_train[t, 0] - y_train[t - 1, 0])
            - fitted_dyn.p.beta0 * (y_train[t, 1] - y_train[t - 1, 1])
        )
    batch_returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)

    for t in range(T_train):
        x1t = float(x1_train[t])
        H_t = np.array([
            [0.0, x1t, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ])
        batch_H[0, t] = torch.tensor(H_t, dtype=torch.float32)

    return batch_y, batch_returns, batch_H


def train_neural_kalman_4state(model, dataloader, fitted_dyn,
                               epochs_stage1=15, epochs_stage2=15,
                               verbose=True):
    """Two-stage training for the 4-state model [X1, beta, M, R].

    Stage 1: minimise MSE on the X2 observation (spread leg reconstruction).
    Stage 2: fine-tune to maximise differentiable Sharpe ratio.
    """
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    sharpe_loss = DifferentiableSharpeLoss()

    rho = fitted_dyn.p.rho
    F_mat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, rho, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    def _init_state(batch_y, B):
        x_upd = torch.zeros(B, model.state_dim)
        x_upd[:, 0] = batch_y[:, 0, 1]                                          # X1_0
        x_upd[:, 1] = fitted_dyn.p.beta0                                       # beta_0
        x_upd[:, 2] = 0.0                                                      # M_0 = 0
        x_upd[:, 3] = batch_y[:, 0, 0] - fitted_dyn.p.beta0 * batch_y[:, 0, 1] # R_0
        return x_upd

    # Stage 1: MSE Pre-training
    if verbose:
        print("  Stage 1: MSE pre-training (X2 reconstruction)...")
    for epoch in range(epochs_stage1):
        total_loss = 0.0
        for batch_y, batch_returns, batch_H in dataloader:
            B, T, _ = batch_y.shape
            optimizer.zero_grad()
            h_prev = torch.zeros(B, model.hidden_dim)
            x_upd = _init_state(batch_y, B)

            loss = torch.tensor(0.0)
            for t in range(1, T):
                x_prior = torch.matmul(x_upd, F_mat.T)
                H_t = batch_H[:, t]
                y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
                e_t = batch_y[:, t] - y_prior
                loss = loss + mse_loss(y_prior[:, 0], batch_y[:, t, 0])
                K_t, h_prev = model(e_t, x_prior, h_prev)
                x_upd = torch.clamp(x_prior + torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1), -50.0, 50.0)

            (loss / T).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs_stage1}  MSE={total_loss:.4f}")

    # Stage 2: Sharpe Fine-Tuning
    if verbose:
        print("  Stage 2: Differentiable Sharpe fine-tuning...")
    for epoch in range(epochs_stage2):
        for batch_y, batch_returns, batch_H in dataloader:
            B, T, _ = batch_y.shape
            optimizer.zero_grad()
            h_prev = torch.zeros(B, model.hidden_dim)
            x_upd = _init_state(batch_y, B)

            e_t_traj = []
            for t in range(1, T):
                x_prior = torch.matmul(x_upd, F_mat.T)
                H_t = batch_H[:, t]
                y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
                e_t = batch_y[:, t] - y_prior
                K_t, h_prev = model(e_t, x_prior, h_prev)
                x_upd = torch.clamp(x_prior + torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1), -50.0, 50.0)
                e_t_traj.append(e_t[:, 0])

            e_t_traj = torch.stack(e_t_traj, dim=1)
            sigma = torch.std(e_t_traj, dim=1, keepdim=True).expand_as(e_t_traj)
            loss = sharpe_loss(e_t_traj, sigma, batch_returns[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs_stage2}")

    return model


def run_neural_filter_4state(model, x1_test, x2_test, fitted_dyn):
    """Run trained 4-state model out-of-sample; returns (beta_traj, M_traj)."""
    y_data = np.stack([x2_test, x1_test], axis=1)
    T = len(y_data)

    rho = fitted_dyn.p.rho
    F_mat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, rho, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    beta_traj, M_traj = [], []
    model.eval()
    with torch.no_grad():
        x_upd = torch.zeros(1, model.state_dim)
        x_upd[0, 0] = float(x1_test[0])
        x_upd[0, 1] = fitted_dyn.p.beta0
        x_upd[0, 2] = 0.0
        x_upd[0, 3] = float(x2_test[0]) - fitted_dyn.p.beta0 * float(x1_test[0])
        h_prev = torch.zeros(1, model.hidden_dim)
        batch_y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(0)

        for t in range(1, T):
            x_prior = torch.matmul(x_upd, F_mat.T)
            x1t = float(x1_test[t])
            H_t = torch.tensor([[[0.0, x1t, 1.0, 1.0],
                                  [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            e_t = batch_y[:, t] - y_prior
            K_t, h_prev = model(e_t, x_prior, h_prev)
            x_upd = torch.clamp(x_prior + torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1), -50.0, 50.0)

            beta_traj.append(x_upd[0, 1].item())
            M_traj.append(x_upd[0, 2].item())

    return (
        np.array([fitted_dyn.p.beta0] + beta_traj),
        np.array([0.0] + M_traj),
    )
