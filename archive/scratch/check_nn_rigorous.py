import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import numpy as np
from data_loader import fetch_pair_data
from dynamic_pci_3state import fit_pci3
from neural_kalman_pci import KalmanNetGain, DifferentiableSharpeLoss

def _build_F(rho: float) -> torch.Tensor:
    return torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, rho, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

pairs_to_test = [("KO", "PEP"), ("AAP", "IBM"), ("MRK", "VZ")]

for t1, t2 in pairs_to_test:
    print(f"\n{'='*25} DIAGNOSTIC: {t1}-{t2} {'='*25}", flush=True)
    p = fetch_pair_data(t1, t2)
    x1_form = p.x1_formation / p.x1_formation[0]
    x2_form = p.x2_formation / p.x2_formation[0]
    x1_test = p.x1_trading / p.x1_trading[0]
    x2_test = p.x2_trading / p.x2_trading[0]

    params, _ = fit_pci3(x1_form, x2_form, allow_dynamic_beta=True, verbose=False)
    F_mat = _build_F(params.rho)

    T_form = len(x1_form)
    batch_y = torch.tensor(np.stack([x1_form, x2_form], axis=1), dtype=torch.float32).unsqueeze(0)
    batch_H = torch.zeros(1, T_form, 1, 3, dtype=torch.float32)
    for t in range(T_form):
        batch_H[0, t, 0] = torch.tensor([x1_form[t], 1.0, 1.0])

    returns = np.zeros(T_form)
    for t in range(1, T_form):
        returns[t] = ((x2_form[t] - x2_form[t - 1]) - params.beta0 * (x1_form[t] - x1_form[t - 1]))
    batch_ret = torch.tensor(returns, dtype=torch.float32).unsqueeze(0)

    model = KalmanNetGain(state_dim=3, obs_dim=1, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    sharpe_loss = DifferentiableSharpeLoss()

    def _init_state(B):
        r0 = float(x2_form[0]) - params.beta0 * float(x1_form[0])
        z = torch.zeros(B, 3)
        z[:, 0] = params.beta0
        z[:, 1] = 0.0
        z[:, 2] = r0
        return z

    print("\n--- STAGE 1: MSE Pre-training (10 Epochs) ---", flush=True)
    for ep in range(10):
        optimizer.zero_grad()
        h_prev = torch.zeros(1, model.hidden_dim)
        x_upd = _init_state(1)
        loss = torch.tensor(0.0)
        
        max_abs_x = 0.0
        clamped_count = 0
        
        for t in range(1, T_form):
            x_prior = torch.matmul(x_upd, F_mat.T)
            H_t = batch_H[:, t]
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            e_t = batch_y[:, t, 1:2] - y_prior
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_raw = x_prior + corr
            
            abs_val = x_raw.detach().abs().max().item()
            if abs_val > max_abs_x:
                max_abs_x = abs_val
            if abs_val >= 100.0:
                clamped_count += 1
                
            x_upd = torch.clamp(x_raw, -100.0, 100.0)
            loss = loss + mse_loss(y_prior[:, 0], batch_y[:, t, 1])

        (loss / T_form).backward()

        # Compute raw gradient norm BEFORE clipping
        raw_grad_norm = 0.0
        for p_param in model.parameters():
            if p_param.grad is not None:
                raw_grad_norm += p_param.grad.data.norm(2).item() ** 2
        raw_grad_norm = raw_grad_norm ** 0.5

        # Clip grad norm
        clipped = raw_grad_norm > 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"  Ep {ep+1:2d}/10 | Raw Grad Norm: {raw_grad_norm:10.4f} (Hit 1.0 Max? {str(clipped):<5}) | "
              f"Max |x_upd|: {max_abs_x:7.4f} (Clamped: {clamped_count:4d}/{T_form-1} = {clamped_count/(T_form-1)*100:4.1f}%) | "
              f"MSE: {loss.item()/T_form:.6f}", flush=True)

    print("\n--- STAGE 2: Differentiable Sharpe Fine-Tuning (10 Epochs) ---", flush=True)
    for ep in range(10):
        optimizer.zero_grad()
        h_prev = torch.zeros(1, model.hidden_dim)
        x_upd = _init_state(1)
        e_traj = []
        max_abs_x = 0.0
        clamped_count = 0

        for t in range(1, T_form):
            x_prior = torch.matmul(x_upd, F_mat.T)
            H_t = batch_H[:, t]
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            e_t = batch_y[:, t, 1:2] - y_prior
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_raw = x_prior + corr

            abs_val = x_raw.detach().abs().max().item()
            if abs_val > max_abs_x:
                max_abs_x = abs_val
            if abs_val >= 100.0:
                clamped_count += 1

            x_upd = torch.clamp(x_raw, -100.0, 100.0)
            e_traj.append(e_t[:, 0])

        e_traj = torch.stack(e_traj, dim=1)
        sigma = torch.std(e_traj, dim=1, keepdim=True).expand_as(e_traj)
        sl = sharpe_loss(e_traj, sigma, batch_ret[:, 1:])
        sl.backward()

        raw_grad_norm = 0.0
        for p_param in model.parameters():
            if p_param.grad is not None:
                raw_grad_norm += p_param.grad.data.norm(2).item() ** 2
        raw_grad_norm = raw_grad_norm ** 0.5

        clipped = raw_grad_norm > 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"  Ep {ep+1:2d}/10 | Raw Grad Norm: {raw_grad_norm:10.4f} (Hit 1.0 Max? {str(clipped):<5}) | "
              f"Max |x_upd|: {max_abs_x:7.4f} (Clamped: {clamped_count:4d}/{T_form-1} = {clamped_count/(T_form-1)*100:4.1f}%) | "
              f"neg-Sharpe: {sl.item():.4f}", flush=True)

    # ---- Out-of-Sample Trading Pass ----
    print("\n--- OUT-OF-SAMPLE TEST PASS (1,007 Trading Days) ---", flush=True)
    T_trade = len(x1_test)
    model.eval()
    with torch.no_grad():
        r0 = float(x2_test[0]) - params.beta0 * float(x1_test[0])
        x_upd = torch.zeros(1, 3)
        x_upd[0, 0] = params.beta0
        x_upd[0, 1] = 0.0
        x_upd[0, 2] = r0
        h_prev = torch.zeros(1, model.hidden_dim)

        max_trade_x = 0.0
        trade_clamped = 0

        for t in range(1, T_trade):
            x_prior = torch.matmul(x_upd, F_mat.T)
            x1t = float(x1_test[t])
            H_t = torch.tensor([[[x1t, 1.0, 1.0]]], dtype=torch.float32)
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            e_t = torch.tensor([[float(x2_test[t]) - float(y_prior[0, 0])]], dtype=torch.float32)
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_raw = x_prior + corr

            abs_val = x_raw.abs().max().item()
            if abs_val > max_trade_x:
                max_trade_x = abs_val
            if abs_val >= 100.0:
                trade_clamped += 1

            x_upd = torch.clamp(x_raw, -100.0, 100.0)

        print(f"  Out-of-Sample Result: Max |x_upd| = {max_trade_x:.4f} | Clamped Steps: {trade_clamped}/{T_trade-1} ({trade_clamped/(T_trade-1)*100:.2f}%)", flush=True)
