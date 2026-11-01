import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
from data_loader import fetch_pair_data
from dynamic_pci_3state import fit_pci3
from neural_kalman_pci import KalmanNetGain, DifferentiableSharpeLoss

for t1, t2 in [('KO', 'PEP'), ('MRK', 'VZ')]:
    print(f"\n=================== {t1}-{t2} ===================", flush=True)
    data = fetch_pair_data(t1, t2, formation_start='2009-09-01', formation_end='2017-08-30', trading_start='2017-08-31', trading_end='2021-08-31')
    x1_form = data.x1_formation / data.x1_formation[0]
    x2_form = data.x2_formation / data.x2_formation[0]
    p_dyn, _ = fit_pci3(x1_form, x2_form, allow_dynamic_beta=True, verbose=False)
    
    T = len(x1_form)
    batch_y = torch.tensor(np.column_stack([x1_form, x2_form]), dtype=torch.float32).unsqueeze(0)
    batch_ret = torch.tensor(np.column_stack([np.diff(x1_form)/x1_form[:-1], np.diff(x2_form)/x2_form[:-1]]), dtype=torch.float32).unsqueeze(0)
    
    t_idx = np.arange(T, dtype=np.float32)
    H_mat = np.zeros((T, 1, 3), dtype=np.float32)
    H_mat[:, 0, 0] = x1_form
    H_mat[:, 0, 1] = t_idx
    H_mat[:, 0, 2] = 1.0
    batch_H = torch.tensor(H_mat).unsqueeze(0)
    
    F_mat = torch.eye(3, dtype=torch.float32)
    F_mat[2, 2] = float(p_dyn.rho)
    
    model = KalmanNetGain(state_dim=3, obs_dim=1, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = torch.nn.MSELoss()
    sharpe_loss = DifferentiableSharpeLoss()
    
    print("--- Stage 1 (MSE Pre-training) ---", flush=True)
    for ep in range(10):
        optimizer.zero_grad()
        h_prev = torch.zeros(1, 32)
        x_upd = torch.tensor([[float(p_dyn.beta0), 0.0, float(x2_form[0] - p_dyn.beta0*x1_form[0])]], dtype=torch.float32)
        loss = torch.tensor(0.0)
        max_state_val = 0.0
        clamped_count = 0
        
        for t in range(1, T):
            x_prior = torch.matmul(x_upd, F_mat.T)
            H_t = batch_H[:, t]
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            e_t = batch_y[:, t, 1:2] - y_prior
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_raw = x_prior + corr
            
            val = x_raw.detach().abs().max().item()
            if val > max_state_val:
                max_state_val = val
            if val >= 100.0:
                clamped_count += 1
                
            x_upd = torch.clamp(x_raw, -100.0, 100.0)
            loss = loss + mse_loss(y_prior[:, 0], batch_y[:, t, 1])
            
        (loss / T).backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (ep + 1) in [1, 2, 5, 10]:
            print(f"  Ep {ep+1:2d} | Raw Grad Norm: {total_norm:7.4f} (clipped at 1.0? {total_norm > 1.0}) | Max |x_upd|: {max_state_val:6.3f} (Clamped steps: {clamped_count}/{T}) | MSE: {loss.item()/T:.6f}", flush=True)

    print("--- Stage 2 (Differentiable Sharpe Fine-tuning) ---", flush=True)
    for ep in range(10):
        optimizer.zero_grad()
        h_prev = torch.zeros(1, 32)
        x_upd = torch.tensor([[float(p_dyn.beta0), 0.0, float(x2_form[0] - p_dyn.beta0*x1_form[0])]], dtype=torch.float32)
        e_traj = []
        max_state_val = 0.0
        clamped_count = 0
        
        for t in range(1, T):
            x_prior = torch.matmul(x_upd, F_mat.T)
            H_t = batch_H[:, t]
            y_prior = torch.bmm(H_t, x_prior.unsqueeze(-1)).squeeze(-1)
            e_t = batch_y[:, t, 1:2] - y_prior
            K_t, h_prev = model(e_t, x_prior, h_prev)
            corr = torch.bmm(K_t, e_t.unsqueeze(-1)).squeeze(-1)
            x_raw = x_prior + corr
            
            val = x_raw.detach().abs().max().item()
            if val > max_state_val:
                max_state_val = val
            if val >= 100.0:
                clamped_count += 1
                
            x_upd = torch.clamp(x_raw, -100.0, 100.0)
            e_traj.append(e_t[:, 0])
            
        e_traj = torch.stack(e_traj, dim=1)
        sigma = torch.std(e_traj, dim=1, keepdim=True).expand_as(e_traj)
        sl = sharpe_loss(e_traj, sigma, batch_ret[:, 1:])
        sl.backward()
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (ep + 1) in [1, 2, 5, 10]:
            print(f"  Ep {ep+1:2d} | Raw Grad Norm: {total_norm:7.4f} (clipped at 1.0? {total_norm > 1.0}) | Max |x_upd|: {max_state_val:6.3f} (Clamped steps: {clamped_count}/{T}) | neg-Sharpe: {sl.item():.4f}", flush=True)
