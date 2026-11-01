"""
dynamic_pci.py — CORE MODEL: 4-State Dynamic Partial Cointegration Kalman Filter
=================================================================================

Role: Core mathematical engine. Defines the state-space model, Kalman filter,
      and Maximum Likelihood Estimation (MLE) for parameter fitting.
      This module is imported by nearly every other file in the project.

Combines two ideas from the academic literature:

  1. Yu (2022) "Cointegration Approach for the Pair Trading based on the
     Kalman Filter": the hedge ratio β is not fixed over a formation
     period, but evolves as a random walk and is tracked in real time
     with a Kalman filter. The process noise variance q_β determines how
     quickly the hedge ratio is allowed to adapt.

  2. Clegg & Krauss (2018) "Pairs trading with partial cointegration":
     the spread between two cointegrated assets is modeled as the sum of
     a mean-reverting AR(1) component M_t and a random-walk component R_t.
     The R²_MR metric quantifies what fraction of spread variance is
     actually mean-reverting (and thus tradeable).

Dynamic-PCI model (4-State Formulation)
----------------------------------------

State vector: z_t = [X1_t, β_t, M_t, R_t]^T

Observation equation (2D — couples X2 and X1):
    y_t = [X2_t, X1_t]^T = H_t · z_t

    H_t = [[0,  X1_t,  1,  1],     ← X2_t = β_t·X1_t + M_t + R_t  (partial cointegration)
            [1,    0,   0,  0]]     ← X1_t = X1_t                   (driver price identity)

State transitions:
    X1_t   = X1_{t-1} + w_X_t,       w_X_t    ~ N(0, σ_X²)      Random walk (driver asset)
    β_t    = β_{t-1}  + w_β_t,       w_β_t    ~ N(0, q_β)        Random walk (hedge ratio)
    M_t    = ρ · M_{t-1} + w_M_t,    w_M_t    ~ N(0, σ_M²)      AR(1) mean-reverting spread
    R_t    = R_{t-1}  + w_R_t,       w_R_t    ~ N(0, σ_R²)      Random walk (permanent spread)

Transition matrix F (4×4):
    F = [[1, 0,  0,  0],
         [0, 1,  0,  0],
         [0, 0,  ρ,  0],
         [0, 0,  0,  1]]

Process noise covariance Q (4×4):
    Q = diag([σ_X², q_β, σ_M², σ_R²])

Key Exported Classes & Functions:
  - DynamicPCIParams     : Dataclass holding all model parameters + derived metrics (R²_MR, half-life)
  - DynamicPCIKalmanFilter : The 4-state Kalman filter (predict-update loop)
  - fit_dynamic_pci()    : MLE parameter estimation via L-BFGS-B optimization
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class DynamicPCIParams:
    beta0: float           # initial hedge ratio
    rho: float             # AR(1) coefficient of mean-reverting component M
    sigma_M: float         # std dev of innovations to M
    sigma_R: float         # std dev of innovations to R (random-walk component)
    q_beta: float          # process noise variance of beta's random walk
    sigma_X: float = None  # std dev of innovations to X1 (driver asset process)
    p_beta0: float = 1.0   # initial variance of beta (uncertainty at t=0)

    @property
    def r2_mr(self) -> float:
        """Proportion of variance attributable to mean-reversion, per
        Clegg & Krauss eq. (2): R^2_MR = 2*sigma_M^2 / (2*sigma_M^2 + (1+rho)*sigma_R^2)
        """
        num = 2 * self.sigma_M ** 2
        den = num + (1 + self.rho) * self.sigma_R ** 2
        return num / den if den > 0 else 0.0

    @property
    def half_life(self) -> float:
        """Half-life of mean reversion, in periods."""
        if not (-1 < self.rho < 1) or self.rho <= 0:
            return np.inf
        return np.log(0.5) / np.log(self.rho)


class DynamicPCIKalmanFilter:
    """Joint 4-state Kalman filter for the Dynamic-PCI state-space model."""

    def __init__(self, params: DynamicPCIParams):
        self.p = params

    def _matrices(self, sigma_X: float = 1.0):
        p = self.p
        # 4x4 State Transition Matrix: state = [X1_t, beta_t, M_t, R_t]^T
        F = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, p.rho, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # 4x4 Process Noise Covariance
        Q = np.diag([sigma_X ** 2, p.q_beta, p.sigma_M ** 2, p.sigma_R ** 2])
        return F, Q

    def filter(self, x1: np.ndarray, x2: np.ndarray):
        """
        Run the 4-state Kalman filter forward through the data.

        Parameters
        ----------
        x1, x2 : 1-D arrays of the two price series (same length, aligned).

        Returns
        -------
        dict with keys:
            x1, beta, M, R            : filtered posterior state estimates, shape (n,)
            x1_pred, beta_pred, ...   : one-step-ahead predictions, shape (n,)
            innovation                 : one-step-ahead spread prediction error e_t
            innovation_var             : variance of the spread innovation S_t
            loglik                     : total log-likelihood (excluding t=0)
            P_final                    : filtered state covariance at final step (4x4)
            F, Q                       : 4x4 system transition and noise covariance
        """
        n = len(x1)
        sigma_X = self.p.sigma_X if self.p.sigma_X is not None else float(np.std(np.diff(x1)))
        if sigma_X <= 0 or not np.isfinite(sigma_X):
            sigma_X = 1e-4

        F, Q = self._matrices(sigma_X=sigma_X)

        # Initial state: z0 = [X1_0, beta0, 0, X2_0 - beta0*X1_0]^T
        r0 = x2[0] - self.p.beta0 * x1[0]
        z = np.array([x1[0], self.p.beta0, 0.0, r0])
        # Initial covariance: X1 known exactly, beta uncertain, M and R
        # initialised with their process-noise variance so the filter doesn't
        # start over-confident about M_0=0 and R_0=r0.
        P = np.diag([
            1e-8,                  # X1_0: near-certain (directly observed)
            self.p.p_beta0,        # beta_0: uncertain by design
            self.p.sigma_M ** 2,   # M_0: one period of M-noise uncertainty
            self.p.sigma_R ** 2,   # R_0: one period of R-noise uncertainty
        ])

        x1_f = np.zeros(n); beta_f = np.zeros(n); M_f = np.zeros(n); R_f = np.zeros(n)
        x1_p = np.zeros(n); beta_p = np.zeros(n); M_p = np.zeros(n); R_p = np.zeros(n)
        innov = np.zeros(n); innov_var = np.zeros(n)

        x1_f[0], beta_f[0], M_f[0], R_f[0] = z
        x1_p[0], beta_p[0], M_p[0], R_p[0] = z
        innov[0] = 0.0
        innov_var[0] = np.nan

        loglik = 0.0
        I4 = np.eye(4)
        K_traj = []

        for t in range(1, n):
            # ---- 1. Predict Step (4x4) ----
            z_pred = F @ z
            P_pred = F @ P @ F.T + Q

            # ---- 2. Observation Update (2x4) ----
            y = np.array([x2[t], x1[t]])
            H = np.array([
                [0.0, x1[t], 1.0, 1.0],
                [1.0, 0.0,   0.0, 0.0]
            ])

            y_pred = H @ z_pred
            e = y - y_pred
            S = H @ P_pred @ H.T

            # Bug 4 fix: add observation noise R_obs so the gain isn't
            # infinitely confident in the observed prices. Use 10% of sigma_X
            # as a proxy for bid/ask + discretisation noise.
            obs_noise = (sigma_X * 0.1) ** 2
            S[0, 0] += obs_noise   # X2 observation noise
            S[1, 1] += obs_noise   # X1 observation noise

            S[0, 0] = max(S[0, 0], 1e-12)
            S[1, 1] = max(S[1, 1], 1e-12)

            det_S = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            det_S = max(det_S, 1e-12)
            inv_S = np.array([
                [S[1, 1], -S[0, 1]],
                [-S[1, 0], S[0, 0]]
            ]) / det_S

            K = P_pred @ H.T @ inv_S

            z = z_pred + K @ e
            P = (I4 - K @ H) @ P_pred

            x1_p[t], beta_p[t], M_p[t], R_p[t] = z_pred
            x1_f[t], beta_f[t], M_f[t], R_f[t] = z

            innov[t] = e[0]
            innov_var[t] = S[0, 0]

            K_traj.append(K)
            
            loglik += -0.5 * (np.log(2 * np.pi * S[0, 0]) + (e[0] ** 2) / S[0, 0])
            
        K_traj = np.stack([np.zeros((4, 2))] + K_traj)

        return dict(
            x1=x1_f, beta=beta_f, M=M_f, R=R_f,
            x1_pred=x1_p, beta_pred=beta_p, M_pred=M_p, R_pred=R_p,
            innovation=innov, innovation_var=innov_var,
            loglik=loglik, P_final=P,
            F=F, Q=Q, K_traj=K_traj
        )


def fit_dynamic_pci(x1: np.ndarray, x2: np.ndarray, beta0_guess: float = None,
                     allow_dynamic_beta: bool = True, p_beta0: float = 1.0, verbose: bool = False,
                     maxiter: int = 500):
    """
    Fit Dynamic-PCI parameters via Maximum Likelihood Estimation (MLE).
    Optimizes 5 parameters (beta0, rho, sigma_M, sigma_R, q_beta).

    If allow_dynamic_beta=False, q_beta is fixed at 1e-12, recovering the static
    Clegg & Krauss baseline.
    """
    if beta0_guess is None:
        # OLS estimate of beta as initial starting point
        beta0_guess = float(np.cov(x1, x2)[0, 1] / np.var(x1))

    sigma_X = float(np.std(np.diff(x1)))
    if sigma_X <= 0 or not np.isfinite(sigma_X):
        sigma_X = 1e-4

    if not allow_dynamic_beta:
        q_beta_fixed = 1e-12
    else:
        q_beta_fixed = None

    if q_beta_fixed is not None:
        # q_beta is fixed (static baseline).
        # We optimize (beta0, rho, sigma_M, sigma_R) on the formation data.
        def objective(theta):
            b0, rho, log_sM, log_sR = theta
            if not (-0.999 < rho < 0.999):
                return 1e10
            params = DynamicPCIParams(
                beta0=b0, rho=rho,
                sigma_M=np.exp(log_sM), sigma_R=np.exp(log_sR),
                q_beta=q_beta_fixed, sigma_X=sigma_X, p_beta0=p_beta0,
            )
            kf = DynamicPCIKalmanFilter(params)
            try:
                res = kf.filter(x1, x2)
                ll = res["loglik"]
                return -ll if np.isfinite(ll) else 1e10
            except Exception:
                return 1e10

        diff_s = np.diff(x2 - beta0_guess * x1)
        std_s = float(np.std(diff_s)) if len(diff_s) > 0 else 1.0
        x0 = np.array([
            beta0_guess,
            0.9,
            np.log(std_s + 1e-6),
            np.log(std_s + 1e-6),
        ])
        bounds = [
            (None, None),
            (-0.999, 0.999),
            (np.log(1e-8), np.log(1e3)),
            (np.log(1e-8), np.log(1e3)),
        ]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter))
        b0, rho, log_sM, log_sR = result.x
        fitted = DynamicPCIParams(
            beta0=b0, rho=rho,
            sigma_M=np.exp(log_sM), sigma_R=np.exp(log_sR),
            q_beta=q_beta_fixed, sigma_X=sigma_X, p_beta0=p_beta0,
        )
        return fitted, result
    else:
        # 5D MLE
        def objective_5d(theta):
            b0, rho, log_sM, log_sR, log_qb = theta
            if not (-0.999 < rho < 0.999):
                return 1e10
            qb = np.exp(log_qb)
            params = DynamicPCIParams(
                beta0=b0, rho=rho,
                sigma_M=np.exp(log_sM), sigma_R=np.exp(log_sR),
                q_beta=qb, sigma_X=sigma_X, p_beta0=p_beta0,
            )
            kf = DynamicPCIKalmanFilter(params)
            try:
                res = kf.filter(x1, x2)
                ll = res["loglik"]
                obj_val = -ll
                return obj_val if np.isfinite(obj_val) else 1e10
            except Exception:
                return 1e10

        diff_s = np.diff(x2 - beta0_guess * x1)
        std_s = float(np.std(diff_s)) if len(diff_s) > 0 else 1.0
        
        qb_guess = 1e-6
        x0 = np.array([
            beta0_guess,
            0.9,
            np.log(std_s + 1e-6),
            np.log(std_s + 1e-6),
            np.log(qb_guess),
        ])
        bounds = [
            (None, None),
            (-0.999, 0.999),
            (np.log(1e-8), np.log(1e3)),
            (np.log(1e-8), np.log(1e3)),
            (np.log(1e-12), np.log(1e2)),
        ]
        result = minimize(objective_5d, x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter))
        b0, rho, log_sM, log_sR, log_qb = result.x
        fitted = DynamicPCIParams(
            beta0=b0, rho=rho,
            sigma_M=np.exp(log_sM), sigma_R=np.exp(log_sR),
            q_beta=np.exp(log_qb), sigma_X=sigma_X, p_beta0=p_beta0,
        )
        return fitted, result
