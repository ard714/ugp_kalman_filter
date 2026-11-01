"""
dynamic_pci_3state.py — 3-State Partial Cointegration Kalman Filter
====================================================================

IEEE-paper formulation: X1 is a purely *exogenous* driver.
State vector  z_t = [beta_t, M_t, R_t]^T   (dim=3)

Observation equation (scalar, observed X2_t):
    y_t = X1_t*beta_t + M_t + R_t + v_t,  v_t ~ N(0, sigma_obs^2)
    H_t = [X1_t, 1, 1]   (time-varying due to X1_t)

State transitions:
    beta_t = beta_{t-1} + w_b,   w_b ~ N(0, q_beta^2)
    M_t    = rho*M_{t-1} + w_M,  w_M ~ N(0, sigma_M^2)
    R_t    = R_{t-1}     + w_R,  w_R ~ N(0, sigma_R^2)

F = [[1, 0,   0],
     [0, rho, 0],
     [0, 0,   1]]

Q = diag([q_beta^2, sigma_M^2, sigma_R^2])
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class PCI3Params:
    beta0: float      # Initial hedge ratio
    rho: float        # AR(1) coefficient of M
    sigma_M: float    # Std dev of innovations to M
    sigma_R: float    # Std dev of innovations to R
    q_beta: float     # Process noise std dev of beta random walk
    sigma_obs: float  # Observation noise std dev
    p_beta0: float = 1.0  # Initial variance of beta

    @property
    def r2_mr(self) -> float:
        """R^2_MR = 2*sigma_M^2 / (2*sigma_M^2 + (1+rho)*sigma_R^2)."""
        num = 2.0 * self.sigma_M ** 2
        den = num + (1.0 + self.rho) * self.sigma_R ** 2
        return num / den if den > 0 else 0.0

    @property
    def half_life(self) -> float:
        """Mean-reversion half-life in trading periods."""
        if not (0 < self.rho < 0.9999):
            return np.inf
        return np.log(0.5) / np.log(self.rho)

    @property
    def pci_index(self) -> float:
        return self.r2_mr


# ---------------------------------------------------------------------------
# Kalman Filter
# ---------------------------------------------------------------------------

class PCI3KalmanFilter:
    """3-state Kalman filter: state = [beta_t, M_t, R_t], X1 exogenous."""

    def __init__(self, params: PCI3Params):
        self.p = params

    def _FQ(self):
        p = self.p
        F = np.array([
            [1.0,   0.0,  0.0],
            [0.0, p.rho,  0.0],
            [0.0,   0.0,  1.0],
        ])
        Q = np.diag([p.q_beta**2, p.sigma_M**2, p.sigma_R**2])
        return F, Q

    def filter(self, x1: np.ndarray, x2: np.ndarray) -> dict:
        """
        Forward Kalman filter pass.

        Returns dict: beta, M, R (posterior), beta_pred, M_pred, R_pred (prior),
        innovation, innovation_var, loglik, P_final, F, Q.
        """
        n = len(x1)
        p = self.p
        F, Q = self._FQ()
        R_obs = p.sigma_obs ** 2

        # Initial state: beta=beta0, M=0, R=X2_0 - beta0*X1_0
        r0 = float(x2[0]) - p.beta0 * float(x1[0])
        z = np.array([p.beta0, 0.0, r0])
        P = np.diag([p.p_beta0, p.sigma_M**2, p.sigma_R**2])

        beta_f = np.zeros(n); M_f = np.zeros(n); R_f = np.zeros(n)
        beta_p = np.zeros(n); M_p = np.zeros(n); R_p = np.zeros(n)
        innov  = np.zeros(n); innov_var = np.zeros(n)

        beta_f[0], M_f[0], R_f[0] = z
        beta_p[0], M_p[0], R_p[0] = z
        innov[0]     = 0.0
        innov_var[0] = np.nan

        loglik = 0.0
        I3 = np.eye(3)

        for t in range(1, n):
            # Predict
            z_pred = F @ z
            P_pred = F @ P @ F.T + Q

            # Observation: H_t = [X1_t, 1, 1]
            x1t = float(x1[t])
            H   = np.array([[x1t, 1.0, 1.0]])   # (1, 3)

            y_pred_val = float((H @ z_pred)[0])
            e = float(x2[t]) - y_pred_val

            S = float((H @ P_pred @ H.T)[0, 0]) + R_obs
            S = max(S, 1e-12)

            # Kalman gain (3, 1)
            K = (P_pred @ H.T) / S

            # Update
            z = z_pred + K[:, 0] * e
            P = (I3 - K @ H) @ P_pred

            beta_p[t], M_p[t], R_p[t] = z_pred
            beta_f[t], M_f[t], R_f[t] = z
            innov[t]     = e
            innov_var[t] = S

            loglik += -0.5 * (np.log(2.0 * np.pi * S) + (e**2) / S)

        return dict(
            beta=beta_f, M=M_f, R=R_f,
            beta_pred=beta_p, M_pred=M_p, R_pred=R_p,
            innovation=innov, innovation_var=innov_var,
            loglik=loglik, P_final=P,
            F=F, Q=Q,
        )


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------

def fit_pci3(
    x1: np.ndarray,
    x2: np.ndarray,
    beta0_guess: float = None,
    allow_dynamic_beta: bool = True,
    p_beta0: float = 1.0,
    verbose: bool = False,
):
    """
    Fit PCI3 parameters via Maximum Likelihood.

    If allow_dynamic_beta=False, q_beta fixed at 1e-8 (static-beta baseline).

    Returns (PCI3Params, scipy OptimizeResult).
    """
    if beta0_guess is None:
        beta0_guess = float(np.cov(x1, x2)[0, 1] / (np.var(x1) + 1e-12))

    spread = x2 - beta0_guess * x1
    std_s  = float(np.std(np.diff(spread))) if len(spread) > 1 else 1.0

    q_beta_fixed = None if allow_dynamic_beta else 1e-8

    def _neg_loglik(theta):
        if q_beta_fixed is None:
            b0, rho, log_sM, log_sR, log_qb, log_sobs = theta
            qb = np.exp(log_qb)
        else:
            b0, rho, log_sM, log_sR, log_sobs = theta
            qb = q_beta_fixed
        if not (-0.9999 < rho < 0.9999):
            return 1e10
        params = PCI3Params(
            beta0=b0, rho=rho,
            sigma_M=np.exp(log_sM), sigma_R=np.exp(log_sR),
            q_beta=np.sqrt(qb),
            sigma_obs=np.exp(log_sobs),
            p_beta0=p_beta0,
        )
        kf = PCI3KalmanFilter(params)
        try:
            res = kf.filter(x1, x2)
            ll  = res["loglik"]
            return -ll if np.isfinite(ll) else 1e10
        except Exception:
            return 1e10

    log_sM0   = np.log(std_s + 1e-6)
    log_sR0   = np.log(std_s + 1e-6)
    log_sobs0 = np.log(std_s * 0.05 + 1e-8)

    if q_beta_fixed is None:
        x0     = [beta0_guess, 0.9, log_sM0, log_sR0, np.log(1e-6), log_sobs0]
        bounds = [
            (None, None),
            (-0.9999, 0.9999),
            (np.log(1e-8), np.log(1e3)),
            (np.log(1e-8), np.log(1e3)),
            (np.log(1e-12), np.log(1e2)),
            (np.log(1e-8), np.log(1e2)),
        ]
    else:
        x0     = [beta0_guess, 0.9, log_sM0, log_sR0, log_sobs0]
        bounds = [
            (None, None),
            (-0.9999, 0.9999),
            (np.log(1e-8), np.log(1e3)),
            (np.log(1e-8), np.log(1e3)),
            (np.log(1e-8), np.log(1e2)),
        ]

    result = minimize(
        _neg_loglik, x0, method="L-BFGS-B", bounds=bounds,
        options=dict(maxiter=1000, ftol=1e-12, gtol=1e-8),
    )

    if verbose:
        print(f"  MLE converged={result.success}, fun={result.fun:.4f}, nit={result.nit}")

    if q_beta_fixed is None:
        b0, rho, log_sM, log_sR, log_qb, log_sobs = result.x
        qb = np.exp(log_qb)
    else:
        b0, rho, log_sM, log_sR, log_sobs = result.x
        qb = q_beta_fixed

    fitted = PCI3Params(
        beta0=b0, rho=rho,
        sigma_M=np.exp(log_sM), sigma_R=np.exp(log_sR),
        q_beta=np.sqrt(qb),
        sigma_obs=np.exp(log_sobs),
        p_beta0=p_beta0,
    )
    return fitted, result
