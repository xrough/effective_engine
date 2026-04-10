# ============================================================
# File: bsde_hedge_validation.py  (demo/python/validation/)
# Role: Validate the deep BSDE hedger against a standard
#       Black-Scholes delta hedge on controlled Rough Heston paths.
#
# Two-stage test:
#   Stage 1 — In-distribution (OOS pre-generated paths):
#     Uses artifacts/oos_states_raw.npy — exact U-factor trajectories
#     from the same simulator used for ONNX training.  No state-
#     estimation error; pure hedger-quality comparison.
#
#   Stage 2 — Bayer-Breneis fresh paths:
#     Generates new (S, V) paths via the Rough-Pricing repo's
#     order-2 weak scheme; reconstructs OU factors using
#     BSDEStateEstimator (port of C++ LiftedHestonStateEstimator).
#     Tests generalization to unseen, high-accuracy paths.
#
# Hedge error definition:
#   error_i = hedge_pnl_i - payoff_i + Y0
#   (P&L of the hedging portfolio: received Y0, paid payoff,
#    earned hedge_pnl from delta-rebalancing)
# ============================================================

import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import norm
import onnxruntime as ort

# ── Rough-Pricing repo ────────────────────────────────────────
_ROUGH_REPO = Path("/Users/xiaohao/rough_pricing_env/Rough-Pricing/src")
sys.path.insert(0, str(_ROUGH_REPO))
from roughvol.models.rough_heston_model import RoughHestonModel
from roughvol.types import MarketData, SimConfig

# ── Kernel weights (must match C++ LiftedHestonSim C_DEFAULT / LAMBDA_DEFAULT)
# Computed via NNLS on 500 log-uniform points [1e-4, 10], H=0.1, m=4
_C   = np.array([0.6797,  1.7847,  11.3396,  41.0108],  dtype=np.float64)
_LAM = np.array([0.1,     4.6416,  215.4435, 10000.0],  dtype=np.float64)


# ─────────────────────────────────────────────────────────────
# BSDEStateEstimator
# Port of demo/cpp/analytics/LiftedHestonStateEstimator.hpp
# ─────────────────────────────────────────────────────────────
class BSDEStateEstimator:
    """Reconstruct 4 OU factors from observed variance changes."""

    def __init__(self, kappa: float, theta: float, xi: float, V0: float):
        self.kappa = kappa
        self.theta = theta
        self.xi    = xi
        self.V0    = V0
        self.U     = np.zeros(4)
        self.V     = V0

    def reset(self, V0: float | None = None):
        self.U[:] = 0.0
        self.V    = V0 if V0 is not None else self.V0

    def update(self, V_new: float, dt: float) -> tuple[np.ndarray, float]:
        V_f   = max(self.V, 1e-6)
        decay = np.exp(-_LAM * dt)

        # Deterministic drift of each OU factor
        drift = (1.0 - decay) / _LAM * self.kappa * (self.theta - V_f)

        # Back-out dW2 from observed variance change
        dV_drift = (_C * drift).sum()
        sum_ce   = (_C * self.xi * np.sqrt(V_f) * np.exp(-_LAM * dt / 2.0)).sum()
        if abs(sum_ce) > 1e-10:
            dW2 = float(np.clip((V_new - self.V - dV_drift) / sum_ce, -5.0, 5.0))
        else:
            dW2 = 0.0

        # Update OU factors
        diff    = self.xi * np.sqrt(V_f) * np.exp(-_LAM * dt / 2.0) * dW2
        self.U  = decay * self.U + drift + diff
        self.V  = max(self.V0 + (_C * self.U).sum(), 1e-6)
        return self.U.copy(), self.V


# ─────────────────────────────────────────────────────────────
# Delta hedge (vectorized over all paths × time steps)
# ─────────────────────────────────────────────────────────────
def compute_delta_hedge_pnl(
    states: np.ndarray,   # (n_paths, n_times, 7)
    K: float,
    r: float,
) -> np.ndarray:          # (n_paths,)
    tau     = states[:, :, 0]                                    # (n, T+1)
    log_m   = states[:, :, 1]
    V_t     = states[:, :, 2]

    S_t     = K * np.exp(log_m)                                  # (n, T+1)
    sigma_t = np.sqrt(np.clip(V_t, 1e-8, None))
    tau_s   = np.maximum(tau, 1e-8)

    d1      = (log_m + (r + 0.5 * sigma_t**2) * tau_s) / (sigma_t * np.sqrt(tau_s))
    delta_t = norm.cdf(d1)                                       # (n, T+1)

    # Hedge P&L: use delta at t=0..T-1, spot change t=1..T
    dS         = S_t[:, 1:] - S_t[:, :-1]                       # (n, T)
    hedge_pnl  = (delta_t[:, :-1] * dS).sum(axis=1)             # (n,)
    return hedge_pnl


# ─────────────────────────────────────────────────────────────
# BSDE hedge (batched ONNX inference)
# ─────────────────────────────────────────────────────────────
def compute_bsde_hedge_pnl(
    states:    np.ndarray,         # (n_paths, n_times, 7)
    sess:      ort.InferenceSession,
    norm_mean: np.ndarray,         # (7,)
    norm_std:  np.ndarray,         # (7,)
    K:         float,
) -> np.ndarray:                   # (n_paths,)
    n_paths, n_times, _ = states.shape

    # Normalize entire trajectory in one shot
    norm_s = (states.astype(np.float32) - norm_mean) / norm_std  # (n, T+1, 7)

    # Batch ONNX: reshape to (n*T+1, 7)
    flat   = norm_s.reshape(-1, 7)
    Z_all  = sess.run(["Z"], {"state": flat})[0]                 # (n*(T+1), 7)
    # Z_spot = first component of Z: hedge w.r.t. spot Brownian W1
    Z_flat = Z_all[:, 0]                                         # (n*(T+1),)
    Z      = Z_flat.reshape(n_paths, n_times)                    # (n, T+1)

    # Convert Z_spot → delta = Z / (sigma * S)
    S_t     = K * np.exp(states[:, :, 1])
    sigma_t = np.sqrt(np.clip(states[:, :, 2], 1e-8, None))
    denom   = np.maximum(sigma_t * S_t, 1e-10)
    delta_t = Z / denom                                          # (n, T+1)

    dS        = S_t[:, 1:] - S_t[:, :-1]
    hedge_pnl = (delta_t[:, :-1] * dS).sum(axis=1)
    return hedge_pnl


# ─────────────────────────────────────────────────────────────
# Stage 2: Bayer-Breneis path generation + state reconstruction
# ─────────────────────────────────────────────────────────────
def run_bayer_breneis_paths(
    model_params: dict,
    n_paths:      int   = 2000,
    n_steps:      int   = 50,
    seed:         int   = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Rough Heston paths via Bayer-Breneis (order-2 weak scheme),
    reconstruct OU factors via BSDEStateEstimator, and return:
      states  (n_paths, n_steps+1, 7)  [tau, log(S/K), V_t, U1..U4]
      payoff  (n_paths,)               discounted European call payoff
    """
    p      = model_params
    T      = 1.0        # must match ONNX training
    K      = 100.0
    r      = 0.05

    model  = RoughHestonModel(
        hurst=p["H"], lam=p["kappa"], theta=p["theta"],
        nu=p["xi"],   rho=p["rho"],   v0=p["V0"],
        scheme="bayer-breneis", n_factors=8,
    )
    market = MarketData(spot=float(p["S0"]), rate=r, div_yield=0.0)
    cfg    = SimConfig(
        n_paths=n_paths, maturity=T, n_steps=n_steps,
        seed=seed, antithetic=False, store_paths=True,
    )

    print(f"  Simulating {n_paths} Bayer-Breneis paths (n_steps={n_steps})…", flush=True)
    paths  = model.simulate_paths(market=market, config=cfg)
    S_arr  = paths.state["spot"].astype(np.float64)  # (n, T+1) or (n, T)
    V_arr  = paths.state["var"].astype(np.float64)
    t_grid = paths.t.astype(np.float64)              # (T+1,) or (T,)

    # Ensure we have the initial time point
    n_t    = t_grid.shape[0]
    dt_arr = np.diff(t_grid)                         # (n_steps,)

    states = np.zeros((n_paths, n_t, 7), dtype=np.float32)
    est    = BSDEStateEstimator(
        kappa=p["kappa"], theta=p["theta"], xi=p["xi"], V0=p["V0"]
    )

    print(f"  Reconstructing OU factors for {n_paths} paths…", flush=True)
    for i in range(n_paths):
        if i % 500 == 0:
            print(f"    path {i}/{n_paths}", flush=True)
        est.reset(V0=V_arr[i, 0])
        for j in range(n_t):
            tau_j     = T - t_grid[j]
            S_j       = S_arr[i, j]
            V_j       = V_arr[i, j]
            log_m_j   = np.log(max(S_j / K, 1e-10))
            if j == 0:
                U_j = np.zeros(4)
            else:
                U_j, _ = est.update(V_j, dt_arr[j - 1])
            states[i, j] = [tau_j, log_m_j, V_j, U_j[0], U_j[1], U_j[2], U_j[3]]

    payoff = np.maximum(S_arr[:, -1] - K, 0.0).astype(np.float32)
    return states, payoff


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────
def _metrics(errors: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae":  float(np.mean(np.abs(errors))),
        "std":  float(np.std(errors)),
        "p5":   float(np.percentile(errors, 5)),
        "p95":  float(np.percentile(errors, 95)),
        "mean": float(np.mean(errors)),
    }


def print_report(
    label:        str,
    delta_errors: np.ndarray,
    bsde_errors:  np.ndarray,
    Y0:           float,
    model_params: dict,
) -> None:
    dm = _metrics(delta_errors)
    bm = _metrics(bsde_errors)
    p  = model_params

    def impr(d, b):
        return f"{(d - b) / max(abs(d), 1e-10) * 100.0:+.1f}%"

    W = 72
    bar = "═" * W
    sep = "─" * W

    rows = [
        ("RMSE ($)",  dm["rmse"], bm["rmse"]),
        ("MAE  ($)",  dm["mae"],  bm["mae"]),
        ("Std  ($)",  dm["std"],  bm["std"]),
        ("Mean ($)",  dm["mean"], bm["mean"]),
        ("P5  error", dm["p5"],   bm["p5"]),
        ("P95 error", dm["p95"],  bm["p95"]),
    ]

    print(f"\n╔{bar}╗")
    print(f"║  BSDE Hedge Validation — {label:<{W - 28}}║")
    print(f"╠{bar}╣")
    print(f"║  {'Metric':<18}  {'Delta':>12}  {'BSDE':>12}  {'Impr %':>10}  ║")
    print(f"║  {sep[:66]}  ║")
    for name, dv, bv in rows:
        print(f"║  {name:<18}  {dv:>12.4f}  {bv:>12.4f}  {impr(dv, bv):>10}  ║")
    print(f"╠{bar}╣")
    footer = (f"Y0=${Y0:.3f}  H={p['H']}  κ={p['kappa']}  "
              f"θ={p['theta']}  ξ={p['xi']}  ρ={p['rho']}  T=1yr")
    print(f"║  {footer:<{W - 2}}║")
    print(f"╚{bar}╝\n")


# ─────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────
def _sanity(label: str, delta_pnl: np.ndarray, bsde_pnl: np.ndarray,
            payoff: np.ndarray) -> None:
    assert not np.isnan(delta_pnl).any(), f"{label}: NaN in delta hedge P&L"
    assert not np.isnan(bsde_pnl).any(),  f"{label}: NaN in BSDE hedge P&L"
    print(f"  [{label}] sample path 0:"
          f"  payoff={payoff[0]:.3f}"
          f"  delta_pnl={delta_pnl[0]:.3f}"
          f"  bsde_pnl={bsde_pnl[0]:.3f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    BASE = Path(__file__).parents[2] / "artifacts"

    norm_cfg   = json.load(open(BASE / "normalization.json"))
    Y0         = json.load(open(BASE / "Y0_init.json"))["Y0"]
    K          = float(norm_cfg["K"])
    r          = float(norm_cfg["r"])
    model_p    = norm_cfg["model_params"]
    norm_mean  = np.array(norm_cfg["mean"], dtype=np.float32)
    norm_std   = np.array(norm_cfg["std"],  dtype=np.float32)

    print(f"Loading ONNX model from {BASE / 'neural_bsde.onnx'} …")
    sess = ort.InferenceSession(str(BASE / "neural_bsde.onnx"))

    # ── Stage 1: OOS pre-generated paths ─────────────────────
    print("\n── Stage 1: OOS pre-generated paths ──────────────────────")
    states = np.load(BASE / "oos_states_raw.npy").astype(np.float32)  # (2000, 51, 7)
    payoff = np.load(BASE / "oos_payoff.npy").astype(np.float32)       # (2000,)

    delta_pnl = compute_delta_hedge_pnl(states, K, r)
    bsde_pnl  = compute_bsde_hedge_pnl(states, sess, norm_mean, norm_std, K)

    _sanity("Stage1", delta_pnl, bsde_pnl, payoff)
    delta_err = delta_pnl - payoff + Y0
    bsde_err  = bsde_pnl  - payoff + Y0
    print_report(f"OOS pre-generated (n={len(payoff)})", delta_err, bsde_err, Y0, model_p)

    # ── Stage 2: Bayer-Breneis fresh paths ────────────────────
    print("── Stage 2: Bayer-Breneis fresh paths ────────────────────")
    bb_states, bb_payoff = run_bayer_breneis_paths(
        model_params=model_p, n_paths=2000, n_steps=50, seed=1,
    )

    delta_pnl2 = compute_delta_hedge_pnl(bb_states, K, r)
    bsde_pnl2  = compute_bsde_hedge_pnl(bb_states, sess, norm_mean, norm_std, K)

    _sanity("Stage2", delta_pnl2, bsde_pnl2, bb_payoff)
    delta_err2 = delta_pnl2 - bb_payoff + Y0
    bsde_err2  = bsde_pnl2  - bb_payoff + Y0
    print_report(f"Bayer-Breneis fresh (n={len(bb_payoff)})", delta_err2, bsde_err2, Y0, model_p)


if __name__ == "__main__":
    main()
