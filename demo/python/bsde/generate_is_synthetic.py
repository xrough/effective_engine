"""
generate_is_synthetic.py — Build IS-calibrated augmented BSDE training dataset
==============================================================================

Steps:
  1. Load real IS paths from extract_real_paths.py output (artifacts_is/)
  2. Simulate N_SYN synthetic LRH paths using IS-calibrated V0
  3. Combine real + synthetic (raw/unnormalized states)
  4. Compute normalization stats (mean/std per feature) from combined raw states
  5. Normalize combined states → training_states.npy
  6. Save dW1, payoff, tau_grid, and normalization_is.json to artifacts_is/

The resulting artifacts_is/ directory can be passed to trainer.py via --artifacts:
  python trainer.py --config configs/lifted_rough_heston.yaml --artifacts artifacts_is

Usage:
  cd demo/python/bsde
  python generate_is_synthetic.py               # default N_SYN=9500, seed=42
  python generate_is_synthetic.py --n-syn 5000  # fewer synthetic paths
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── LRH kernel constants (must match C++ LiftedHestonSim C_DEFAULT / LAMBDA_DEFAULT) ──
_C   = np.array([0.6796910829880118, 1.7847209579297232,
                 11.339598592234626, 41.010764478435718], dtype=np.float64)
_LAM = np.array([0.1, 4.641588833612779,
                 215.4434690031883,  10000.0], dtype=np.float64)

# ── Default LRH parameters (production values from demo/cpp main.cpp) ──
KAPPA = 0.3
THETA = 0.04
XI    = 0.5
RHO   = -0.507   # SPY/AAPL calibration value
S0    = 100.0
K     = 100.0
T_SIM = 1.0      # 1-year options (must match ONNX model's state_dim=7 training T)
R     = 0.05
V_FLOOR = 1e-6


def simulate_lrh_batch(
    n_paths:  int,
    n_steps:  int,
    V0:       float,
    seed:     int,
    antithetic: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Euler-Maruyama simulation of the Markovian Lifted Rough Heston model.

    Uses the same exponential integrator for U_k as LiftedHestonSim.cpp:
      U_k_{t+1} = exp(-λ_k dt) U_k_t
               + (1-exp(-λ_k dt))/λ_k · κ(θ - V_t)
               + ξ √V_t exp(-λ_k dt/2) dW2_t

    Returns:
      states_raw : (n_paths, n_steps+1, 7)  raw [tau, log(S/K), V, U1..U4]
      dW1        : (n_paths, n_steps)        raw log-return increments
      payoff     : (n_paths,)               discounted call payoff
    """
    rng    = np.random.default_rng(seed)
    dt     = T_SIM / n_steps
    disc   = np.exp(-R * T_SIM)

    n_base = n_paths // 2 if antithetic else n_paths

    # ── Correlated BM increments: dW1, dW2 ──
    Z1_base = rng.standard_normal((n_base, n_steps)) * np.sqrt(dt)
    Z2_base = rng.standard_normal((n_base, n_steps)) * np.sqrt(dt)
    dW1_base = Z1_base
    dW2_base = RHO * Z1_base + np.sqrt(1 - RHO**2) * Z2_base

    if antithetic:
        dW1 = np.concatenate([dW1_base, -dW1_base], axis=0)   # (N, T)
        dW2 = np.concatenate([dW2_base, -dW2_base], axis=0)
    else:
        dW1 = dW1_base
        dW2 = dW2_base

    # Pre-compute exponential-integrator constants
    decay  = np.exp(-_LAM * dt)                  # (4,)
    inv_l  = np.where(_LAM > 1e-12, 1.0 / _LAM, dt)  # (4,)
    e_half = np.exp(-_LAM * dt * 0.5)            # (4,)

    # ── Path simulation ──
    tau_grid = np.array([T_SIM - i * dt for i in range(n_steps + 1)], dtype=np.float32)

    states_raw = np.zeros((n_paths, n_steps + 1, 7), dtype=np.float32)
    dW1_out    = dW1.astype(np.float32)
    payoff_out = np.zeros(n_paths, dtype=np.float32)

    # Vectorized over paths: use (n_paths, 4) for U
    logS = np.full(n_paths, np.log(S0 / K), dtype=np.float64)
    V    = np.full(n_paths, max(V0, V_FLOOR), dtype=np.float64)
    U    = np.zeros((n_paths, 4), dtype=np.float64)

    # Store initial state
    states_raw[:, 0, 0] = tau_grid[0]
    states_raw[:, 0, 1] = logS.astype(np.float32)
    states_raw[:, 0, 2] = V.astype(np.float32)
    states_raw[:, 0, 3:7] = U.astype(np.float32)

    for i in range(n_steps):
        V_safe = np.maximum(V, V_FLOOR)           # (N,)
        sqrtV  = np.sqrt(V_safe)                  # (N,)
        dw1_i  = dW1[:, i]                        # (N,)
        dw2_i  = dW2[:, i]                        # (N,)

        # Update log-spot: log-normal Euler step
        logS = logS + (R - 0.5 * V_safe) * dt + sqrtV * dw1_i

        # Update U_k (vectorized over paths and factors)
        # drift: (N, 4) via broadcasting; diff: (N, 4)
        drift_k = (1 - decay) * inv_l * KAPPA * (THETA - V_safe[:, None])
        diff_k  = XI * sqrtV[:, None] * e_half * dw2_i[:, None]
        U = decay * U + drift_k + diff_k

        # Reconstruct V from U factors
        V_recon = V0 + (U * _C).sum(axis=1)
        V = np.maximum(V_recon, V_FLOOR)

        # Store state for step i+1
        states_raw[:, i + 1, 0] = tau_grid[i + 1]
        states_raw[:, i + 1, 1] = logS.astype(np.float32)
        states_raw[:, i + 1, 2] = V.astype(np.float32)
        states_raw[:, i + 1, 3:7] = U.astype(np.float32)

    # Terminal payoff: discounted European call
    S_T = K * np.exp(logS)
    payoff_out[:] = (disc * np.maximum(S_T - K, 0.0)).astype(np.float32)

    return states_raw, dW1_out, payoff_out


def build_is_training_data(
    real_states_path:  str | Path,
    real_dW1_path:     str | Path,
    real_payoff_path:  str | Path,
    meta_path:         str | Path,
    n_syn:             int   = 9500,
    seed:              int   = 42,
    output_dir:        str | Path = "artifacts_is",
    n_steps:           int   = 50,
) -> None:
    """
    Combine real IS paths with IS-calibrated synthetic paths and save
    normalized training data to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real IS paths ──
    real_states = np.load(real_states_path)   # (N_real, 51, 7) raw
    real_dW1    = np.load(real_dW1_path)      # (N_real, 50)
    real_payoff = np.load(real_payoff_path)   # (N_real,)
    with open(meta_path) as f:
        meta = json.load(f)
    is_v0_mean = float(meta["is_v0_mean"])
    n_real     = real_states.shape[0]
    n_steps_real = real_states.shape[1] - 1

    print(f"[gen] Real IS paths: {n_real}  (V0_mean={is_v0_mean:.6f})")

    # Handle step mismatch (edge case: real paths extracted with different n_steps)
    if n_steps_real != n_steps:
        # Resample real states to n_steps via linear interpolation along time axis
        import scipy.interpolate as sci
        xp  = np.linspace(0, 1, n_steps_real + 1)
        xq  = np.linspace(0, 1, n_steps + 1)
        new_states = np.zeros((n_real, n_steps + 1, 7), dtype=np.float32)
        for f_idx in range(7):
            new_states[:, :, f_idx] = sci.interp1d(xp, real_states[:, :, f_idx],
                                                     axis=1)(xq)
        real_states = new_states
        # Resample dW1 by summing consecutive increments where needed
        if n_steps_real > n_steps:
            factor = n_steps_real // n_steps
            real_dW1 = real_dW1.reshape(n_real, n_steps, factor).sum(axis=2)
        elif n_steps_real < n_steps:
            real_dW1 = np.pad(real_dW1, ((0,0),(0, n_steps - n_steps_real)), constant_values=0)
        print(f"[gen] Resampled real states: {real_states.shape}")

    # ── Generate IS-calibrated synthetic paths ──
    print(f"[gen] Simulating {n_syn} synthetic LRH paths (V0={is_v0_mean:.6f}, T={T_SIM}, steps={n_steps}) ...")
    syn_states, syn_dW1, syn_payoff = simulate_lrh_batch(
        n_paths=n_syn, n_steps=n_steps, V0=is_v0_mean, seed=seed, antithetic=True
    )
    print(f"[gen] Synthetic paths shape: {syn_states.shape}")
    print(f"[gen] Synthetic mean payoff: {syn_payoff.mean():.4f}  std: {syn_payoff.std():.4f}")

    # ── Combine ──
    all_states = np.concatenate([real_states, syn_states], axis=0)   # (N, 51, 7)
    all_dW1    = np.concatenate([real_dW1,    syn_dW1],    axis=0)   # (N, 50)
    all_payoff = np.concatenate([real_payoff, syn_payoff], axis=0)   # (N,)
    n_total    = all_states.shape[0]
    print(f"[gen] Combined: {n_total} paths ({n_real} real + {n_syn} synthetic)")

    # Shuffle combined dataset
    rng_idx = np.random.default_rng(seed + 1)
    perm    = rng_idx.permutation(n_total)
    all_states = all_states[perm]
    all_dW1    = all_dW1[perm]
    all_payoff = all_payoff[perm]

    # ── Normalization stats from combined raw states ──
    flat      = all_states.reshape(-1, 7).astype(np.float64)
    norm_mean = flat.mean(axis=0).astype(np.float32)
    norm_std  = flat.std(axis=0).astype(np.float32)
    norm_std  = np.where(norm_std < 1e-6, 1.0, norm_std)  # prevent divide by zero

    print(f"[gen] Normalization mean: {norm_mean}")
    print(f"[gen] Normalization std:  {norm_std}")

    # ── Normalize states ──
    norm_states = ((all_states.astype(np.float64) - norm_mean) / norm_std).astype(np.float32)

    # Tau grid (uniform reference, same as LRH C++ generator)
    tau_grid = np.array([T_SIM - i * (T_SIM / n_steps) for i in range(n_steps + 1)],
                        dtype=np.float32)

    # ── Save training data (same filenames as LRHDataset expects) ──
    np.save(output_dir / "training_states.npy",  norm_states)
    np.save(output_dir / "training_dW1.npy",     all_dW1)
    np.save(output_dir / "training_payoff.npy",  all_payoff)
    np.save(output_dir / "training_tau_grid.npy", tau_grid)

    print(f"[gen] Saved training_*.npy to {output_dir}/")

    # ── Save normalization JSON (for C++ NeuralBSDEHedger runtime) ──
    norm_json = {
        "mean":       norm_mean.tolist(),
        "std":        norm_std.tolist(),
        "state_dim":  7,
        "K_train":    float(K),    # nominal training strike — used by NeuralBSDEHedger for Z→delta scaling
        "model_params": {
            "kappa":   KAPPA,
            "theta":   THETA,
            "xi":      XI,
            "rho":     RHO,
            "V0":      is_v0_mean,
            "T":       T_SIM,
            "n_steps": n_steps,
        },
        "is_training_info": {
            "n_real_paths":      n_real,
            "n_synthetic_paths": n_syn,
            "n_total_paths":     n_total,
            "is_v0_mean":        is_v0_mean,
            "split_date":        meta.get("split_date", "2026-01-01"),
        },
    }
    # Save both normalization.json (for LRHDataset compatibility) and normalization_is.json
    with open(output_dir / "normalization.json",    "w") as f:
        json.dump(norm_json, f, indent=2)
    with open(output_dir / "normalization_is.json", "w") as f:
        json.dump(norm_json, f, indent=2)
    print(f"[gen] Saved normalization.json + normalization_is.json")
    print(f"\n[gen] Done. Training data summary:")
    print(f"  training_states.npy : {norm_states.shape}  (normalized)")
    print(f"  training_dW1.npy    : {all_dW1.shape}")
    print(f"  training_payoff.npy : {all_payoff.shape}")
    print(f"  mean payoff: {all_payoff.mean():.4f}  std: {all_payoff.std():.4f}")
    print(f"\n  Next: python trainer.py --config configs/lifted_rough_heston.yaml --artifacts {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate IS-calibrated augmented BSDE training dataset")
    parser.add_argument("--n-syn",       type=int, default=9500,
                        help="Number of synthetic LRH paths to generate (default: 9500)")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--n-steps",     type=int, default=50)
    parser.add_argument("--artifacts",   default=None,
                        help="artifacts_is directory (default: ./artifacts_is)")
    args = parser.parse_args()

    here = Path(__file__).parent
    artifacts = Path(args.artifacts) if args.artifacts else here / "artifacts_is"

    build_is_training_data(
        real_states_path = artifacts / "real_is_states_raw.npy",
        real_dW1_path    = artifacts / "real_is_dW1.npy",
        real_payoff_path = artifacts / "real_is_payoff.npy",
        meta_path        = artifacts / "real_is_meta.json",
        n_syn            = args.n_syn,
        seed             = args.seed,
        output_dir       = artifacts,
        n_steps          = args.n_steps,
    )


if __name__ == "__main__":
    main()
