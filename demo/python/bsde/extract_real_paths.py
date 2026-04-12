"""
extract_real_paths.py — Extract BSDE training paths from IS-period SPY data
===========================================================================

Reads spy_chain_panel.csv (IS period only: start → SPLIT_DATE), groups rows
by expiry_date (one path per expiry cycle), builds the 7D BSDE state
[tau, log(S/K), V_t, U1..U4] at each bar, subsamples to N_STEPS uniform
steps, and saves raw (unnormalized) path arrays.

Outputs (in artifacts_is/):
  real_is_states_raw.npy  — (N_paths, N_STEPS+1, 7) float32, unnormalized
  real_is_dW1.npy         — (N_paths, N_STEPS)     float32
  real_is_payoff.npy      — (N_paths,)             float32
  real_is_tau_grid.npy    — (N_STEPS+1,)           float32  (uniform [T, 0])
  real_is_meta.json       — diagnostics (N_paths, mean V0, date range, etc.)

Usage:
  cd demo/python/bsde
  python extract_real_paths.py                          # default paths
  python extract_real_paths.py --split-date 2026-01-01 --n-steps 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Lifted-Rough-Heston kernel constants (same as C++ LiftedHestonSim / StateEstimator) ──
_C   = np.array([0.6796910829880118, 1.7847209579297232,
                 11.339598592234626, 41.010764478435718], dtype=np.float64)
_LAM = np.array([0.1, 4.641588833612779,
                 215.4434690031883, 10000.0], dtype=np.float64)
KAPPA, THETA, XI = 0.3, 0.04, 0.5
R_RATE = 0.05
V_FLOOR = 1e-6


def _invert_dW2(V_curr: float, V_new: float, dt: float) -> float:
    """
    Invert observed variance change V_curr → V_new over dt to get dW2_hat.
    Python port of LiftedHestonStateEstimator::update() (C++).
    """
    V_safe = max(V_curr, V_FLOOR)
    sum_ce  = float(np.sum(_C * XI * np.sqrt(V_safe) * np.exp(-_LAM * dt * 0.5)))
    drift_V = float(np.sum(_C * (1 - np.exp(-_LAM * dt)) *
                           np.where(_LAM > 1e-12, 1.0 / _LAM, dt) *
                           KAPPA * (THETA - V_safe)))
    if abs(sum_ce) < 1e-10:
        return 0.0
    return float(np.clip((V_new - V_curr - drift_V) / sum_ce, -5.0, 5.0))


def _step_U(U: np.ndarray, V: float, dt: float, dW2_hat: float) -> np.ndarray:
    """
    Step U_k forward using exponential integrator.
    Python port of LiftedHestonStateEstimator::update() (C++).
    """
    V_safe = max(V, V_FLOOR)
    decay  = np.exp(-_LAM * dt)
    inv_l  = np.where(_LAM > 1e-12, 1.0 / _LAM, dt)
    drift  = (1 - decay) * inv_l * KAPPA * (THETA - V_safe)
    diff   = XI * np.sqrt(V_safe) * np.exp(-_LAM * dt * 0.5) * dW2_hat
    return decay * U + drift + diff


def extract_real_paths(
    csv_path: str | Path,
    split_date: str = "2026-01-01",
    n_steps: int = 50,
    min_rows: int | None = None,  # default: n_steps + 1
    output_dir: str | Path = "artifacts_is",
) -> dict:
    """
    Extract IS-period BSDE training paths from spy_chain_panel.csv.

    Returns dict with:
      states_raw: (N, n_steps+1, 7)  — unnormalized [tau, logm, V, U1..U4]
      dW1:        (N, n_steps)
      payoff:     (N,)
      is_v0_mean: float  — IS mean implied variance (for synthetic augmentation)
    """
    if min_rows is None:
        min_rows = n_steps + 1

    csv_path   = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Reading {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        usecols=["timestamp_utc", "underlying_price", "atm_strike",
                 "expiry_date", "time_to_expiry", "call_mid", "atm_iv"],
        dtype={
            "underlying_price": "float32",
            "atm_strike":       "float32",
            "time_to_expiry":   "float32",
            "call_mid":         "float32",
            "atm_iv":           "float32",
        },
    )
    df["date"] = df["timestamp_utc"].str[:10]
    df = df[df["date"] < split_date].copy()
    df.sort_values(["expiry_date", "timestamp_utc"], inplace=True, ignore_index=True)

    # Replace zero/NaN atm_iv with forward-fill within each expiry cycle
    df["atm_iv"] = df["atm_iv"].replace(0.0, np.nan)
    df["atm_iv"] = df.groupby("expiry_date")["atm_iv"].transform(
        lambda s: s.ffill().bfill()
    )
    df["V_t"] = df["atm_iv"].clip(lower=0.01) ** 2  # implied variance floor at (1%)^2

    is_v0_mean = float(df["V_t"].mean())
    print(f"[extract] IS rows: {len(df)}, date range: {df['date'].min()} → {df['date'].max()}")
    print(f"[extract] IS mean atm_iv: {df['atm_iv'].mean():.4f}, mean V0: {is_v0_mean:.6f}")

    all_states: list[np.ndarray] = []
    all_dW1:   list[np.ndarray] = []
    all_payoff: list[float]     = []
    skipped = 0

    for expiry, grp in df.groupby("expiry_date"):
        grp = grp.reset_index(drop=True)
        if len(grp) < min_rows:
            skipped += 1
            continue

        # Uniform subsampling: pick n_steps+1 indices
        idx = np.linspace(0, len(grp) - 1, n_steps + 1, dtype=int)
        sub = grp.iloc[idx].reset_index(drop=True)

        T0 = float(sub["time_to_expiry"].iloc[0])
        K  = float(sub["atm_strike"].iloc[0])
        if T0 <= 0 or K <= 0:
            skipped += 1
            continue

        # Terminal payoff: discounted call intrinsic at last subsampled bar
        S_last   = float(sub["underlying_price"].iloc[-1])
        tau_last = float(sub["time_to_expiry"].iloc[-1])
        payoff   = float(np.exp(-R_RATE * T0) * max(S_last - K, 0.0))

        # Build 7D state trajectory and dW1 increments
        states = np.zeros((n_steps + 1, 7), dtype=np.float32)
        dW1    = np.zeros(n_steps, dtype=np.float32)
        U      = np.zeros(4, dtype=np.float64)
        V_prev = float(sub["V_t"].iloc[0])

        for i in range(n_steps + 1):
            row   = sub.iloc[i]
            tau   = float(row["time_to_expiry"])
            S     = float(row["underlying_price"])
            V_t   = float(row["V_t"])

            if i > 0:
                prev  = sub.iloc[i - 1]
                dt    = float(prev["time_to_expiry"]) - tau  # > 0 (tau decreasing)
                if dt > 1e-8:
                    dW2_hat = _invert_dW2(V_prev, V_t, dt)
                    U       = _step_U(U, V_prev, dt, dW2_hat)
                    S_prev  = float(prev["underlying_price"])
                    dlog_S  = np.log(max(S, 1e-4) / max(S_prev, 1e-4))
                    dW1[i - 1] = float(dlog_S)  # log-return ≈ spot BM increment

            logm = float(np.log(max(S, 1e-4) / max(K, 1e-4)))
            states[i] = [tau, logm, V_t, U[0], U[1], U[2], U[3]]
            V_prev = V_t

        all_states.append(states)
        all_dW1.append(dW1)
        all_payoff.append(payoff)

    n_paths = len(all_states)
    print(f"[extract] Extracted {n_paths} paths  (skipped {skipped} — too short)")

    if n_paths == 0:
        raise RuntimeError("No paths extracted — check CSV path and split_date")

    states_arr  = np.stack(all_states, axis=0)   # (N, 51, 7)
    dW1_arr     = np.stack(all_dW1,   axis=0)    # (N, 50)
    payoff_arr  = np.array(all_payoff, dtype=np.float32)  # (N,)
    # Uniform tau_grid for reference (actual taus embedded in states[:,:,0])
    tau_grid = np.linspace(states_arr[0, 0, 0], states_arr[0, -1, 0],
                           n_steps + 1, dtype=np.float32)

    # Save raw (unnormalized) outputs
    np.save(output_dir / "real_is_states_raw.npy",  states_arr)
    np.save(output_dir / "real_is_dW1.npy",         dW1_arr)
    np.save(output_dir / "real_is_payoff.npy",       payoff_arr)
    np.save(output_dir / "real_is_tau_grid.npy",     tau_grid)

    meta = {
        "n_paths":     n_paths,
        "n_steps":     n_steps,
        "state_dim":   7,
        "is_v0_mean":  is_v0_mean,
        "split_date":  split_date,
        "date_range":  [df["date"].min(), df["date"].max()],
        "payoff_mean": float(payoff_arr.mean()),
        "payoff_std":  float(payoff_arr.std()),
    }
    with open(output_dir / "real_is_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[extract] Saved to {output_dir}/")
    print(f"  real_is_states_raw.npy : {states_arr.shape}")
    print(f"  real_is_dW1.npy        : {dW1_arr.shape}")
    print(f"  real_is_payoff.npy     : {payoff_arr.shape}")
    print(f"  mean payoff: {payoff_arr.mean():.4f}  std: {payoff_arr.std():.4f}")

    return {
        "states_raw": states_arr,
        "dW1":        dW1_arr,
        "payoff":     payoff_arr,
        "tau_grid":   tau_grid,
        "is_v0_mean": is_v0_mean,
        "n_paths":    n_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract IS-period BSDE training paths")
    parser.add_argument("--csv",         default=None,
                        help="Path to spy_chain_panel.csv (default: auto-detect)")
    parser.add_argument("--split-date",  default="2026-01-01",
                        help="Date before which rows are IS (exclusive, YYYY-MM-DD)")
    parser.add_argument("--n-steps",     type=int, default=50,
                        help="Number of BSDE time steps (subsampling target)")
    parser.add_argument("--output-dir",  default=None,
                        help="Output directory (default: artifacts_is/ next to this script)")
    args = parser.parse_args()

    # Auto-detect CSV
    if args.csv is None:
        here    = Path(__file__).parent
        options = [
            here.parent.parent / "data" / "spy_chain_panel.csv",
            here.parent.parent / "data" / "spy_atm_chain.csv",
        ]
        csv_path = next((p for p in options if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError(
                "spy_chain_panel.csv not found. Pass --csv explicitly.")
    else:
        csv_path = Path(args.csv)

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "artifacts_is"
    )

    extract_real_paths(
        csv_path=csv_path,
        split_date=args.split_date,
        n_steps=args.n_steps,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
