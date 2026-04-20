"""
calibrate_and_retrain.py — Walk-forward SPY calibration and BSDE retraining
============================================================================

Estimates LRH parameters from a historical SPY window, generates
SPY-calibrated synthetic training paths, and warmstart-retrains the
NeuralBSDEHedger.  Strictly causal: only data before --train-end is used.

Walk-forward schedule (example with 5 trading days):
  python calibrate_and_retrain.py --train-end 2025-08-11   # train on days 1-3 → deploy day 4+
  python calibrate_and_retrain.py --train-end 2025-08-12   # train on days 1-4 → deploy day 5

Usage:
  cd demo/python/bsde
  python calibrate_and_retrain.py \\
      --csv  ../../data/spy_chain_panel.csv \\
      --train-end 2025-08-11 \\
      --epochs 100

Outputs (overwrites demo/artifacts/):
  artifacts/neural_bsde.onnx
  artifacts/normalization.json   (SPY-calibrated mean/std, K=ATM strike, T=T_SIM)
  artifacts/Y0_init.json
  artifacts/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Module path setup ──────────────────────────────────────────────────────────
_bsde_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_bsde_dir))

import generate_is_synthetic as gen_mod
from model   import SharedWeightMLP, bsde_forward, bsde_loss, compute_bs_delta_target
from data_loader import LRHDataset, load_norm_stats
from export  import export_onnx


# ── SPY parameter estimation ───────────────────────────────────────────────────

def estimate_spy_params(csv_path: str, train_end: str) -> dict:
    """
    Read spy_chain_panel.csv up to train_end (exclusive) and estimate LRH params.

    Returns dict with:
      theta   — long-run variance (mean V_t from window)
      K       — ATM strike from first bar (nominal training strike)
      T_sim   — median time-to-expiry × 2 (so training paths span a full cycle)
      rho     — median ssvi_rho (market skew, more accurate than AAPL default)
      v0_mean — mean V_t (used as V0 for simulation)
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
    df["date"] = df["timestamp_utc"].dt.normalize().dt.tz_localize(None)
    df = df[df["date"] < pd.Timestamp(train_end)]

    if df.empty:
        raise ValueError(f"No rows before {train_end} in {csv_path}")

    df["V_t"] = df["atm_iv"].clip(lower=0.01) ** 2

    theta   = float(df["V_t"].mean())
    v0_mean = theta
    K       = float(df["atm_strike"].iloc[0])
    # T_sim: train on paths with same T as actual options (~front weekly mean × 2)
    T_sim   = float(df["time_to_expiry"].median()) * 2.0
    T_sim   = max(T_sim, 0.04)   # floor at ~15 calendar days

    # SPY skew from SSVI fit (more accurate than AAPL default -0.507)
    rho_series = df["ssvi_rho"].dropna()
    rho = float(rho_series.median()) if len(rho_series) > 0 else -0.507

    print(f"[calibrate] SPY params from {len(df)} bars (up to {train_end}):")
    print(f"  θ (long-run V)  = {theta:.6f}  (ann. vol ≈ {theta**0.5*100:.1f}%)")
    print(f"  V0              = {v0_mean:.6f}")
    print(f"  K (ATM strike)  = {K:.1f}")
    print(f"  T_sim           = {T_sim:.4f} yr  (~{T_sim*365:.0f} cal days)")
    print(f"  ρ (SSVI skew)   = {rho:.4f}")

    return dict(theta=theta, K=K, T_sim=T_sim, rho=rho, v0_mean=v0_mean)


# ── Data generation ────────────────────────────────────────────────────────────

def generate_spy_paths(params: dict, n_syn: int, seed: int,
                        n_steps: int, artifacts_dir: Path) -> None:
    """
    Monkey-patch gen_mod constants with SPY values, simulate paths,
    compute normalization, and save training_*.npy + normalization.json
    to artifacts_dir.
    """
    # Override module-level constants to SPY values
    gen_mod.THETA = params["theta"]
    gen_mod.K     = params["K"]
    gen_mod.S0    = params["K"]      # S0 = K (start ATM)
    gen_mod.T_SIM = params["T_sim"]
    gen_mod.RHO   = params["rho"]
    # Keep KAPPA=0.3, XI=0.5 (not well-identified from short window)

    print(f"\n[calibrate] Simulating {n_syn} SPY-calibrated LRH paths "
          f"(K={params['K']:.0f}, θ={params['theta']:.4f}, T={params['T_sim']:.4f})...")

    states_raw, dW1, payoff = gen_mod.simulate_lrh_batch(
        n_paths=n_syn, n_steps=n_steps,
        V0=params["v0_mean"], seed=seed, antithetic=True
    )
    print(f"  states shape: {states_raw.shape}  payoff mean: {payoff.mean():.4f}")

    # Normalization stats from synthetic paths
    flat      = states_raw.reshape(-1, 7).astype(np.float64)
    norm_mean = flat.mean(axis=0).astype(np.float32)
    norm_std  = flat.std(axis=0).astype(np.float32)
    norm_std  = np.where(norm_std < 1e-6, 1.0, norm_std)

    print(f"  norm mean (V_t): {norm_mean[2]:.5f}  std (V_t): {norm_std[2]:.5f}")
    print(f"  norm mean (tau): {norm_mean[0]:.5f}  std (tau): {norm_std[0]:.5f}")

    norm_states = ((states_raw.astype(np.float64) - norm_mean) / norm_std).astype(np.float32)

    tau_grid = np.array(
        [params["T_sim"] - i * (params["T_sim"] / n_steps) for i in range(n_steps + 1)],
        dtype=np.float32
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifacts_dir / "training_states.npy",   norm_states)
    np.save(artifacts_dir / "training_dW1.npy",      dW1)
    np.save(artifacts_dir / "training_payoff.npy",   payoff)
    np.save(artifacts_dir / "training_tau_grid.npy", tau_grid)

    # Write normalization.json (read by NeuralBSDEHedger at startup)
    norm_json = {
        "feature_order": ["tau", "log_moneyness", "V_t", "U1", "U2", "U3", "U4"],
        "mean":       norm_mean.tolist(),
        "std":        norm_std.tolist(),
        "state_dim":  7,
        "m":          4,
        "K":          float(params["K"]),
        "T":          float(params["T_sim"]),
        "r":          0.05,
        "model_params": {
            "S0":    float(params["K"]),
            "V0":    float(params["v0_mean"]),
            "kappa": float(gen_mod.KAPPA),
            "theta": float(params["theta"]),
            "xi":    float(gen_mod.XI),
            "rho":   float(params["rho"]),
        },
    }
    with open(artifacts_dir / "normalization.json", "w") as f:
        json.dump(norm_json, f, indent=2)
    print(f"  Saved normalization.json  (K={params['K']:.0f}, T={params['T_sim']:.4f})")


# ── Warmstart training ─────────────────────────────────────────────────────────

def warmstart_train(artifacts_dir: Path, warmstart_path: Path | None,
                    n_epochs: int, params: dict) -> None:
    """
    Train in lrh_delta mode (learn only delta, leave gamma+vega as alpha).
    Warmstarts from warmstart_path if provided and architecture matches.
    """
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds     = LRHDataset(artifacts_dir)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    state_dim = ds.state_dim
    hidden_dim = 64

    norm_stats  = load_norm_stats(artifacts_dir)
    norm_mean_t = torch.tensor(norm_stats["mean"], dtype=torch.float32).to(device)
    norm_std_t  = torch.tensor(norm_stats["std"],  dtype=torch.float32).to(device)

    net = SharedWeightMLP(state_dim=state_dim, hidden_dim=hidden_dim).to(device)

    # Warmstart: load weights if checkpoint exists and state_dim matches
    loaded_warmstart = False
    if warmstart_path and warmstart_path.exists():
        try:
            ckpt_old = torch.load(warmstart_path, map_location="cpu", weights_only=False)
            if ckpt_old.get("state_dim", 0) == state_dim:
                net.load_state_dict(ckpt_old["net"])
                print(f"[calibrate] Warmstart from {warmstart_path} "
                      f"(epoch {ckpt_old['epoch']}, loss {ckpt_old['loss']:.5f})")
                loaded_warmstart = True
        except Exception as e:
            print(f"[calibrate] Warmstart failed ({e}), training from scratch")

    if not loaded_warmstart:
        print("[calibrate] Training from scratch (no compatible checkpoint)")

    # Y0 init from mean BS delta PnL on first batch
    with torch.no_grad():
        s0, dw0, _ = next(iter(loader))
        s0, dw0 = s0.to(device), dw0.to(device)
        y0_init = compute_bs_delta_target(
            s0, dw0, norm_mean_t, norm_std_t,
            K=params["K"], r=0.05,
        ).mean().item()
    print(f"[calibrate] Y0 init = {y0_init:.4f}")

    Y0 = nn.Parameter(torch.tensor([y0_init], device=device))
    lambda_z = 0.005

    opt_net  = torch.optim.Adam(net.parameters(), lr=1e-4 if loaded_warmstart else 5e-4)
    opt_Y0   = torch.optim.Adam([Y0], lr=0.1)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_net, T_max=n_epochs)

    (artifacts_dir / "checkpoints").mkdir(exist_ok=True)
    best_loss, best_ckpt = float("inf"), None

    for epoch in range(1, n_epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches  = 0

        for s_b, dw_b, p_b in loader:
            s_b, dw_b = s_b.to(device), dw_b.to(device)
            target = compute_bs_delta_target(
                s_b, dw_b, norm_mean_t, norm_std_t,
                K=params["K"], r=0.05,
            )
            opt_net.zero_grad()
            opt_Y0.zero_grad()
            Y_T, Z_norms = bsde_forward(net, Y0, s_b, dw_b)
            loss, comps  = bsde_loss(Y_T, target, Z_norms, lambda_z=lambda_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt_net.step()
            opt_Y0.step()
            epoch_loss += comps["total"]
            n_batches  += 1

        sched.step()
        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{n_epochs}  loss={avg_loss:.5f}  "
                  f"Y0={Y0.item():.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = {
                "epoch": epoch, "net": net.state_dict(),
                "Y0": Y0.item(), "state_dim": state_dim,
                "hidden_dim": hidden_dim, "loss": avg_loss,
            }
            torch.save(best_ckpt, artifacts_dir / "checkpoints" / "best.pt")

    print(f"\n[calibrate] Best loss={best_loss:.5f}  final Y0={Y0.item():.4f}")

    # Sanity check: delta at ATM (S=K, log_moneyness=0, tau=T_sim/2)
    _check_atm_delta(net, norm_mean_t, norm_std_t, params, device)


def _check_atm_delta(net, norm_mean_t, norm_std_t, params, device):
    """Print model delta at ATM vs Black-Scholes benchmark."""
    import math
    T   = params["T_sim"]
    V_t = params["v0_mean"]
    sig = math.sqrt(max(V_t, 1e-6))

    raw = np.array([[T/2, 0.0, V_t, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    state_norm = ((raw - norm_mean_t.cpu().numpy()) / norm_std_t.cpu().numpy()).astype(np.float32)

    net.eval()
    with torch.no_grad():
        _, Z = net(torch.from_numpy(state_norm).to(device))
    Z_spot = float(Z[0, 0].cpu())
    model_delta = Z_spot / (sig * params["K"])

    # BS delta at ATM: d1 = 0.5*σ*√T → N(d1)
    from math import erf, sqrt, pi
    norm_cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    d1 = 0.5 * sig * math.sqrt(T/2)
    bs_delta = norm_cdf(d1)

    print(f"\n[sanity] ATM delta check  (tau={T/2:.3f}, σ={sig:.3f}, K={params['K']:.0f})")
    print(f"  Model delta = {model_delta:.4f}")
    print(f"  BS delta    = {bs_delta:.4f}  (N(d1))")
    print(f"  Error       = {abs(model_delta - bs_delta):.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward SPY LRH calibration + BSDE retraining"
    )
    parser.add_argument("--csv",       default="../../data/spy_chain_panel.csv",
                        help="Path to spy_chain_panel.csv")
    parser.add_argument("--train-end", default="2025-08-11",
                        help="Exclusive upper date for training window (YYYY-MM-DD)")
    parser.add_argument("--epochs",    type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--n-syn",     type=int, default=9500,
                        help="Number of synthetic paths (default: 9500)")
    parser.add_argument("--n-steps",   type=int, default=50,
                        help="Time steps per path (default: 50)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--artifacts", default="../../artifacts",
                        help="Output artifacts directory")
    parser.add_argument("--no-warmstart", action="store_true",
                        help="Train from scratch (ignore existing checkpoint)")
    args = parser.parse_args()

    artifacts_dir   = Path(args.artifacts).resolve()
    warmstart_path  = None if args.no_warmstart else (artifacts_dir / "checkpoints" / "best.pt")

    print("=" * 60)
    print("Walk-forward SPY BSDE Calibration")
    print(f"  Training window : up to {args.train_end}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Artifacts dir   : {artifacts_dir}")
    print("=" * 60)

    # Step 1: Estimate SPY params from real data
    params = estimate_spy_params(args.csv, args.train_end)

    # Step 2: Generate SPY-calibrated synthetic training data
    generate_spy_paths(params, n_syn=args.n_syn, seed=args.seed,
                        n_steps=args.n_steps, artifacts_dir=artifacts_dir)

    # Step 3: Warmstart retrain in lrh_delta mode
    print(f"\n[calibrate] Training {args.epochs} epochs (lrh_delta mode)...")
    warmstart_train(artifacts_dir, warmstart_path, args.epochs, params)

    # Step 4: Export ONNX
    print(f"\n[calibrate] Exporting ONNX model...")
    export_onnx(
        checkpoint_path=artifacts_dir / "checkpoints" / "best.pt",
        artifacts_dir=artifacts_dir,
        validate=True,
    )

    print("\n" + "=" * 60)
    print("Done. Next step: cd demo && ./build/alpha_runner")
    print("=" * 60)


if __name__ == "__main__":
    main()
