#!/usr/bin/env python3
"""
synthetic/sanity_check.py
=========================
Synthetic sanity harness for Gate 0 and Gate 0B.

It generates a noisy option-chain panel whose latent smile evolves on a stable
rough-style manifold, re-extracts features through smile_pipeline, and then
runs the same forecast comparisons used on Databento data.

This is the intended answer to:
  "If the world were rough-structured, would our gates recover that edge?"
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "shared"))
sys.path.insert(0, str(_HERE.parents[1] / "roughtemporal_intraday"))
sys.path.insert(0, str(_HERE.parents[1] / "conditional_dynamics"))

from synthetic_smile import SyntheticRoughConfig, generate_rough_synthetic_records
from smile_pipeline import _Tee, evaluate_forecasts
from gate0_forecast_benchmark import print_report as print_gate0_report
from benchmark import evaluate_conditional, print_report as print_conditional_report

OUT_DIR = _HERE.parent / "output"


def _run(cfg: SyntheticRoughConfig, move_pct: float):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    panel_df, state_df, records = generate_rough_synthetic_records(cfg)

    if not records:
        raise SystemExit("Synthetic generator produced no valid smile records.")

    print("\nsynthetic rough sanity check")
    print(f"  Run:      {run_ts}")
    print(f"  H:        {cfg.H:.2f}")
    print(f"  Bars:     {cfg.n_bars}  ({cfg.bar_minutes}-min)")
    print(f"  Expiries: {cfg.expiries_dte}")
    print(f"  Panel rows: {len(panel_df)}  |  extracted records: {len(records)}")
    print(f"  Active threshold: |ret| > {cfg.active_move_threshold:.4f}")
    print(f"  Noise: quiet={cfg.quiet_iv_noise:.4f}, active={cfg.active_iv_noise:.4f}\n")

    expiry_groups: dict[str, list] = defaultdict(list)
    for rec in records:
        expiry_groups[str(rec["expiry"])].append(rec)

    gate0_results = {}
    cond_results = {}
    for exp_str, ts_data in sorted(expiry_groups.items()):
        ts_data.sort(key=lambda r: r["ts"])
        gate0_results[exp_str] = evaluate_forecasts(list(ts_data), cfg.H)
        cond_results[exp_str] = evaluate_conditional(list(ts_data), cfg.H, move_pct=move_pct, min_bars=15)

    run_meta = {
        "run_ts": run_ts,
        "days": int(state_df["ts"].dt.date.nunique()),
        "first_date": str(state_df["ts"].min().date()),
        "last_date": str(state_df["ts"].max().date()),
        "select_mode": "synthetic",
        "total_raw": len(records),
        "total_resampled": len(records),
    }
    print_gate0_report(gate0_results, cfg.H, cfg.bar_minutes, run_meta)

    cond_meta = {
        "run_ts": run_ts,
        "days": int(state_df["ts"].dt.date.nunique()),
        "first_date": str(state_df["ts"].min().date()),
        "last_date": str(state_df["ts"].max().date()),
        "select_mode": "synthetic",
    }
    print_conditional_report(cond_results, cfg.H, cfg.bar_minutes, move_pct, cond_meta)

    rr_carry = np.mean([v["rr25"]["carry"]["rmse"] for v in gate0_results.values()])
    rr_rough = np.mean([v["rr25"]["rough"]["rmse"] for v in gate0_results.values()])
    bf_carry = np.mean([v["bf25"]["carry"]["rmse"] for v in gate0_results.values()])
    bf_rough = np.mean([v["bf25"]["rough"]["rmse"] for v in gate0_results.values()])
    print("Synthetic summary")
    print(f"  Gate 0 rr25 RMSE: carry={rr_carry:.6f}  rough={rr_rough:.6f}")
    print(f"  Gate 0 bf25 RMSE: carry={bf_carry:.6f}  rough={bf_rough:.6f}")


def main():
    ap = argparse.ArgumentParser(description="Synthetic sanity harness for Gate 0 / 0B")
    ap.add_argument("--h", type=float, default=0.10, help="Hurst exponent used in the latent world")
    ap.add_argument("--bars", type=int, default=180, help="Number of synthetic bars")
    ap.add_argument("--bar-minutes", type=int, default=30, help="Bar size in minutes")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--move-pct", type=float, default=0.20, help="Active-regime top fraction for Gate 0B")
    ap.add_argument("--output", action="store_true", help="Save the full report under synthetic/output/")
    args = ap.parse_args()

    cfg = SyntheticRoughConfig(
        seed=args.seed,
        H=args.h,
        n_bars=args.bars,
        bar_minutes=args.bar_minutes,
    )

    buf = None
    if args.output:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        sys.stdout = _Tee(sys.__stdout__, buf)

    try:
        _run(cfg, args.move_pct)
    finally:
        if buf is not None:
            sys.stdout = sys.__stdout__
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"synthetic_sanity_{stamp}.txt"
            out_path.write_text(buf.getvalue())
            print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()

