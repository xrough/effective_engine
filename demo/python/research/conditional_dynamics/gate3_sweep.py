#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
Gate 3: Regime Dynamics — Robustness Sweep
==========================================
Scores rough-vs-carry conditional performance across a grid of
H × resample_min × move_pct.

Workflow
--------
1. Extract 1-min records once (via _day_worker pool); cache to disk.
2. For each (H, resample, move_pct) cell:
   a. resample_panel(records, resample)
   b. pool_by_tenor_bucket(resampled)   ← follow front/mid/back roles
   c. recompute_structural(pooled, H)
   d. Per tenor series: evaluate_conditional(ts, H, move_pct)
   e. classify_gate3_cell(active_agg, quiet_agg) → PASS/MARGINAL/FAIL/SKIP
3. Write gate3_sweep_<timestamp>.csv; print summary tables (one per move_pct).

Usage
-----
  python gate3_sweep.py [--days N] [--far] [--workers N] [--device STR]
                         [--h-grid "0.05,0.10,0.20"] [--resample-grid "1,30,120"]
                         [--move-pct-grid "0.10,0.20,0.30"] [--no-cache] [--output]
"""

import argparse
import io
import math
import multiprocessing as mp
import re
import sys
import time
import zipfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

_HERE    = Path(__file__).resolve()
ROOT     = _HERE.parents[4]   # .../MVP
OPRA_ZIP = ROOT / "demo" / "data" / "OPRA-20260208-3T68RYYKF9.zip"
OUT_DIR  = _HERE.parent / "output"

sys.path.insert(0, str(_HERE.parents[1] / "shared"))
from smile_pipeline import (
    RATE, DIV, MIN_DTE, MAX_DTE,
    _Tee, get_device, evaluate_forecasts, process_day_full,
)
from robustness_sweeps import (
    SweepConfig,
    DEFAULT_H_GRID, DEFAULT_RESAMPLE_GRID, DEFAULT_MOVE_PCT_GRID,
    load_cache, save_cache,
    extraction_heartbeat,
    resample_panel, pool_by_tenor_bucket, recompute_structural,
    _agg_metrics, _avg_metrics,
    classify_gate3_cell, format_gate3_summary, _log,
)

# re-use the extraction worker from the single-run benchmark
sys.path.insert(0, str(_HERE.parent))
from benchmark import _day_worker, evaluate_conditional


# ─────────────────────────────────────────────────────────────────────────────
# Extraction (run once, then cached)
# ─────────────────────────────────────────────────────────────────────────────

def _extract(cfg: SweepConfig, dbn_files: list[tuple]) -> list:
    """Extract all-expiry 1-min records for all days. Returns flat record list."""
    total     = len(dbn_files)
    task_args = [
        (str(OPRA_ZIP), fname, tdate, 0.10, cfg.device, MIN_DTE, MAX_DTE)
        for fname, tdate in dbn_files
    ]

    t0    = time.time()
    day_map: dict = {}
    progress = {"completed": 0, "current": None, "start_ts": t0}

    with extraction_heartbeat(total, progress, label="Extracting Gate 3 days"):
        if cfg.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(cfg.workers) as pool:
                for tdate, recs in pool.imap_unordered(_day_worker, task_args):
                    progress["completed"] += 1
                    _log(f"Day {progress['completed']:3d}/{total:3d} — {tdate} — {len(recs)} records")
                    day_map[tdate] = recs
        else:
            with zipfile.ZipFile(OPRA_ZIP) as zf:
                for i, (fname, tdate) in enumerate(dbn_files, 1):
                    progress["current"] = tdate
                    _log(f"Day {i:3d}/{total:3d} — {tdate} — starting")
                    try:
                        recs = process_day_full(
                            zf, fname, tdate, 0.10, cfg.device, MIN_DTE, MAX_DTE)
                    except Exception as e:
                        _log(f"Day {i:3d}/{total:3d} — {tdate} — SKIP: {e}")
                        recs = []
                    else:
                        _log(f"Day {i:3d}/{total:3d} — {tdate} — {len(recs)} records")
                    day_map[tdate] = recs
                    progress["completed"] = i
                    progress["current"] = None

    elapsed = time.time() - t0
    flat = [r for recs in day_map.values() for r in recs]
    _log(f"Extraction done: {total} days, {len(flat)} records  (elapsed {elapsed:.0f}s)")
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Sweep evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_cell(records: list, H: float, resample: int,
                   move_pct: float,
                   select_far: bool) -> tuple[dict, dict, list]:
    """
    Evaluate one (H, resample, move_pct) cell.

    Returns (agg_active, agg_quiet, per_series_rows).
    agg_active / agg_quiet are averaged across tenor-role series; rows go into
    the CSV.
    """
    resampled = resample_panel(records, resample)
    pooled    = pool_by_tenor_bucket(resampled, select_far=select_far)
    selected  = recompute_structural(pooled, H)

    by_exp: dict = defaultdict(list)
    for r in selected:
        by_exp[r["series_id"]].append(r)

    min_bars = max(20, 60 // max(resample, 1))
    active_metrics: list[dict] = []
    quiet_metrics:  list[dict] = []
    per_expiry_rows: list[dict] = []

    for exp, ts in sorted(by_exp.items()):
        ts.sort(key=lambda r: r["ts"])
        if len(ts) < min_bars:
            continue

        cond = evaluate_conditional(ts, H, move_pct)
        rc   = cond["regime_counts"]

        act_agg = _agg_metrics(cond["active"]) if cond["active"] else None
        qui_agg = _agg_metrics(cond["quiet"])  if cond["quiet"]  else None
        all_agg = _agg_metrics(cond["all"])    if cond["all"]    else {}

        if act_agg:
            active_metrics.append(act_agg)
        if qui_agg:
            quiet_metrics.append(qui_agg)

        verdict = classify_gate3_cell(act_agg, qui_agg)

        row: dict = {
            "H": H, "resample_min": resample, "move_pct": move_pct,
            "series_id": str(exp),
            "expiry": str(exp),
            "n_bars_all":    len(ts),
            "n_bars_active": rc.get("active", 0),
            "n_bars_quiet":  rc.get("quiet",  0),
        }
        for suffix, agg in [("all", all_agg),
                             ("active", act_agg or {}),
                             ("quiet",  qui_agg  or {})]:
            for feat in ["rr25", "bf25"]:
                row[f"{feat}_rmse_carry_{suffix}"] = agg.get(f"{feat}_rmse_carry", float("nan"))
                row[f"{feat}_rmse_rough_{suffix}"] = agg.get(f"{feat}_rmse_rough", float("nan"))
        row["verdict"] = verdict
        per_expiry_rows.append(row)

    if not per_expiry_rows:
        return {}, {}, []

    avg_act = _avg_metrics(active_metrics) if active_metrics else {}
    avg_qui = _avg_metrics(quiet_metrics)  if quiet_metrics  else {}
    overall_verdict = classify_gate3_cell(
        avg_act if avg_act else None,
        avg_qui if avg_qui else None,
    )

    agg_row: dict = {
        "H": H, "resample_min": resample, "move_pct": move_pct,
        "series_id": "ALL",
        "expiry": "ALL",
        "n_bars_all":    sum(r["n_bars_all"]    for r in per_expiry_rows),
        "n_bars_active": sum(r["n_bars_active"] for r in per_expiry_rows),
        "n_bars_quiet":  sum(r["n_bars_quiet"]  for r in per_expiry_rows),
    }
    for feat in ["rr25", "bf25"]:
        for suffix, agg in [("active", avg_act), ("quiet", avg_qui)]:
            agg_row[f"{feat}_rmse_carry_{suffix}"] = agg.get(f"{feat}_rmse_carry", float("nan"))
            agg_row[f"{feat}_rmse_rough_{suffix}"] = agg.get(f"{feat}_rmse_rough", float("nan"))
        agg_row[f"{feat}_rmse_carry_all"] = float("nan")
        agg_row[f"{feat}_rmse_rough_all"] = float("nan")
    agg_row["verdict"] = overall_verdict
    per_expiry_rows.append(agg_row)

    return avg_act, avg_qui, per_expiry_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Gate 3 Robustness Sweep")
    ap.add_argument("--days",           type=int,   default=None)
    ap.add_argument("--far",            action="store_true")
    ap.add_argument("--workers",        type=int,   default=1)
    ap.add_argument("--device",         type=str,   default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--h-grid",         type=str,   default=None,
                    help='Comma-separated H values, e.g. "0.05,0.10,0.20"')
    ap.add_argument("--resample-grid",  type=str,   default=None,
                    help='Comma-separated resample minutes, e.g. "1,30,120"')
    ap.add_argument("--move-pct-grid",  type=str,   default=None,
                    help='Comma-separated move_pct values, e.g. "0.10,0.20,0.30"')
    ap.add_argument("--no-cache",       action="store_true",
                    help="Force re-extraction even if cache exists")
    ap.add_argument("--output",         action="store_true",
                    help="Save report to output/gate3_sweep_<timestamp>.txt")
    args = ap.parse_args()

    h_grid = ([float(x) for x in args.h_grid.split(",")]
              if args.h_grid else DEFAULT_H_GRID)
    resample_grid = ([int(x) for x in args.resample_grid.split(",")]
                     if args.resample_grid else DEFAULT_RESAMPLE_GRID)
    move_pct_grid = ([float(x) for x in args.move_pct_grid.split(",")]
                     if args.move_pct_grid else DEFAULT_MOVE_PCT_GRID)
    device = get_device(args.device)

    cfg = SweepConfig(
        gate_id="gate3",
        h_grid=h_grid,
        resample_grid=resample_grid,
        move_pct_grid=move_pct_grid,
        n_days=args.days,
        select_far=args.far,
        workers=args.workers,
        device=device,
    )

    if not OPRA_ZIP.exists():
        sys.exit(f"OPRA zip not found: {OPRA_ZIP}")

    buf = None
    if args.output:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        sys.stdout = _Tee(sys.__stdout__, buf)

    try:
        _run(cfg, no_cache=args.no_cache)
    finally:
        if buf is not None:
            sys.stdout = sys.__stdout__
            stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"gate3_sweep_{stamp}.txt"
            out_path.write_text(buf.getvalue())
            print(f"\nReport saved → {out_path}")


def _run(cfg: SweepConfig, no_cache: bool = False):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── resolve day list ──────────────────────────────────────────────────────
    with zipfile.ZipFile(OPRA_ZIP) as zf_index:
        dbn_files_raw = sorted(f for f in zf_index.namelist()
                               if f.endswith(".dbn.zst"))
    if cfg.n_days:
        dbn_files_raw = dbn_files_raw[:cfg.n_days]

    dbn_files: list[tuple] = []
    for fname in dbn_files_raw:
        ds    = re.search(r"(\d{8})", fname).group(1)
        tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        dbn_files.append((fname, tdate))

    sel_label = "far" if cfg.select_far else "near"
    n_cells   = len(cfg.h_grid) * len(cfg.resample_grid) * len(cfg.move_pct_grid)

    print(f"\nGate 3 Robustness Sweep")
    print(f"  Run:          {run_ts}")
    print(f"  Days:         {len(dbn_files)}  |  select={sel_label}")
    print(f"  Workers:      {cfg.workers}  |  device={cfg.device}")
    print(f"  H grid:       {cfg.h_grid}")
    print(f"  Resample:     {cfg.resample_grid} min")
    print(f"  Move-pct:     {cfg.move_pct_grid}")
    print(f"  Total cells:  {n_cells}")

    # ── load or build cache ───────────────────────────────────────────────────
    records = None if no_cache else load_cache(cfg)
    if records is None:
        _log(f"Extracting {len(dbn_files)} days with {cfg.workers} workers ...")
        records = _extract(cfg, dbn_files)
        save_cache(cfg, records)

    if not records:
        sys.exit("No records extracted. Check data path and market-hours filter.")

    _log(f"Starting sweep: {n_cells} cells "
         f"({len(cfg.h_grid)} H × {len(cfg.resample_grid)} resample × "
         f"{len(cfg.move_pct_grid)} move_pct)")

    # ── sweep ─────────────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    cell_idx = 0

    for H in cfg.h_grid:
        for resample in cfg.resample_grid:
            for move_pct in cfg.move_pct_grid:
                cell_idx += 1
                t_cell = time.time()
                cell_state = {
                    "completed": 0,
                    "current": f"H={H:.2f} resample={resample} move_pct={move_pct:.0%}",
                    "start_ts": t_cell,
                }
                _log(f"Cell {cell_idx:3d}/{n_cells:3d} — starting  H={H:.2f}  "
                     f"resample={resample:4d} min  move_pct={move_pct:.0%}")
                with extraction_heartbeat(1, cell_state,
                                          label="Evaluating Gate 3 cell",
                                          interval_s=5.0):
                    avg_act, avg_qui, rows = _evaluate_cell(
                        records, H, resample, move_pct, cfg.select_far)
                    cell_state["completed"] = 1
                elapsed_c = time.time() - t_cell

                verdict = classify_gate3_cell(
                    avg_act if avg_act else None,
                    avg_qui if avg_qui else None,
                )
                act_rr_rough = avg_act.get("rr25_rmse_rough", float("nan")) if avg_act else float("nan")
                act_rr_carry = avg_act.get("rr25_rmse_carry", float("nan")) if avg_act else float("nan")
                act_bf_rough = avg_act.get("bf25_rmse_rough", float("nan")) if avg_act else float("nan")
                act_bf_carry = avg_act.get("bf25_rmse_carry", float("nan")) if avg_act else float("nan")
                _log(f"Cell {cell_idx:3d}/{n_cells:3d}  H={H:.2f}  "
                     f"resample={resample:4d} min  move_pct={move_pct:.0%}  →  "
                     f"{verdict:<9s}  "
                     f"(active rr25 carry={act_rr_carry:.5f} rough={act_rr_rough:.5f} "
                     f"| bf25 carry={act_bf_carry:.5f} rough={act_bf_rough:.5f})  "
                     f"[{elapsed_c:.1f}s]")

                all_rows.extend(rows)

    if not all_rows:
        sys.exit("Sweep produced no results.")

    results_df = pd.DataFrame(all_rows)

    # ── CSV output ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"gate3_sweep_{stamp}.csv"
    results_df.to_csv(csv_path, index=False)
    _log(f"CSV saved → {csv_path}")

    # ── summary tables (one per move_pct) ─────────────────────────────────────
    for move_pct in cfg.move_pct_grid:
        print(format_gate3_summary(results_df, move_pct))
    print()


if __name__ == "__main__":
    main()
