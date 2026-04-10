#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
Gate 2: Temporal Forecast — Robustness Sweep
=============================================
Scores rough-vs-carry performance across a grid of H × resample_min.

Workflow
--------
1. Extract 1-min records once (via _gate0_worker pool); cache to disk.
2. For each (H, resample) cell:
   a. resample_panel(records, resample)
   b. recompute_structural(resampled, H)   ← only alpha/gamma change
   c. Per expiry: evaluate_forecasts(ts, H)
   d. classify_gate0_cell → PASS/MARGINAL/FAIL
3. Write gate0_sweep_<timestamp>.csv; print summary pivot table.

Usage
-----
  python gate0_sweep.py [--days N] [--far] [--workers N] [--device STR]
                        [--h-grid "0.05,0.10,0.20"] [--resample-grid "1,30,120"]
                        [--no-cache] [--output]
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
    _Tee, get_device, evaluate_forecasts,
)
from robustness_sweeps import (
    SweepConfig, DEFAULT_H_GRID, DEFAULT_RESAMPLE_GRID,
    load_cache, save_cache,
    extraction_heartbeat, resample_panel, recompute_structural,
    _agg_metrics, _avg_metrics, classify_gate0_cell,
    format_gate0_summary, _log,
)

# re-use the extraction worker from the single-run benchmark
sys.path.insert(0, str(_HERE.parent))
from gate0_forecast_benchmark import _gate0_worker, process_day

N_EXP = 2


# ─────────────────────────────────────────────────────────────────────────────
# Extraction (run once, then cached)
# ─────────────────────────────────────────────────────────────────────────────

def _extract(cfg: SweepConfig, dbn_files: list[tuple]) -> list:
    """Extract 1-min records for all days. Returns flat list of record dicts."""
    total     = len(dbn_files)
    task_args = [
        (str(OPRA_ZIP), fname, tdate, 0.10, cfg.select_far)   # H=0.10 placeholder; recomputed later
        for fname, tdate in dbn_files
    ]

    t0    = time.time()
    day_map: dict = {}
    progress = {"completed": 0, "current": None, "start_ts": t0}

    with extraction_heartbeat(total, progress, label="Extracting Gate 0 days"):
        if cfg.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(cfg.workers) as pool:
                for tdate, recs in pool.imap_unordered(_gate0_worker, task_args):
                    progress["completed"] += 1
                    _log(f"Day {progress['completed']:3d}/{total:3d} — {tdate} — {len(recs)} records")
                    day_map[tdate] = recs
        else:
            with zipfile.ZipFile(OPRA_ZIP) as zf:
                for i, (fname, tdate) in enumerate(dbn_files, 1):
                    progress["current"] = tdate
                    _log(f"Day {i:3d}/{total:3d} — {tdate} — starting")
                    try:
                        recs = process_day(zf, fname, tdate, 0.10, cfg.select_far)
                    except Exception as e:
                        _log(f"Day {i:3d}/{total:3d} — {tdate} — SKIP: {e}")
                        recs = []
                    else:
                        _log(f"Day {i:3d}/{total:3d} — {tdate} — {len(recs)} records")
                    day_map[tdate] = recs
                    progress["completed"] = i
                    progress["current"] = None

    elapsed = time.time() - t0
    flat = [r for tdate, recs in day_map.items() for r in recs]
    _log(f"Extraction done: {total} days, {len(flat)} records  (elapsed {elapsed:.0f}s)")
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# Sweep evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_cell(records: list, H: float, resample: int) -> tuple[dict, list]:
    """
    Evaluate one (H, resample) cell.

    Returns (agg_metrics, per_expiry_rows) where agg_metrics is averaged across
    expiries and per_expiry_rows is a list of row dicts for the CSV.
    """
    resampled = resample_panel(records, resample)
    resampled = recompute_structural(resampled, H)

    # group by expiry
    by_exp: dict = defaultdict(list)
    for r in resampled:
        by_exp[r["expiry"]].append(r)

    min_bars = max(20, 60 // max(resample, 1))
    expiry_metrics: list[dict] = []
    per_expiry_rows: list[dict] = []

    for exp, ts in sorted(by_exp.items()):
        ts.sort(key=lambda r: r["ts"])
        if len(ts) < min_bars:
            continue
        ev  = evaluate_forecasts(ts, H)
        agg = _agg_metrics(ev)
        expiry_metrics.append(agg)

        row = {"H": H, "resample_min": resample, "expiry": str(exp),
               "n_bars": len(ts)}
        row.update(agg)
        row["verdict"] = classify_gate0_cell(agg)
        per_expiry_rows.append(row)

    if not expiry_metrics:
        return {}, []

    avg = _avg_metrics(expiry_metrics)
    verdict = classify_gate0_cell(avg)
    agg_row = {"H": H, "resample_min": resample, "expiry": "ALL",
               "n_bars": sum(r["n_bars"] for r in per_expiry_rows)}
    agg_row.update(avg)
    agg_row["verdict"] = verdict
    per_expiry_rows.append(agg_row)

    return avg, per_expiry_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Gate 0 Robustness Sweep")
    ap.add_argument("--days",          type=int,   default=None)
    ap.add_argument("--far",           action="store_true")
    ap.add_argument("--workers",       type=int,   default=1)
    ap.add_argument("--device",        type=str,   default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--h-grid",        type=str,   default=None,
                    help='Comma-separated H values, e.g. "0.05,0.10,0.20"')
    ap.add_argument("--resample-grid", type=str,   default=None,
                    help='Comma-separated resample minutes, e.g. "1,30,120"')
    ap.add_argument("--no-cache",      action="store_true",
                    help="Force re-extraction even if cache exists")
    ap.add_argument("--output",        action="store_true",
                    help="Save report to output/gate0_sweep_<timestamp>.txt")
    args = ap.parse_args()

    h_grid = ([float(x) for x in args.h_grid.split(",")]
              if args.h_grid else DEFAULT_H_GRID)
    resample_grid = ([int(x) for x in args.resample_grid.split(",")]
                     if args.resample_grid else DEFAULT_RESAMPLE_GRID)
    device = get_device(args.device)

    cfg = SweepConfig(
        gate_id="gate0",
        h_grid=h_grid,
        resample_grid=resample_grid,
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
            out_path = OUT_DIR / f"gate0_sweep_{stamp}.txt"
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
    n_cells   = len(cfg.h_grid) * len(cfg.resample_grid)

    print(f"\nGate 0 Robustness Sweep")
    print(f"  Run:          {run_ts}")
    print(f"  Days:         {len(dbn_files)}  |  select={sel_label}")
    print(f"  Workers:      {cfg.workers}  |  device={cfg.device}")
    print(f"  H grid:       {cfg.h_grid}")
    print(f"  Resample:     {cfg.resample_grid} min")
    print(f"  Total cells:  {n_cells}")

    # ── load or build cache ───────────────────────────────────────────────────
    records = None if no_cache else load_cache(cfg)
    if records is None:
        _log(f"Extracting {len(dbn_files)} days with {cfg.workers} workers ...")
        records = _extract(cfg, dbn_files)
        save_cache(cfg, records)

    if not records:
        sys.exit("No records extracted. Check data path and market-hours filter.")

    _log(f"Starting sweep: {n_cells} cells ({len(cfg.h_grid)} H × "
         f"{len(cfg.resample_grid)} resample)")

    # ── sweep ─────────────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    cell_idx = 0

    for H in cfg.h_grid:
        for resample in cfg.resample_grid:
            cell_idx += 1
            t_cell = time.time()
            cell_state = {
                "completed": 0,
                "current": f"H={H:.2f} resample={resample}",
                "start_ts": t_cell,
            }
            _log(f"Cell {cell_idx:3d}/{n_cells:3d} — starting  H={H:.2f}  "
                 f"resample={resample:4d} min")
            with extraction_heartbeat(1, cell_state,
                                      label="Evaluating Gate 0 cell",
                                      interval_s=5.0):
                agg, rows = _evaluate_cell(records, H, resample)
                cell_state["completed"] = 1
            elapsed_c = time.time() - t_cell

            verdict = agg and classify_gate0_cell(agg) or "SKIP"
            rr_carry = agg.get("rr25_rmse_carry", float("nan"))
            rr_rough = agg.get("rr25_rmse_rough", float("nan"))
            _log(f"Cell {cell_idx:3d}/{n_cells:3d}  H={H:.2f}  "
                 f"resample={resample:4d} min  →  {verdict:<9s} "
                 f"(rr25 carry={rr_carry:.5f} rough={rr_rough:.5f})  "
                 f"[{elapsed_c:.1f}s]")

            all_rows.extend(rows)

    if not all_rows:
        sys.exit("Sweep produced no results.")

    results_df = pd.DataFrame(all_rows)

    # ── CSV output ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"gate0_sweep_{stamp}.csv"
    results_df.to_csv(csv_path, index=False)
    _log(f"CSV saved → {csv_path}")

    # ── summary table ─────────────────────────────────────────────────────────
    print(format_gate0_summary(results_df))
    print()


if __name__ == "__main__":
    main()
