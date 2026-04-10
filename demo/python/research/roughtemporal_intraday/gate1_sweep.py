#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
Gate 4: Incremental Edge — Robustness Sweep
============================================
Scores whether rough adds incremental information over carry across a grid of
H x resample_min.

Gate 1 methods
--------------
- rough_cond_carry : carry corrected by the rough-vs-carry spread
- rough_recent     : recency-weighted rough coefficients (EWMA alpha/gamma)
"""

import argparse
import io
import multiprocessing as mp
import re
import sys
import time
import zipfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[4]
OPRA_ZIP = ROOT / "demo" / "data" / "OPRA-20260208-3T68RYYKF9.zip"
OUT_DIR = _HERE.parent / "output"

sys.path.insert(0, str(_HERE.parents[1] / "shared"))
from smile_pipeline import _Tee, get_device, evaluate_forecasts
from robustness_sweeps import (
    SweepConfig, DEFAULT_H_GRID, DEFAULT_RESAMPLE_GRID, DEFAULT_GATE1_METHODS,
    load_cache, save_cache, extraction_heartbeat, resample_panel,
    recompute_structural, _agg_metrics, _avg_metrics,
    _best_carry_improvement, classify_gate1_cell, _log,
)

sys.path.insert(0, str(_HERE.parent))
from gate0_forecast_benchmark import _gate0_worker, process_day


def _extract(cfg: SweepConfig, dbn_files: list[tuple]) -> list:
    total = len(dbn_files)
    task_args = [(str(OPRA_ZIP), fname, tdate, 0.10, cfg.select_far)
                 for fname, tdate in dbn_files]
    t0 = time.time()
    day_map = {}
    progress = {"completed": 0, "current": None, "start_ts": t0}

    with extraction_heartbeat(total, progress, label="Extracting Gate 1 days"):
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

    flat = [r for recs in day_map.values() for r in recs]
    elapsed = time.time() - t0
    _log(f"Extraction done: {total} days, {len(flat)} records  (elapsed {elapsed:.0f}s)")
    return flat


def _evaluate_cell(records: list, H: float, resample: int) -> tuple[dict, list]:
    resampled = resample_panel(records, resample)
    resampled = recompute_structural(resampled, H)

    by_exp: dict = defaultdict(list)
    for r in resampled:
        by_exp[r["expiry"]].append(r)

    min_bars = max(20, 60 // max(resample, 1))
    expiry_metrics = []
    rows = []

    for exp, ts in sorted(by_exp.items()):
        ts.sort(key=lambda r: r["ts"])
        if len(ts) < min_bars:
            continue
        ev = evaluate_forecasts(ts, H)
        agg = _agg_metrics(ev)
        best_imp, best_method, best_feat = _best_carry_improvement(
            agg, methods=list(DEFAULT_GATE1_METHODS))
        expiry_metrics.append(agg)
        row = {"H": H, "resample_min": resample, "expiry": str(exp), "n_bars": len(ts)}
        row.update(agg)
        row["best_imp"] = best_imp
        row["best_method"] = best_method or ""
        row["best_feature"] = best_feat or ""
        row["verdict"] = classify_gate1_cell(agg)
        rows.append(row)

    if not expiry_metrics:
        return {}, []

    avg = _avg_metrics(expiry_metrics)
    best_imp, best_method, best_feat = _best_carry_improvement(
        avg, methods=list(DEFAULT_GATE1_METHODS))
    agg_row = {"H": H, "resample_min": resample, "expiry": "ALL",
               "n_bars": sum(r["n_bars"] for r in rows)}
    agg_row.update(avg)
    agg_row["best_imp"] = best_imp
    agg_row["best_method"] = best_method or ""
    agg_row["best_feature"] = best_feat or ""
    agg_row["verdict"] = classify_gate1_cell(avg)
    rows.append(agg_row)
    return avg, rows


def format_gate1_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no Gate 1 results)"
    agg = df[df["expiry"] == "ALL"] if "expiry" in df.columns else df
    pivot = agg.pivot_table(index="H", columns="resample_min", values="verdict", aggfunc="first")
    col_w = max(10, *(len(str(c)) for c in pivot.columns))
    h_w = 6
    sep = "-" * (h_w + 3 + (col_w + 3) * len(pivot.columns))
    lines = [
        "\nGate 1 Robustness Sweep — incremental over carry",
        f"{'H':>{h_w}} | " + " | ".join(f"{c:>{col_w}}" for c in pivot.columns),
        sep,
    ]
    n_pass = n_total = 0
    for h_val, row in pivot.iterrows():
        cells = []
        for v in row:
            s = str(v) if pd.notna(v) else "SKIP"
            cells.append(f"{s:>{col_w}}")
            if s == "PASS":
                n_pass += 1
            if s != "SKIP":
                n_total += 1
        lines.append(f"{h_val:>{h_w}.2f} | " + " | ".join(cells))
    pct = 100.0 * n_pass / n_total if n_total else 0.0
    lines += [sep, f"\nPASS cells: {n_pass}/{n_total} ({pct:.0f}%)"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Gate 1 Robustness Sweep")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--far", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--h-grid", type=str, default=None)
    ap.add_argument("--resample-grid", type=str, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--output", action="store_true")
    args = ap.parse_args()

    h_grid = [float(x) for x in args.h_grid.split(",")] if args.h_grid else DEFAULT_H_GRID
    resample_grid = [int(x) for x in args.resample_grid.split(",")] if args.resample_grid else DEFAULT_RESAMPLE_GRID
    device = get_device(args.device)

    cfg = SweepConfig(
        gate_id="gate0",  # reuse Gate 0 extraction cache
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
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"gate1_sweep_{stamp}.txt"
            out_path.write_text(buf.getvalue())
            print(f"\nReport saved → {out_path}")


def _run(cfg: SweepConfig, no_cache: bool = False):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with zipfile.ZipFile(OPRA_ZIP) as zf_index:
        dbn_files_raw = sorted(f for f in zf_index.namelist() if f.endswith(".dbn.zst"))
    if cfg.n_days:
        dbn_files_raw = dbn_files_raw[:cfg.n_days]

    dbn_files = []
    for fname in dbn_files_raw:
        ds = re.search(r"(\d{8})", fname).group(1)
        tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        dbn_files.append((fname, tdate))

    print("\nGate 1 Robustness Sweep")
    print(f"  Run:          {run_ts}")
    print(f"  Days:         {len(dbn_files)}")
    print(f"  Workers:      {cfg.workers}  |  device={cfg.device}")
    print(f"  H grid:       {cfg.h_grid}")
    print(f"  Resample:     {cfg.resample_grid} min")
    print(f"  Methods:      {DEFAULT_GATE1_METHODS}")

    records = None if no_cache else load_cache(cfg)
    if records is None:
        _log(f"Extracting {len(dbn_files)} days with {cfg.workers} workers ...")
        records = _extract(cfg, dbn_files)
        save_cache(cfg, records)
    if not records:
        sys.exit("No records extracted.")

    rows = []
    n_cells = len(cfg.h_grid) * len(cfg.resample_grid)
    cell_idx = 0
    for H in cfg.h_grid:
        for resample in cfg.resample_grid:
            cell_idx += 1
            t_cell = time.time()
            state = {"completed": 0, "current": f"H={H:.2f} resample={resample}", "start_ts": t_cell}
            _log(f"Cell {cell_idx:3d}/{n_cells:3d} — starting  H={H:.2f}  resample={resample:4d} min")
            with extraction_heartbeat(1, state, label="Evaluating Gate 1 cell", interval_s=5.0):
                agg, cell_rows = _evaluate_cell(records, H, resample)
                state["completed"] = 1
            best_imp, best_method, best_feat = _best_carry_improvement(
                agg, methods=list(DEFAULT_GATE1_METHODS))
            verdict = classify_gate1_cell(agg) if agg else "SKIP"
            _log(f"Cell {cell_idx:3d}/{n_cells:3d}  H={H:.2f}  resample={resample:4d} min  "
                 f"→ {verdict:<8s}  best={best_method or 'n/a'}:{best_feat or 'n/a'} "
                 f"({best_imp:+.1%})  [{time.time() - t_cell:.1f}s]")
            rows.extend(cell_rows)

    if not rows:
        sys.exit("Gate 1 sweep produced no rows.")

    results_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"gate1_sweep_{stamp}.csv"
    results_df.to_csv(csv_path, index=False)
    _log(f"CSV saved → {csv_path}")
    print(format_gate1_summary(results_df))
    print()


if __name__ == "__main__":
    main()
