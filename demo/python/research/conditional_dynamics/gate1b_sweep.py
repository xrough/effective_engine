#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
Gate 5: Edge Concentration — Robustness Sweep
==============================================
Scores whether rough adds incremental information over carry in conditional
ACTIVE/QUIET regimes across:

  H x resample_min x move_pct
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
from smile_pipeline import _Tee, get_device, process_day_full
from robustness_sweeps import (
    SweepConfig, DEFAULT_H_GRID, DEFAULT_RESAMPLE_GRID, DEFAULT_MOVE_PCT_GRID,
    DEFAULT_GATE1_METHODS, load_cache, save_cache, extraction_heartbeat,
    resample_panel, pool_by_tenor_bucket, recompute_structural,
    _agg_metrics, _avg_metrics, _best_carry_improvement,
    classify_gate1b_cell, _log,
)

sys.path.insert(0, str(_HERE.parent))
from benchmark import _day_worker, evaluate_conditional
from gate0b_sweep import _extract as _extract_gate0b


def _row_with_methods(row: dict, agg: dict, suffix: str) -> None:
    methods = ["carry", "rough", "rough_cond_carry", "rough_recent", "ar1"]
    for feat in ["rr25", "bf25"]:
        for method in methods:
            row[f"{feat}_rmse_{method}_{suffix}"] = agg.get(f"{feat}_rmse_{method}", float("nan"))


def _evaluate_cell(records: list, H: float, resample: int,
                   move_pct: float, select_far: bool) -> tuple[dict, dict, list]:
    resampled = resample_panel(records, resample)
    pooled = pool_by_tenor_bucket(resampled, select_far=select_far)
    selected = recompute_structural(pooled, H)

    by_series: dict = defaultdict(list)
    for r in selected:
        by_series[r["series_id"]].append(r)

    min_bars = max(20, 60 // max(resample, 1))
    active_metrics = []
    quiet_metrics = []
    rows = []

    for series_id, ts in sorted(by_series.items()):
        ts.sort(key=lambda r: r["ts"])
        if len(ts) < min_bars:
            continue

        cond = evaluate_conditional(ts, H, move_pct)
        rc = cond["regime_counts"]
        act_agg = _agg_metrics(cond["active"]) if cond["active"] else None
        qui_agg = _agg_metrics(cond["quiet"]) if cond["quiet"] else None
        all_agg = _agg_metrics(cond["all"]) if cond["all"] else {}

        if act_agg:
            active_metrics.append(act_agg)
        if qui_agg:
            quiet_metrics.append(qui_agg)

        best_act_imp, best_act_method, best_act_feat = _best_carry_improvement(
            act_agg, methods=list(DEFAULT_GATE1_METHODS))
        best_qui_imp, best_qui_method, best_qui_feat = _best_carry_improvement(
            qui_agg, methods=list(DEFAULT_GATE1_METHODS))
        verdict = classify_gate1b_cell(act_agg, qui_agg)

        row = {
            "H": H, "resample_min": resample, "move_pct": move_pct,
            "series_id": str(series_id), "expiry": str(series_id),
            "n_bars_all": len(ts), "n_bars_active": rc.get("active", 0),
            "n_bars_quiet": rc.get("quiet", 0),
            "best_imp_active": best_act_imp,
            "best_method_active": best_act_method or "",
            "best_feature_active": best_act_feat or "",
            "best_imp_quiet": best_qui_imp,
            "best_method_quiet": best_qui_method or "",
            "best_feature_quiet": best_qui_feat or "",
            "verdict": verdict,
        }
        _row_with_methods(row, all_agg, "all")
        _row_with_methods(row, act_agg or {}, "active")
        _row_with_methods(row, qui_agg or {}, "quiet")
        rows.append(row)

    if not rows:
        return {}, {}, []

    avg_act = _avg_metrics(active_metrics) if active_metrics else {}
    avg_qui = _avg_metrics(quiet_metrics) if quiet_metrics else {}
    best_act_imp, best_act_method, best_act_feat = _best_carry_improvement(
        avg_act, methods=list(DEFAULT_GATE1_METHODS))
    best_qui_imp, best_qui_method, best_qui_feat = _best_carry_improvement(
        avg_qui, methods=list(DEFAULT_GATE1_METHODS))

    agg_row = {
        "H": H, "resample_min": resample, "move_pct": move_pct,
        "series_id": "ALL", "expiry": "ALL",
        "n_bars_all": sum(r["n_bars_all"] for r in rows),
        "n_bars_active": sum(r["n_bars_active"] for r in rows),
        "n_bars_quiet": sum(r["n_bars_quiet"] for r in rows),
        "best_imp_active": best_act_imp,
        "best_method_active": best_act_method or "",
        "best_feature_active": best_act_feat or "",
        "best_imp_quiet": best_qui_imp,
        "best_method_quiet": best_qui_method or "",
        "best_feature_quiet": best_qui_feat or "",
        "verdict": classify_gate1b_cell(avg_act if avg_act else None, avg_qui if avg_qui else None),
    }
    _row_with_methods(agg_row, {}, "all")
    _row_with_methods(agg_row, avg_act, "active")
    _row_with_methods(agg_row, avg_qui, "quiet")
    rows.append(agg_row)
    return avg_act, avg_qui, rows


def format_gate1b_summary(df: pd.DataFrame, move_pct: float) -> str:
    sub = df[(df["move_pct"] == move_pct) & (df["series_id"] == "ALL")]
    if sub.empty:
        return f"(no Gate 1B rows for move_pct={move_pct:.0%})"
    pivot = sub.pivot_table(index="H", columns="resample_min", values="verdict", aggfunc="first")
    col_w = max(10, *(len(str(c)) for c in pivot.columns))
    h_w = 6
    sep = "-" * (h_w + 3 + (col_w + 3) * len(pivot.columns))
    lines = [
        f"\nGate 1B Robustness Sweep — move_pct={move_pct:.0%}",
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
    lines += [sep, f"  {n_pass}/{n_total} cells PASS ({pct:.0f}%)"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Gate 1B Robustness Sweep")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--far", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--h-grid", type=str, default=None)
    ap.add_argument("--resample-grid", type=str, default=None)
    ap.add_argument("--move-pct-grid", type=str, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--output", action="store_true")
    args = ap.parse_args()

    h_grid = [float(x) for x in args.h_grid.split(",")] if args.h_grid else DEFAULT_H_GRID
    resample_grid = [int(x) for x in args.resample_grid.split(",")] if args.resample_grid else DEFAULT_RESAMPLE_GRID
    move_pct_grid = [float(x) for x in args.move_pct_grid.split(",")] if args.move_pct_grid else DEFAULT_MOVE_PCT_GRID
    device = get_device(args.device)

    cfg = SweepConfig(
        gate_id="gate0b",  # reuse Gate 0B extraction cache
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
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"gate1b_sweep_{stamp}.txt"
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

    print("\nGate 1B Robustness Sweep")
    print(f"  Run:          {run_ts}")
    print(f"  Days:         {len(dbn_files)}  |  select={'far' if cfg.select_far else 'near'}")
    print(f"  Workers:      {cfg.workers}  |  device={cfg.device}")
    print(f"  H grid:       {cfg.h_grid}")
    print(f"  Resample:     {cfg.resample_grid} min")
    print(f"  Move-pct:     {cfg.move_pct_grid}")
    print(f"  Methods:      {DEFAULT_GATE1_METHODS}")

    records = None if no_cache else load_cache(cfg)
    if records is None:
        _log(f"Extracting {len(dbn_files)} days with {cfg.workers} workers ...")
        records = _extract_gate0b(cfg, dbn_files)
        save_cache(cfg, records)
    if not records:
        sys.exit("No records extracted.")

    rows = []
    n_cells = len(cfg.h_grid) * len(cfg.resample_grid) * len(cfg.move_pct_grid)
    cell_idx = 0
    for H in cfg.h_grid:
        for resample in cfg.resample_grid:
            for move_pct in cfg.move_pct_grid:
                cell_idx += 1
                t_cell = time.time()
                state = {"completed": 0, "current": f"H={H:.2f} resample={resample} move_pct={move_pct:.2f}", "start_ts": t_cell}
                _log(f"Cell {cell_idx:3d}/{n_cells:3d} — starting  H={H:.2f}  resample={resample:4d} min  move_pct={move_pct:.0%}")
                with extraction_heartbeat(1, state, label="Evaluating Gate 1B cell", interval_s=5.0):
                    avg_act, avg_qui, cell_rows = _evaluate_cell(records, H, resample, move_pct, cfg.select_far)
                    state["completed"] = 1
                best_imp, best_method, best_feat = _best_carry_improvement(
                    avg_act, methods=list(DEFAULT_GATE1_METHODS))
                verdict = classify_gate1b_cell(avg_act if avg_act else None, avg_qui if avg_qui else None)
                _log(f"Cell {cell_idx:3d}/{n_cells:3d}  H={H:.2f}  resample={resample:4d} min  move_pct={move_pct:.0%}  "
                     f"→ {verdict:<8s}  active_best={best_method or 'n/a'}:{best_feat or 'n/a'} "
                     f"({best_imp:+.1%})  [{time.time() - t_cell:.1f}s]")
                rows.extend(cell_rows)

    if not rows:
        sys.exit("Gate 1B sweep produced no rows.")

    results_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"gate1b_sweep_{stamp}.csv"
    results_df.to_csv(csv_path, index=False)
    _log(f"CSV saved → {csv_path}")
    for move_pct in cfg.move_pct_grid:
        print(format_gate1b_summary(results_df, move_pct))
    print()


if __name__ == "__main__":
    main()
