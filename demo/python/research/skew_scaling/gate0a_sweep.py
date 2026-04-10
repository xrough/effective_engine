#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
Gate 0A Robustness Sweep
========================
Scores cross-sectional rr25 maturity scaling across a grid of:

  H × DTE-window × subperiod

Workflow
--------
1. Extract all-expiry smile records once over the broadest requested DTE envelope.
2. For each DTE-window and subperiod:
   a. group records by timestamp
   b. fit log|rr25| = a + beta log(T) whenever at least 3 expiries survive
3. For each H in the grid:
   a. score the fit distribution against theory beta = H
   b. classify the cell as PROCEED / WEAK / ABORT
4. Write gate0a_sweep_<timestamp>.csv and print summary grids.

Usage
-----
  python gate0a_sweep.py [--days N] [--workers N] [--device STR]
                         [--h-grid "0.05,0.10,0.20"]
                         [--dte-grid "7-21,14-30,21-45,30-60,7-60"]
                         [--subperiod-mode none|halves]
                         [--no-cache] [--output]
"""

from __future__ import annotations

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
ROOT = _HERE.parents[4]   # .../MVP
OPRA_ZIP = ROOT / "demo" / "data" / "OPRA-20260208-3T68RYYKF9.zip"
OUT_DIR = _HERE.parent / "output"

sys.path.insert(0, str(_HERE.parents[1] / "shared"))
from smile_pipeline import RATE, DIV, _Tee, get_device
from robustness_sweeps import (
    SweepConfig,
    DEFAULT_H_GRID,
    extraction_heartbeat,
    load_cache,
    save_cache,
    _log,
)

sys.path.insert(0, str(_HERE.parent))
from benchmark import _day_worker, fit_skew_scaling, summarize_skew_fits

DEFAULT_DTE_GRID = [(7, 21), (14, 30), (21, 45), (30, 60), (7, 60)]


def format_dte_window(window: tuple[int, int]) -> str:
    return f"{window[0]}-{window[1]}"


def parse_dte_grid(spec: str | None) -> list[tuple[int, int]]:
    """Parse '7-21,14-30' style DTE grid strings."""
    if not spec:
        return list(DEFAULT_DTE_GRID)

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if not m:
            raise ValueError(f"Invalid DTE window '{token}'. Expected e.g. 7-21")
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= 0 or hi <= 0 or lo >= hi:
            raise ValueError(f"Invalid DTE bounds '{token}'. Require 0 < lo < hi")
        window = (lo, hi)
        if window not in seen:
            out.append(window)
            seen.add(window)

    if not out:
        raise ValueError("Empty DTE grid after parsing")
    return out


def build_subperiod_slices(records: list[dict], mode: str) -> list[dict]:
    """
    Build full-sample and optional split-date subperiod slices.

    Returns rows with keys:
      label, dates, start_date, end_date
    """
    dates = sorted({r["ts"].date() for r in records})
    if not dates:
        return [{"label": "full", "dates": set(), "start_date": None, "end_date": None}]

    out = [{
        "label": "full",
        "dates": set(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
    }]
    if mode == "none" or len(dates) < 2:
        return out

    split = len(dates) // 2
    first = dates[:split]
    second = dates[split:]
    if first and second:
        out.append({
            "label": "first_half",
            "dates": set(first),
            "start_date": first[0],
            "end_date": first[-1],
        })
        out.append({
            "label": "second_half",
            "dates": set(second),
            "start_date": second[0],
            "end_date": second[-1],
        })
    return out


def _extract(cfg: SweepConfig,
             dbn_files: list[tuple[str, date]],
             min_dte: int,
             max_dte: int) -> list[dict]:
    """Extract all-expiry records for the broadest requested DTE envelope."""
    total = len(dbn_files)
    task_args = [
        (str(OPRA_ZIP), fname, tdate, 0.10, cfg.device, min_dte, max_dte)
        for fname, tdate in dbn_files
    ]

    t0 = time.time()
    day_map: dict[date, list[dict]] = {}
    progress = {"completed": 0, "current": None, "start_ts": t0}

    with extraction_heartbeat(total, progress, label="Extracting Gate 0A days"):
        if cfg.workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(cfg.workers) as pool:
                for tdate, recs in pool.imap_unordered(_day_worker, task_args):
                    progress["completed"] += 1
                    _log(f"Day {progress['completed']:3d}/{total:3d} — {tdate} — {len(recs)} records")
                    day_map[tdate] = recs
        else:
            for i, (fname, tdate) in enumerate(dbn_files, 1):
                progress["current"] = tdate
                _log(f"Day {i:3d}/{total:3d} — {tdate} — starting")
                try:
                    recs = _day_worker((str(OPRA_ZIP), fname, tdate, 0.10,
                                        cfg.device, min_dte, max_dte))[1]
                except Exception as e:
                    _log(f"Day {i:3d}/{total:3d} — {tdate} — SKIP: {e}")
                    recs = []
                else:
                    _log(f"Day {i:3d}/{total:3d} — {tdate} — {len(recs)} records")
                day_map[tdate] = recs
                progress["completed"] = i
                progress["current"] = None

    flat = [r for _, recs in sorted(day_map.items()) for r in recs]
    elapsed = time.time() - t0
    _log(f"Extraction done: {total} days, {len(flat)} records  (elapsed {elapsed:.0f}s)")
    return flat


def _fit_filtered_records(filtered_records: list[dict]) -> tuple[int, list[dict]]:
    """Run per-timestamp skew regressions on an already filtered record list."""
    by_ts: dict = defaultdict(list)

    for rec in filtered_records:
        by_ts[rec["ts"]].append(rec)

    fit_results: list[dict] = []
    for ts in sorted(by_ts):
        result = fit_skew_scaling(by_ts[ts])
        if result is not None:
            fit_results.append(result)

    return len(by_ts), fit_results


def evaluate_gate0a_cell(records: list[dict],
                         H: float,
                         dte_window: tuple[int, int],
                         allowed_dates: set[date] | None = None) -> dict:
    """Evaluate one Gate 0A sweep cell."""
    filtered_records = [
        rec for rec in records
        if (allowed_dates is None or rec["ts"].date() in allowed_dates)
        and dte_window[0] / 365.0 <= rec.get("T", -1.0) <= dte_window[1] / 365.0
    ]
    n_ts_total, fit_results = _fit_filtered_records(filtered_records)
    summary = summarize_skew_fits(fit_results, n_ts_total, H)
    summary["status"] = gate0a_status(summary.get("verdict", "ABORT"))
    summary["coverage"] = (
        summary["n_ts_fitted"] / summary["n_ts_total"]
        if summary.get("n_ts_total", 0) else 0.0
    )
    theory = summary.get("theory_beta", H)
    beta_median = summary.get("beta_median", float("nan"))
    summary["beta_gap"] = beta_median - theory if pd.notna(beta_median) else float("nan")
    summary["implied_h"] = beta_median
    return summary


def gate0a_status(verdict: str) -> str:
    if "PROCEED" in verdict:
        return "PROCEED"
    if "WEAK" in verdict:
        return "WEAK"
    return "ABORT"


def format_gate0a_summary(results_df: pd.DataFrame, subperiod: str) -> str:
    """Render one verdict grid for a chosen subperiod."""
    df = results_df[results_df["subperiod"] == subperiod]
    if df.empty:
        return f"(no Gate 0A rows for subperiod={subperiod})"

    pivot = df.pivot_table(
        index="H", columns="dte_window", values="status", aggfunc="first"
    )
    col_w = max(10, *(len(str(c)) for c in pivot.columns))
    h_w = 6
    sep = "-" * (h_w + 3 + (col_w + 3) * len(pivot.columns))

    lines = [
        f"\nGate 0A Robustness Sweep — {subperiod}",
        f"{'H':>{h_w}} | " + " | ".join(f"{c:>{col_w}}" for c in pivot.columns),
        sep,
    ]

    n_proceed = 0
    n_total = 0
    for h_val, row in pivot.iterrows():
        cells = []
        for val in row:
            status = str(val) if pd.notna(val) else "ABORT"
            cells.append(f"{status:>{col_w}}")
            if status == "PROCEED":
                n_proceed += 1
            if status in {"PROCEED", "WEAK", "ABORT"}:
                n_total += 1
        lines.append(f"{h_val:>{h_w}.2f} | " + " | ".join(cells))

    pct = 100.0 * n_proceed / n_total if n_total else 0.0
    lines.append(sep)
    lines.append(f"\nPROCEED cells: {n_proceed}/{n_total} ({pct:.0f}%)")
    if pct >= 50.0:
        lines.append("→ Cross-sectional rough scaling looks robust across this grid.")
    elif pct >= 25.0:
        lines.append("→ Partial robustness; some (H, DTE) regions support the scaling law.")
    else:
        lines.append("→ Robust support is limited across this grid.")
    return "\n".join(lines)


def format_gate0a_subperiod_consistency(results_df: pd.DataFrame) -> str:
    """Summarize whether full-sample PROCEED cells survive both half-sample splits."""
    needed = {"full", "first_half", "second_half"}
    if not needed.issubset(set(results_df["subperiod"].unique())):
        return ""

    def _slice(label: str, suffix: str) -> pd.DataFrame:
        cols = ["H", "dte_window", "status", "beta_median"]
        df = results_df[results_df["subperiod"] == label][cols].copy()
        return df.rename(columns={
            "status": f"status_{suffix}",
            "beta_median": f"beta_median_{suffix}",
        })

    merged = _slice("full", "full")
    merged = merged.merge(_slice("first_half", "first"), on=["H", "dte_window"], how="left")
    merged = merged.merge(_slice("second_half", "second"), on=["H", "dte_window"], how="left")

    full_pass = merged[merged["status_full"] == "PROCEED"]
    if full_pass.empty:
        return "\nSubperiod consistency: no full-sample PROCEED cells to check."

    confirmed = full_pass[
        full_pass["status_first"].isin(["PROCEED", "WEAK"])
        & full_pass["status_second"].isin(["PROCEED", "WEAK"])
    ]
    return (
        "\nSubperiod consistency: "
        f"{len(confirmed)}/{len(full_pass)} full-sample PROCEED cells "
        "remain non-ABORT in both halves."
    )


def main():
    ap = argparse.ArgumentParser(description="Gate 0A Robustness Sweep")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--h-grid", type=str, default=None,
                    help='Comma-separated H values, e.g. "0.05,0.10,0.20"')
    ap.add_argument("--dte-grid", type=str, default=None,
                    help='Comma-separated DTE windows, e.g. "7-21,14-30,21-45,30-60,7-60"')
    ap.add_argument("--subperiod-mode", type=str, default="halves",
                    choices=["none", "halves"])
    ap.add_argument("--no-cache", action="store_true",
                    help="Force re-extraction even if cache exists")
    ap.add_argument("--output", action="store_true",
                    help="Save report to output/gate0a_sweep_<timestamp>.txt")
    args = ap.parse_args()

    h_grid = ([float(x) for x in args.h_grid.split(",")]
              if args.h_grid else list(DEFAULT_H_GRID))
    dte_grid = parse_dte_grid(args.dte_grid)
    dte_envelope = (min(lo for lo, _ in dte_grid), max(hi for _, hi in dte_grid))
    device = get_device(args.device)

    cfg = SweepConfig(
        gate_id=f"gate0a_{dte_envelope[0]}_{dte_envelope[1]}",
        h_grid=h_grid,
        n_days=args.days,
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
        _run(cfg, dte_grid=dte_grid, subperiod_mode=args.subperiod_mode,
             no_cache=args.no_cache)
    finally:
        if buf is not None:
            sys.stdout = sys.__stdout__
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"gate0a_sweep_{stamp}.txt"
            out_path.write_text(buf.getvalue())
            print(f"\nReport saved → {out_path}")


def _run(cfg: SweepConfig,
         dte_grid: list[tuple[int, int]],
         subperiod_mode: str,
         no_cache: bool = False):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with zipfile.ZipFile(OPRA_ZIP) as zf_index:
        dbn_files_raw = sorted(f for f in zf_index.namelist() if f.endswith(".dbn.zst"))
    if cfg.n_days:
        dbn_files_raw = dbn_files_raw[:cfg.n_days]

    dbn_files: list[tuple[str, date]] = []
    for fname in dbn_files_raw:
        ds = re.search(r"(\d{8})", fname).group(1)
        tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        dbn_files.append((fname, tdate))

    n_sub = 1 if subperiod_mode == "none" else 3
    n_cells = len(cfg.h_grid) * len(dte_grid) * n_sub

    print("\nGate 0A Robustness Sweep")
    print(f"  Run:          {run_ts}")
    print(f"  Days:         {len(dbn_files)}")
    print(f"  Workers:      {cfg.workers}  |  device={cfg.device}")
    print(f"  H grid:       {cfg.h_grid}")
    print(f"  DTE windows:  {[format_dte_window(w) for w in dte_grid]}")
    print(f"  Subperiods:   {subperiod_mode}")
    print(f"  Total cells:  {n_cells}")

    records = None if no_cache else load_cache(cfg)
    if records is None:
        env_lo = min(lo for lo, _ in dte_grid)
        env_hi = max(hi for _, hi in dte_grid)
        _log(f"Extracting {len(dbn_files)} days for DTE envelope {env_lo}-{env_hi} ...")
        records = _extract(cfg, dbn_files, env_lo, env_hi)
        save_cache(cfg, records)

    if not records:
        sys.exit("No records extracted. Check data path and DTE selection.")

    slices = build_subperiod_slices(records, subperiod_mode)
    _log(f"Starting Gate 0A sweep: {n_cells} cells "
         f"({len(cfg.h_grid)} H × {len(dte_grid)} DTE × {len(slices)} subperiods)")

    all_rows: list[dict] = []
    cell_idx = 0

    for sub in slices:
        for dte_window in dte_grid:
            window_records = [
                rec for rec in records
                if rec["ts"].date() in sub["dates"]
                and dte_window[0] / 365.0 <= rec.get("T", -1.0) <= dte_window[1] / 365.0
            ]
            n_dates = len({rec["ts"].date() for rec in window_records})
            lo, hi = dte_window

            # H does not change the fitted slopes, only the theory comparison.
            n_ts_total, fit_results = _fit_filtered_records(window_records)
            for H in cfg.h_grid:
                cell_idx += 1
                t_cell = time.time()
                cell_state = {
                    "completed": 0,
                    "current": (f"subperiod={sub['label']} "
                                f"dte={format_dte_window(dte_window)} H={H:.2f}"),
                    "start_ts": t_cell,
                }
                _log(f"Cell {cell_idx:3d}/{n_cells:3d} — starting  "
                     f"subperiod={sub['label']:<11s}  dte={lo:2d}-{hi:2d}  H={H:.2f}")
                with extraction_heartbeat(1, cell_state,
                                          label="Evaluating Gate 0A cell",
                                          interval_s=5.0):
                    summary = summarize_skew_fits(fit_results, n_ts_total, H)
                    cell_state["completed"] = 1
                elapsed_c = time.time() - t_cell

                summary["status"] = gate0a_status(summary.get("verdict", "ABORT"))
                summary["coverage"] = (
                    summary["n_ts_fitted"] / summary["n_ts_total"]
                    if summary.get("n_ts_total", 0) else 0.0
                )
                beta_median = summary.get("beta_median", float("nan"))
                theory = summary.get("theory_beta", H)
                summary["beta_gap"] = beta_median - theory if pd.notna(beta_median) else float("nan")
                summary["implied_h"] = beta_median

                row = {
                    "H": H,
                    "dte_min": lo,
                    "dte_max": hi,
                    "dte_window": format_dte_window(dte_window),
                    "subperiod": sub["label"],
                    "subperiod_start": sub["start_date"],
                    "subperiod_end": sub["end_date"],
                    "n_dates": n_dates,
                    "n_records": len(window_records),
                }
                row.update(summary)
                all_rows.append(row)

                _log(f"Cell {cell_idx:3d}/{n_cells:3d}  "
                     f"subperiod={sub['label']:<11s}  dte={lo:2d}-{hi:2d}  H={H:.2f}  "
                     f"→ {row['status']:<8s}  "
                     f"(beta_med={row.get('beta_median', float('nan')):+.4f}  "
                     f"R²={row.get('r2_mean', float('nan')):.4f})  "
                     f"[{elapsed_c:.1f}s]")

    if not all_rows:
        sys.exit("Sweep produced no Gate 0A rows.")

    results_df = pd.DataFrame(all_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"gate0a_sweep_{stamp}.csv"
    results_df.to_csv(csv_path, index=False)
    _log(f"CSV saved → {csv_path}")

    print(format_gate0a_summary(results_df, "full"))
    if subperiod_mode != "none":
        for label in ["first_half", "second_half"]:
            print(format_gate0a_summary(results_df, label))
        print(format_gate0a_subperiod_consistency(results_df))
    print()


if __name__ == "__main__":
    main()
