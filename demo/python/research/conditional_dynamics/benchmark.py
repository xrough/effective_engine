#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
conditional_dynamics/benchmark.py
==================================
Experiment: Conditional smile response after large spot moves

Research question (Gate 3: Regime Dynamics):
    Does rough-vol structure add value for 1-step-ahead smile forecasting
    conditional on large spot moves — the regime where rough vol is most
    expected to differ from diffusion?

Hypothesis:
    If the rough model has any practical dynamic edge, it should show up
    in the ACTIVE regime (large |spot return|) and be absent in the QUIET
    regime (small |spot return|). Finding an advantage only unconditionally
    is not a useful signal — it's just momentum.

Design
------
1. Load data via process_day_full (includes 'forward' field).
2. Pool exact expiries into tenor-role series (front/mid/back by DTE bucket).
3. For each tenor series: compute r_t = log(F_t / F_{t-1}).
4. Tag regime: ACTIVE if |r_t| > top move_pct-percentile threshold; else QUIET.
5. Run evaluate_forecasts (carry / rough / AR1) on the full series, then
   score:
   - full sample
   - quiet target bars
   - active target bars
6. Compare RMSE, directional accuracy, residual ACF across regimes.

Success criteria (PROCEED):
  In the ACTIVE regime ONLY:
    - rough RMSE < carry RMSE on rr25 or bf25, OR
    - residual ACF(1) < −0.05 (mean-reversion)
  AND this advantage is NOT present in the quiet regime.

Usage
-----
  python benchmark.py [--days N] [--h H] [--quick] [--resample N]
                      [--far] [--move-pct 0.20] [--output]

  --resample N   Bar size in minutes (default: 1). Use 120 for 2-hour bars.
  --far          Track furthest expiries in DTE window (auto-enabled for ≥60 min)
  --move-pct P   Top-P fraction of |spot returns| defines ACTIVE regime (default 0.20)
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

import numpy as np
import pandas as pd

# ── shared pipeline ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "shared"))
from smile_pipeline import (
    RATE, DIV, MIN_DTE, MAX_DTE, MKT_OPEN_UTC, MKT_CLOSE_UTC,
    _Tee, process_day_full, evaluate_forecasts, ar1_forecast, get_device,
)
from robustness_sweeps import extraction_heartbeat, resample_panel, pool_by_tenor_bucket


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── parallel worker (module-level for pickling with spawn context) ─────────────
def _day_worker(args):
    """Open a fresh ZipFile per process — ZipFile handles are not picklable."""
    zip_path, fname, trade_date, H, device, min_dte, max_dte = args
    import zipfile as _zf
    try:
        with _zf.ZipFile(zip_path) as zf:
            return (trade_date,
                    process_day_full(zf, fname, trade_date, H, device, min_dte, max_dte))
    except Exception:
        return (trade_date, [])

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = _HERE.parents[4]   # .../MVP
OPRA_ZIP = ROOT / "demo" / "data" / "OPRA-20260208-3T68RYYKF9.zip"
OUT_DIR  = _HERE.parent / "output"

N_EXP = 2   # expiries to track per bar (same as intraday benchmark)


# ─────────────────────────────────────────────────────────────────────────────
# Regime tagging
# ─────────────────────────────────────────────────────────────────────────────

def tag_move_regime(ts_data: list, move_pct: float = 0.20) -> list:
    """
    Add 'regime' field ('active' | 'quiet') to each record.

    Spot return: r_t = log(F_t / F_{t-1}) using the 'forward' field.
    Regime threshold: |r_t| > percentile(|r|, 100*(1-move_pct)).

    Records at the first bar (no prior forward) are tagged 'quiet'.
    """
    ts_data = sorted(ts_data, key=lambda r: r["ts"])
    n = len(ts_data)

    # Compute log-returns from consecutive forwards
    abs_rets = []
    for i in range(1, n):
        F_prev = ts_data[i-1].get("forward", float("nan"))
        F_curr = ts_data[i].get("forward", float("nan"))
        if F_prev > 0 and F_curr > 0:
            abs_rets.append(abs(math.log(F_curr / F_prev)))
        else:
            abs_rets.append(float("nan"))

    finite_rets = [r for r in abs_rets if math.isfinite(r)]
    if not finite_rets:
        for r in ts_data:
            r["regime"] = "quiet"
        return ts_data

    threshold = float(np.percentile(finite_rets, 100 * (1 - move_pct)))

    ts_data[0]["regime"] = "quiet"   # no prior bar
    for i in range(1, n):
        ar = abs_rets[i-1]
        ts_data[i]["regime"] = "active" if (math.isfinite(ar) and ar > threshold) else "quiet"

    return ts_data


def evaluate_conditional(ts_data: list, H: float,
                          move_pct: float = 0.20,
                          min_bars: int = 15) -> dict:
    """
    Tag a full tenor series by regime, then score forecasts on the masked bars.

    Forecasts are always formed on the full series. ACTIVE/QUIET only control
    which target bars are scored, so the rough forecaster still sees the quiet
    history needed to stabilize alpha/gamma.

    Returns {
        'all':    evaluate_forecasts result or None,
        'quiet':  evaluate_forecasts result or None,
        'active': evaluate_forecasts result or None,
        'regime_counts': {'quiet': int, 'active': int},
        'threshold': float,
    }
    """
    tagged = tag_move_regime(list(ts_data), move_pct)

    quiet_mask  = [r.get("regime") == "quiet" for r in tagged]
    active_mask = [r.get("regime") == "active" for r in tagged]

    # Compute threshold for reporting
    abs_rets = []
    for i in range(1, len(tagged)):
        F_p = tagged[i-1].get("forward", float("nan"))
        F_c = tagged[i].get("forward", float("nan"))
        if F_p > 0 and F_c > 0:
            abs_rets.append(abs(math.log(F_c / F_p)))
    finite_rets = [r for r in abs_rets if math.isfinite(r)]
    threshold = float(np.percentile(finite_rets, 100*(1-move_pct))) if finite_rets else float("nan")

    def safe_eval(mask):
        return evaluate_forecasts(tagged, H, score_mask=mask) if sum(mask) >= min_bars else None

    return {
        "all":            evaluate_forecasts(tagged, H),
        "quiet":          safe_eval(quiet_mask),
        "active":         safe_eval(active_mask),
        "regime_counts":  {"quiet": sum(quiet_mask), "active": sum(active_mask)},
        "threshold":      threshold,
        "scoring_mode":   "full-history forecasts, masked regime scoring",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _avg(lst):
    return float(np.mean(lst)) if lst else float("nan")

def _summarise_eval(ev: dict | None, label: str):
    """Print a compact summary of an evaluate_forecasts result."""
    if ev is None:
        print(f"  [{label}]  insufficient bars (<15)")
        return
    meta = ev.get("_meta", {})
    n    = meta.get("n_bars", "?")
    print(f"  [{label}]  n={n} bars")

    # RMSE
    print(f"    RMSE:     {'Feature':<16} {'Carry':>8} {'Rough':>8} {'AR1':>8}  vs-carry")
    for feat in ["rr25","bf25","atm_total_var"]:
        if feat not in ev: continue
        c = ev[feat]["carry"]["rmse"]
        r = ev[feat]["rough"]["rmse"]
        a = ev[feat]["ar1"]["rmse"]
        pct = (r-c)/c*100 if math.isfinite(c) and c>0 else float("nan")
        best = "rough" if r<c and r<a else ("carry" if c<=r and c<=a else "ar1")
        print(f"              {feat:<16} {c:>8.5f} {r:>8.5f} {a:>8.5f}  {pct:>+7.1f}% ({best})")

    # ACF
    rr_acf  = meta.get("rr25_resid_acf",  float("nan"))
    rr_pval = meta.get("rr25_resid_pval", float("nan"))
    bf_acf  = meta.get("bf25_resid_acf",  float("nan"))
    bf_pval = meta.get("bf25_resid_pval", float("nan"))
    def acf_label(acf):
        if not math.isfinite(acf): return "n/a"
        return "◀ mean-reverts" if acf < -0.05 else "▶ momentum"
    print(f"    ACF(1):   rr25={rr_acf:+.4f} (p={rr_pval:.4f}) {acf_label(rr_acf)}")
    print(f"              bf25={bf_acf:+.4f} (p={bf_pval:.4f}) {acf_label(bf_acf)}")

    # Dir accuracy
    print(f"    Dir-acc:  ", end="")
    parts = []
    for feat in ["rr25","bf25"]:
        if feat not in ev: continue
        c = ev[feat]["carry"]["dir"]*100
        r = ev[feat]["rough"]["dir"]*100
        parts.append(f"{feat}: carry={c:.1f}% rough={r:.1f}%")
    print("  |  ".join(parts))
    print()


def print_report(all_cond_results: dict, H: float, resample: int,
                 move_pct: float, run_meta: dict | None = None):
    W = 72
    bar_label = f"{resample}-min" if resample > 1 else "1-min"

    print("\n" + "=" * W)
    print("  CONDITIONAL SMILE DYNAMICS — Spot Move Regime Benchmark")
    print("  Experiment: conditional_dynamics")
    print(f"  H={H:.2f}  bar={bar_label}  active_regime=top {move_pct*100:.0f}% |Δspot|")
    print("=" * W)

    if run_meta:
        print("\n── 0. Experiment metadata ──────────────────────────────────────────\n")
        print(f"  Run:              {run_meta.get('run_ts','?')}")
        print(f"  Trading days:     {run_meta.get('days','?')}"
              f"  ({run_meta.get('first_date','?')} → {run_meta.get('last_date','?')})")
        print(f"  Tenor selection:  {run_meta.get('select_mode','?')}")
        print(f"  RATE={RATE:.3f}  DIV={DIV:.3f}")

    print("\n── 1. Per-tenor conditional results ────────────────────────────────\n")
    print("  Hypothesis: rough edge (if any) shows in ACTIVE regime, not QUIET.\n")

    # Collect aggregate verdict signals
    active_rough_wins_rr25 = []
    active_rough_wins_bf25 = []
    active_mean_reverts    = []
    quiet_rough_wins_rr25  = []

    for exp_str, cond in sorted(all_cond_results.items()):
        rc = cond["regime_counts"]
        thresh = cond.get("threshold", float("nan"))
        print(f"  Series {exp_str}  "
              f"(quiet={rc['quiet']}, active={rc['active']}, "
              f"threshold=|r|>{thresh:.5f})")

        _summarise_eval(cond.get("all"),    "ALL   ")
        _summarise_eval(cond.get("quiet"),  "QUIET ")
        _summarise_eval(cond.get("active"), "ACTIVE")
        print()

        # Collect for aggregate verdict
        for ev, store_list in [
            (cond.get("active"), None),  # handled below
            (cond.get("quiet"),  None),
        ]:
            pass

        act_ev = cond.get("active")
        qui_ev = cond.get("quiet")
        if act_ev:
            c_rr = act_ev.get("rr25",{}).get("carry",{}).get("rmse", float("nan"))
            r_rr = act_ev.get("rr25",{}).get("rough",{}).get("rmse", float("nan"))
            c_bf = act_ev.get("bf25",{}).get("carry",{}).get("rmse", float("nan"))
            r_bf = act_ev.get("bf25",{}).get("rough",{}).get("rmse", float("nan"))
            active_rough_wins_rr25.append(r_rr < c_rr)
            active_rough_wins_bf25.append(r_bf < c_bf)
            rr_acf = act_ev.get("_meta",{}).get("rr25_resid_acf", float("nan"))
            bf_acf = act_ev.get("_meta",{}).get("bf25_resid_acf", float("nan"))
            active_mean_reverts.append(
                (math.isfinite(rr_acf) and rr_acf < -0.05) or
                (math.isfinite(bf_acf) and bf_acf < -0.05))
        if qui_ev:
            c_rr = qui_ev.get("rr25",{}).get("carry",{}).get("rmse", float("nan"))
            r_rr = qui_ev.get("rr25",{}).get("rough",{}).get("rmse", float("nan"))
            quiet_rough_wins_rr25.append(r_rr < c_rr)

    # ── Aggregate verdict ────────────────────────────────────────────────────
    print("── 2. Aggregate verdict ────────────────────────────────────────────\n")

    n_exp = len(all_cond_results)
    frac_act_rr = sum(active_rough_wins_rr25)/len(active_rough_wins_rr25) if active_rough_wins_rr25 else float("nan")
    frac_act_bf = sum(active_rough_wins_bf25)/len(active_rough_wins_bf25) if active_rough_wins_bf25 else float("nan")
    frac_rev    = sum(active_mean_reverts)/len(active_mean_reverts)       if active_mean_reverts    else float("nan")
    frac_qui_rr = sum(quiet_rough_wins_rr25)/len(quiet_rough_wins_rr25)   if quiet_rough_wins_rr25  else float("nan")

    def pct(f): return f"{f*100:.0f}%" if math.isfinite(f) else "n/a"

    print(f"  Tenor series evaluated: {n_exp}")
    print(f"  ACTIVE regime — rough beats carry on rr25:  {pct(frac_act_rr)}")
    print(f"  ACTIVE regime — rough beats carry on bf25:  {pct(frac_act_bf)}")
    print(f"  ACTIVE regime — residuals mean-revert:      {pct(frac_rev)}")
    print(f"  QUIET  regime — rough beats carry on rr25:  {pct(frac_qui_rr)}")
    print()

    # Verdict logic
    active_edge = (
        (math.isfinite(frac_act_rr) and frac_act_rr > 0.5) or
        (math.isfinite(frac_act_bf) and frac_act_bf > 0.5) or
        (math.isfinite(frac_rev)    and frac_rev    > 0.5)
    )
    quiet_edge = math.isfinite(frac_qui_rr) and frac_qui_rr > 0.5

    if active_edge and not quiet_edge:
        verdict = "PROCEED"
        detail  = ("Rough-vol structure adds value specifically in stressed "
                   "(large spot move) regimes and not in quiet periods. "
                   "Conditional dynamic edge is present.")
    elif active_edge and quiet_edge:
        verdict = "WEAK — unconditional (not regime-specific)"
        detail  = ("Rough beats carry in both regimes. The advantage is "
                   "unconditional — not a regime-specific signal.")
    elif not active_edge and not quiet_edge:
        verdict = f"ABORT — no rough-vol edge at {bar_label} frequency"
        detail  = ("Carry dominates in both regimes. Rough structure adds "
                   "no conditional forecasting power after large spot moves.")
    else:
        verdict = "ABORT — edge only in quiet regime (anomalous)"
        detail  = ("Rough outperforms only in quiet regime. This is inconsistent "
                   "with the rough-vol mechanism and likely noise.")

    print(f"  >>> {verdict}")
    print(f"      {detail}")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Conditional Smile Dynamics benchmark (conditional_dynamics)")
    ap.add_argument("--days",     type=int,   default=None)
    ap.add_argument("--h",        type=float, default=0.1)
    ap.add_argument("--quick",    action="store_true", help="--days 5")
    ap.add_argument("--resample", type=int,   default=1,
                    help="Bar size in minutes (default: 1)")
    ap.add_argument("--far",      action="store_true",
                    help="Track furthest expiries (auto-enabled for ≥60 min)")
    ap.add_argument("--move-pct", type=float, default=0.20,
                    help="Top fraction of |spot returns| defining ACTIVE regime (default 0.20)")
    ap.add_argument("--workers",  type=int,   default=1,
                    help="Parallel day-processing workers (default: 1)")
    ap.add_argument("--device",   type=str,   default="auto",
                    choices=["auto","cpu","cuda","mps"],
                    help="IV computation device (default: auto)")
    ap.add_argument("--output",   action="store_true",
                    help="Save full run log to output/<timestamp>.txt")
    args = ap.parse_args()

    H          = args.h
    resample   = args.resample
    select_far = args.far or (resample >= 60)
    n_days     = 5 if args.quick else (args.days or None)
    move_pct   = args.move_pct
    device     = get_device(args.device)

    if not OPRA_ZIP.exists():
        sys.exit(f"OPRA zip not found: {OPRA_ZIP}")

    buf = None
    if args.output:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        sys.stdout = _Tee(sys.__stdout__, buf)

    try:
        _run(H, resample, select_far, n_days, move_pct, args.workers, device)
    finally:
        if buf is not None:
            sys.stdout = sys.__stdout__
            bar_str  = f"{resample}min"
            stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"conditional_{bar_str}_{stamp}.txt"
            out_path.write_text(buf.getvalue())
            print(f"\nReport saved → {out_path}")


def _run(H: float, resample: int, select_far: bool, n_days: int | None,
         move_pct: float, workers: int = 1, device: str = "cpu"):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with zipfile.ZipFile(OPRA_ZIP) as zf_index:
        dbn_files = sorted(f for f in zf_index.namelist() if f.endswith(".dbn.zst"))
    if n_days:
        dbn_files = dbn_files[:n_days]

    bar_label  = f"{resample}-min" if resample > 1 else "1-min"
    sel_label  = "far" if select_far else "near"

    print(f"\nconditional_dynamics benchmark")
    print(f"  Run:      {run_ts}")
    print(f"  Days:     {len(dbn_files)}  |  H={H}  |  bar={bar_label}")
    print(f"  Select:   {sel_label} tenor buckets  |  active=top {move_pct*100:.0f}% |Δspot|")
    print(f"  Device:   {device}  |  Workers: {workers}")
    print(f"  DTE:      {MIN_DTE}–{MAX_DTE}  |  pooled by tenor role\n")

    dates_order = []
    for fname in dbn_files:
        ds    = re.search(r"(\d{8})", fname).group(1)
        tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        dates_order.append((fname, tdate))

    task_args = [
        (str(OPRA_ZIP), fname, tdate, H, device, MIN_DTE, MAX_DTE)
        for fname, tdate in dates_order
    ]

    total = len(task_args)
    t0    = time.time()
    day_map: dict = {}
    progress = {"completed": 0, "current": None, "start_ts": t0}

    with extraction_heartbeat(total, progress, label="Extracting conditional days"):
        if workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers) as pool:
                for tdate, recs in pool.imap_unordered(_day_worker, task_args):
                    progress["completed"] += 1
                    _log(f"Day {progress['completed']:3d}/{total:3d} — {tdate} — {len(recs)} records")
                    day_map[tdate] = recs
        else:
            with zipfile.ZipFile(OPRA_ZIP) as zf:
                for i, (fname, tdate) in enumerate(dates_order, 1):
                    progress["current"] = tdate
                    _log(f"Day {i:3d}/{total:3d} — {tdate} — starting")
                    try:
                        recs = process_day_full(zf, fname, tdate, H, device, MIN_DTE, MAX_DTE)
                    except Exception as e:
                        _log(f"Day {i:3d}/{total:3d} — {tdate} — SKIP: {e}")
                        recs = []
                    else:
                        _log(f"Day {i:3d}/{total:3d} — {tdate} — {len(recs)} records")
                    day_map[tdate] = recs
                    progress["completed"] = i
                    progress["current"] = None

    elapsed = time.time() - t0
    total_extracted = sum(len(v) for v in day_map.values())
    _log(f"Extraction done: {total} days, {total_extracted} records  (elapsed {elapsed:.0f}s)")

    flat_records: list[dict] = []
    dates_seen = []

    for fname, tdate in dates_order:
        recs = day_map.get(tdate, [])
        dates_seen.append(tdate)
        flat_records.extend(recs)

    if not flat_records:
        sys.exit("No valid records.")

    panel = resample_panel(flat_records, resample)
    if resample > 1:
        print(f"  Resampled to {resample}-min bars: {len(panel)} bar-expiry records")

    pooled = pool_by_tenor_bucket(panel, select_far=select_far)
    if not pooled:
        sys.exit("No valid tenor-bucket records after pooling.")

    all_series_ts: dict[str, list] = defaultdict(list)
    for r in pooled:
        all_series_ts[str(r["series_id"])].append(r)

    min_bars = max(20, 60 // max(resample, 1))
    print(f"\nEvaluating {len(all_series_ts)} tenor series "
          f"(min {min_bars} bars, min 15 bars per regime) ...")

    all_cond_results = {}
    for series_id, ts in sorted(all_series_ts.items()):
        ts.sort(key=lambda r: r["ts"])
        if len(ts) < min_bars:
            continue
        cond = evaluate_conditional(ts, H, move_pct)
        all_cond_results[str(series_id)] = cond
        rc   = cond["regime_counts"]
        print(f"  {series_id}: {len(ts)} bars total  "
              f"(quiet={rc['quiet']}, active={rc['active']})")

    if not all_cond_results:
        sys.exit(f"No tenor series had ≥{min_bars} bars.")

    run_meta = {
        "run_ts":      run_ts,
        "days":        len(dbn_files),
        "first_date":  str(min(dates_seen)) if dates_seen else "?",
        "last_date":   str(max(dates_seen)) if dates_seen else "?",
        "select_mode": sel_label,
    }
    print_report(all_cond_results, H, resample, move_pct, run_meta)


if __name__ == "__main__":
    main()
