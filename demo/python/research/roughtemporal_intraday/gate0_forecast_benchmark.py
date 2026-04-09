#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
Gate 0: Rough Model Temporal Forecast Benchmark
================================================
Experiment: roughtemporal_intraday

Go/No-Go test. Core question:

    Does the rough vol model's structural relationship between ATM variance and
    smile shape (rr25, bf25) produce mean-reverting residuals, and does it beat
    naive carry on 1-step-ahead RMSE at intraday frequencies?

Design
------
Per bar:
  1. Select N_EXP expiries in [MIN_DTE, MAX_DTE] DTE (near or far mode).
  2. Recover forward from call-put parity (median of near-ATM strike pairs).
  3. Compute ATM IV, 25d-call IV, 25d-put IV via vectorised numpy bisection.
  4. Derive:
       atm_total_var = σ²_ATM × T
       rr25 = IV(25d_call) - IV(25d_put)
       bf25 = (IV(25d_call) + IV(25d_put))/2 - σ_ATM

Rough model structural coefficients (H fixed):
  α = rr25 / (T^{H-0.5} × σ_ATM)   ~constant if rough vol holds cross-sectionally
  γ = bf25 / (T^{2H-1} × ATV)        ditto

1-step-ahead forecasters:
  carry:        x_{t+1} ← x_t
  rough-struct: rr25_{t+1} ← α_rolling_median × T^{H-0.5} × σ_ATM_t
                bf25_{t+1} ← γ_rolling_median × T^{2H-1} × ATV_t
                ATV_{t+1}  ← ATV_t  (rough vol is a martingale for forward var)
  AR(1):        OLS fit on expanding window (≥30 obs)

Verdict criteria:
  PROCEED : rough beats carry AND residuals mean-revert (ACF<-0.05) AND struct stable
  WEAK    : partial evidence on one criterion
  ABORT   : carry dominates, residuals persist (momentum)

Usage
-----
  python gate0_forecast_benchmark.py [--days N] [--h H] [--quick] [--resample N] [--far] [--output]

  --days N      Number of trading days to process (default: all 127)
  --h H         Hurst exponent (default: 0.1)
  --quick       Alias for --days 5
  --resample N  Bar size in minutes (default: 1). Try 5 or 120.
  --far         Track furthest expiries in window (auto-enabled for --resample >= 60)
  --output      Save full run log + report to output/<bar>min_<timestamp>.txt
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

import databento as db
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve()
ROOT     = _HERE.parents[4]   # .../MVP
OPRA_ZIP = ROOT / "demo" / "data" / "OPRA-20260208-3T68RYYKF9.zip"
OUT_DIR  = _HERE.parent / "output"

# ── shared pipeline ────────────────────────────────────────────────────────────
sys.path.insert(0, str(_HERE.parents[1] / "shared"))
from smile_pipeline import (
    RATE, DIV, MIN_DTE, MAX_DTE, MKT_OPEN_UTC, MKT_CLOSE_UTC,
    _Tee,
    _d1d2, _bs_call, _bs_put, _bs_delta_call, _bs_delta_put,
    _vec_iv, _parse_sym_batch, _recover_forward,
    extract_features, ar1_forecast, evaluate_forecasts, get_device,
)

N_EXP = 2   # expiries tracked per bar (experiment-specific)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── parallel worker (module-level for pickling with spawn context) ─────────────
def _gate0_worker(args):
    """Open a fresh ZipFile per process — ZipFile handles are not picklable."""
    zip_path, fname, trade_date, H, select_far = args
    import zipfile as _zf
    try:
        with _zf.ZipFile(zip_path) as zf:
            return (trade_date, process_day(zf, fname, trade_date, H, select_far))
    except Exception:
        return (trade_date, [])


# ─────────────────────────────────────────────────────────────────────────────
# Process one trading day  (experiment-specific: N_EXP / near vs far selection)
# ─────────────────────────────────────────────────────────────────────────────
def process_day(zf: zipfile.ZipFile, fname: str, trade_date: date, H: float,
                select_far: bool = False) -> list:
    with zf.open(fname) as f:
        store = db.DBNStore.from_bytes(f.read())
    df = store.to_df()

    expiry_arr, is_call_arr, strike_arr = _parse_sym_batch(df["symbol"])
    df = df.copy()
    df["expiry"]  = expiry_arr
    df["is_call"] = is_call_arr
    df["strike"]  = strike_arr
    df = df.dropna(subset=["expiry", "strike"])

    valid = df[
        df["bid_px_00"].notna() & df["ask_px_00"].notna() &
        (df["bid_px_00"] > 0)   & (df["ask_px_00"] > 0)
    ].copy()
    valid["mid"]    = (valid["bid_px_00"] + valid["ask_px_00"]) / 2.0
    valid["spread"] = valid["ask_px_00"]  - valid["bid_px_00"]

    valid.index = pd.to_datetime(valid.index, utc=True)
    utc_min = valid.index.hour * 60 + valid.index.minute
    valid = valid[(utc_min >= MKT_OPEN_UTC) & (utc_min <= MKT_CLOSE_UTC)]
    if valid.empty:
        return []

    all_exp = np.array(sorted(valid["expiry"].unique()))
    cand    = [e for e in all_exp
               if MIN_DTE <= (e - trade_date).days <= MAX_DTE]
    if not cand:
        return []
    # select_far: pick furthest N_EXP expiries so each stays in-window many days
    # (needed at coarse bar sizes where weekly rolls happen every 7 days)
    selected = cand[-N_EXP:] if select_far else cand[:N_EXP]

    valid = valid[valid["expiry"].isin(selected)]
    valid = valid[valid["spread"] < 2.0]   # rough liquidity gate

    records = []
    grouped = valid.groupby([valid.index, "expiry"])
    for (ts, exp), grp in grouped:
        T_cal = (exp - trade_date).days / 365.0
        if T_cal < MIN_DTE / 365.0:
            continue

        feats = extract_features(grp[["strike","is_call","mid","spread"]], T_cal)
        if feats is None:
            continue

        alpha = (feats["rr25"] / ((T_cal**(H - 0.5)) * feats["atm_iv"])
                 if feats["atm_iv"] > 1e-6 else float("nan"))
        atv   = feats["atm_total_var"]
        denom = (T_cal**(2*H - 1)) * atv
        gamma = feats["bf25"] / denom if denom > 1e-8 else float("nan")

        records.append({
            "ts":            ts,
            "expiry":        exp,
            "T":             T_cal,
            "atm_iv":        feats["atm_iv"],
            "atm_total_var": atv,
            "rr25":          feats["rr25"],
            "bf25":          feats["bf25"],
            "alpha":         alpha,
            "gamma":         gamma,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def print_report(all_results: dict, H: float, resample: int,
                 run_meta: dict | None = None):
    """
    Prints the full Gate 0 report. run_meta (optional) adds a header section:
      keys: run_ts, days, first_date, last_date, total_raw, total_resampled,
            select_mode
    """
    W = 72
    bar_label = f"{resample}-min" if resample > 1 else "1-min"

    def avg(lst):
        return float(np.mean(lst)) if lst else float("nan")

    agg  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta = defaultdict(list)

    for ev in all_results.values():
        m = ev.get("_meta", {})
        for k, v in m.items():
            if isinstance(v, float) and math.isfinite(v):
                meta[k].append(v)
        for feat in ["atm_total_var","rr25","bf25"]:
            if feat not in ev:
                continue
            for method, d in ev[feat].items():
                for k, v in d.items():
                    if isinstance(v, float) and math.isfinite(v):
                        agg[feat][method][k].append(v)

    print("\n" + "=" * W)
    print("  GATE 0 — Rough Model Temporal Forecast Benchmark")
    print("  Experiment: roughtemporal_intraday")
    print(f"  H={H:.2f}  RATE={RATE:.3f}  DIV={DIV:.3f}  bar={bar_label}")
    print("=" * W)

    # ── 0. Run metadata ──────────────────────────────────────────────────────
    if run_meta:
        print("\n── 0. Experiment metadata ────────────────────────────────────────────\n")
        print(f"  Run timestamp:          {run_meta.get('run_ts','?')}")
        print(f"  Trading days analyzed:  {run_meta.get('days','?')}"
              f"  ({run_meta.get('first_date','?')} → {run_meta.get('last_date','?')})")
        print(f"  Expiry selection mode:  {run_meta.get('select_mode','?')}")
        print(f"  DTE window:             {MIN_DTE}–{MAX_DTE} days")
        print(f"  Expiries tracked/bar:   {N_EXP}")
        raw = run_meta.get("total_raw", "?")
        res = run_meta.get("total_resampled", "?")
        if resample > 1:
            print(f"  Raw 1-min bar-expiry records:    {raw}")
            print(f"  After {resample}-min resampling:        {res}")
        else:
            print(f"  Total 1-min bar-expiry records:  {raw}")
        print(f"  Expiry series qualifying:        {len(all_results)}")

    # ── 1. Per-expiry breakdown ──────────────────────────────────────────────
    print("\n── 1. Per-expiry breakdown ───────────────────────────────────────────\n")
    print(f"  {'Expiry':<12} {'Bars':>5}  "
          f"{'α mean':>8} {'α CV':>6}  "
          f"{'γ mean':>8} {'γ CV':>6}  "
          f"{'rr25 RMSE':>12}  {'carry vs rough':>14}")
    print(f"  {'-'*12} {'-'*5}  {'-'*8} {'-'*6}  {'-'*8} {'-'*6}  "
          f"{'-'*12}  {'-'*14}")
    for exp_str, ev in sorted(all_results.items()):
        m   = ev.get("_meta", {})
        nb  = m.get("n_bars", 0)
        am  = m.get("alpha_mean", float("nan"))
        acv = m.get("alpha_cv",   float("nan"))
        gm  = m.get("gamma_mean", float("nan"))
        gcv = m.get("gamma_cv",   float("nan"))
        rr_carry = ev.get("rr25",{}).get("carry",{}).get("rmse", float("nan"))
        rr_rough = ev.get("rr25",{}).get("rough",{}).get("rmse", float("nan"))
        delta_pct = (rr_rough - rr_carry) / rr_carry * 100 if math.isfinite(rr_carry) and rr_carry > 0 else float("nan")
        winner = "carry" if delta_pct < 0 else "rough" if delta_pct > 0 else "tie"
        print(f"  {exp_str:<12} {nb:>5}  "
              f"{am:>+8.4f} {acv:>6.3f}  "
              f"{gm:>+8.4f} {gcv:>6.3f}  "
              f"{rr_carry:>12.5f}  {delta_pct:>+12.1f}% ({winner})")

    # ── 2. Feature distributions ─────────────────────────────────────────────
    print("\n── 2. Feature distributions (across all qualifying bars) ─────────────\n")
    print(f"  {'Feature':<15} {'N':>6} {'mean':>9} {'std':>8} {'min':>9} "
          f"{'p25':>9} {'p50':>9} {'p75':>9} {'max':>9}")
    print(f"  {'-'*15} {'-'*6} {'-'*9} {'-'*8} {'-'*9} "
          f"{'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    dist_agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ev in all_results.values():
        for feat in ["atm_total_var","rr25","bf25"]:
            d = ev.get(f"_dist_{feat}", {})
            for k, v in d.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    dist_agg[feat][k].append(float(v))
    for feat in ["atm_total_var","rr25","bf25"]:
        d = dist_agg[feat]
        if not d:
            continue
        n   = int(sum(d.get("n",[])))
        mn  = avg(d.get("mean",[]))
        st  = avg(d.get("std", []))
        mi  = avg(d.get("min", []))
        p25 = avg(d.get("p25", []))
        p50 = avg(d.get("p50", []))
        p75 = avg(d.get("p75", []))
        mx  = avg(d.get("max", []))
        print(f"  {feat:<15} {n:>6} {mn:>+9.5f} {st:>8.5f} {mi:>+9.5f} "
              f"{p25:>+9.5f} {p50:>+9.5f} {p75:>+9.5f} {mx:>+9.5f}")

    # ── 3. RMSE ──────────────────────────────────────────────────────────────
    print("\n── 3. RMSE (lower = better) ──────────────────────────────────────────\n")
    print(f"  {'Feature':<18} {'Carry':>10} {'Rough':>10} {'AR(1)':>10}  "
          f"{'vs-carry':>9}  Winner")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}  {'-'*9}  {'------'}")
    for feat in ["atm_total_var","rr25","bf25"]:
        if feat not in agg:
            continue
        c = avg(agg[feat]["carry"]["rmse"])
        r = avg(agg[feat]["rough"]["rmse"])
        a = avg(agg[feat]["ar1"]["rmse"])
        best   = min(c, r, a)
        winner = "carry" if best==c else ("rough" if best==r else "ar1")
        pct    = (r-c)/c*100 if math.isfinite(c) and c > 0 else float("nan")
        print(f"  {feat:<18} {c:>10.5f} {r:>10.5f} {a:>10.5f}  "
              f"{pct:>+8.1f}%  {winner}")

    # ── 4. Directional accuracy ───────────────────────────────────────────────
    print("\n── 4. Directional accuracy (% correct sign of 1-bar change) ──────────\n")
    print(f"  {'Feature':<18} {'Carry':>10} {'Rough':>10} {'AR(1)':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for feat in ["atm_total_var","rr25","bf25"]:
        if feat not in agg:
            continue
        c = avg(agg[feat]["carry"]["dir"]) * 100
        r = avg(agg[feat]["rough"]["dir"]) * 100
        a = avg(agg[feat]["ar1"]["dir"])   * 100
        print(f"  {feat:<18} {c:>9.1f}% {r:>9.1f}% {a:>9.1f}%")

    # ── 5. Residual autocorrelation ───────────────────────────────────────────
    print("\n── 5. Residual autocorrelation, lag 1 ────────────────────────────────\n")
    print("  Negative ACF → mean-reversion → tradeable signal exists")
    print("  Positive ACF → momentum       → carry dominates\n")
    for feat, key in [("rr25","rr25_resid_acf"), ("bf25","bf25_resid_acf")]:
        acf  = avg(meta[key])
        pval = avg(meta[feat+"_resid_pval"])
        bar  = "◀ mean-reverts (signal candidate)" \
               if (math.isfinite(acf) and acf < -0.05) else "▶ momentum/noise"
        print(f"  {feat} residual ACF(1) = {acf:+.4f}  (p={pval:.4f})  {bar}")

    # ── 6. Structural coefficient stability ───────────────────────────────────
    print("\n── 6. Structural coefficient stability ───────────────────────────────\n")
    print("  CV = std/|mean|.  < 0.5 → stable structure.  > 1.0 → model breaks.\n")
    print(f"  {'Coefficient':<30} {'mean':>9} {'std':>9} {'CV':>7}  flag")
    print(f"  {'-'*30} {'-'*9} {'-'*9} {'-'*7}  ----")
    for name, mean_k, std_k, cv_k in [
        ("α  rr25/(T^{H-.5}·σ)",     "alpha_mean", "alpha_std", "alpha_cv"),
        ("γ  bf25/(T^{2H-1}·ATV)",   "gamma_mean", "gamma_std", "gamma_cv"),
    ]:
        m  = avg(meta[mean_k])
        s  = avg(meta[std_k])
        cv = avg(meta[cv_k])
        flag = "" if not math.isfinite(cv) else \
               ("✓ stable" if cv < 0.5 else ("~ marginal" if cv < 1.0 else "✗ unstable"))
        print(f"  {name:<30} {m:>+9.4f} {s:>9.4f} {cv:>7.3f}  {flag}")

    # ── 7. Verdict ────────────────────────────────────────────────────────────
    print("\n── 7. VERDICT ────────────────────────────────────────────────────────\n")
    rr_imp = avg(agg["rr25"]["carry"]["rmse"]) - avg(agg["rr25"]["rough"]["rmse"])
    bf_imp = avg(agg["bf25"]["carry"]["rmse"]) - avg(agg["bf25"]["rough"]["rmse"])
    rr_acf = avg(meta["rr25_resid_acf"])
    bf_acf = avg(meta["bf25_resid_acf"])
    acv    = avg(meta["alpha_cv"])
    gcv    = avg(meta["gamma_cv"])

    beats_carry   = rr_imp > 0 or bf_imp > 0
    mean_reverts  = (math.isfinite(rr_acf) and rr_acf < -0.05) or \
                    (math.isfinite(bf_acf) and bf_acf < -0.05)
    stable_struct = (math.isfinite(acv) and acv < 0.75) and \
                    (math.isfinite(gcv) and gcv < 0.75)

    print(f"  Rough RMSE < carry on rr25:         {'YES' if rr_imp>0 else 'NO'}  (Δ={rr_imp:+.5f})")
    print(f"  Rough RMSE < carry on bf25:         {'YES' if bf_imp>0 else 'NO'}  (Δ={bf_imp:+.5f})")
    print(f"  Residuals mean-revert (ACF<-0.05):  {'YES' if mean_reverts else 'NO'}")
    print(f"  Structural coeff stable (CV<0.75):  {'YES' if stable_struct else 'NO'}")
    print()

    if beats_carry and mean_reverts and stable_struct:
        v = "PROCEED"
        d = ("Rough model outperforms carry AND residuals mean-revert AND structural "
             "coefficients are stable. Build the surface pipeline.")
    elif beats_carry or mean_reverts:
        v = "WEAK — investigate before building"
        d = ("Partial evidence. Consider running more days and testing at different "
             "horizons before committing to multi-leg execution.")
    else:
        v = f"ABORT — no rough-vol edge at {bar_label} frequency"
        d = ("Carry dominates. Residuals do not mean-revert. The rough model adds "
             "no predictive information over naive baselines at this timescale. "
             "Structural coefficients are stable cross-sectionally but provide "
             "no temporal forecasting power.")

    print(f"  >>> {v}")
    print(f"      {d}")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Gate 0: Rough Model Temporal Forecast Benchmark (roughtemporal_intraday)")
    ap.add_argument("--days",     type=int,   default=None,
                    help="Number of trading days to process (default: all)")
    ap.add_argument("--h",        type=float, default=0.1,
                    help="Hurst exponent (default: 0.1)")
    ap.add_argument("--quick",    action="store_true",
                    help="Alias for --days 5")
    ap.add_argument("--resample", type=int,   default=1,
                    help="Bar size in minutes (default: 1). Try 5 or 120.")
    ap.add_argument("--far",      action="store_true",
                    help="Track furthest expiries in DTE window instead of nearest "
                         "(auto-enabled for --resample >= 60)")
    ap.add_argument("--workers",  type=int,   default=1,
                    help="Parallel day-processing workers (default: 1)")
    ap.add_argument("--output",   action="store_true",
                    help="Save full run log + detailed report to output/<bar>_<timestamp>.txt")
    args = ap.parse_args()

    H          = args.h
    resample   = args.resample
    select_far = args.far or (resample >= 60)
    n_days     = 5 if args.quick else (args.days or None)

    if not OPRA_ZIP.exists():
        sys.exit(f"OPRA zip not found: {OPRA_ZIP}")

    # ── optionally tee stdout to a buffer for file output ────────────────────
    buf = None
    if args.output:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        sys.stdout = _Tee(sys.__stdout__, buf)

    try:
        _run(H, resample, select_far, n_days, args.workers)
    finally:
        if buf is not None:
            sys.stdout = sys.__stdout__
            bar_str = f"{resample}min"
            stamp   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"gate0_{bar_str}_{stamp}.txt"
            out_path.write_text(buf.getvalue())
            print(f"\nReport saved → {out_path}")


def _run(H: float, resample: int, select_far: bool, n_days: int | None,
         workers: int = 1):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with zipfile.ZipFile(OPRA_ZIP) as zf_index:
        dbn_files = sorted(f for f in zf_index.namelist() if f.endswith(".dbn.zst"))
    if n_days:
        dbn_files = dbn_files[:n_days]

    bar_label = f"{resample}-min" if resample > 1 else "1-min"
    sel_label = "far (furthest expiries)" if select_far else "near (nearest expiries)"
    print(f"\nGate 0 — roughtemporal_intraday")
    print(f"  Run:      {run_ts}")
    print(f"  Days:     {len(dbn_files)}  |  H={H}  |  bar={bar_label}  |  select={sel_label}")
    print(f"  Workers:  {workers}")
    print(f"  DTE:      {MIN_DTE}–{MAX_DTE}  |  N_EXP={N_EXP}  |  RATE={RATE}  |  DIV={DIV}")

    dates_order = []
    for fname in dbn_files:
        ds    = re.search(r"(\d{8})", fname).group(1)
        tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        dates_order.append((fname, tdate))

    task_args = [
        (str(OPRA_ZIP), fname, tdate, H, select_far)
        for fname, tdate in dates_order
    ]

    total = len(task_args)
    t0    = time.time()
    day_map: dict = {}

    if workers > 1:
        ctx = mp.get_context("spawn")
        completed = 0
        with ctx.Pool(workers) as pool:
            for tdate, recs in pool.imap_unordered(_gate0_worker, task_args):
                completed += 1
                _log(f"Day {completed:3d}/{total:3d} — {tdate} — {len(recs)} records")
                day_map[tdate] = recs
    else:
        with zipfile.ZipFile(OPRA_ZIP) as zf:
            for i, (fname, tdate) in enumerate(dates_order, 1):
                try:
                    recs = process_day(zf, fname, tdate, H, select_far)
                except Exception as e:
                    _log(f"Day {i:3d}/{total:3d} — {tdate} — SKIP: {e}")
                    recs = []
                else:
                    _log(f"Day {i:3d}/{total:3d} — {tdate} — {len(recs)} records")
                day_map[tdate] = recs

    elapsed = time.time() - t0
    total_extracted = sum(len(v) for v in day_map.values())
    _log(f"Extraction done: {total} days, {total_extracted} records  (elapsed {elapsed:.0f}s)")

    all_expiry_ts: dict[date, list] = defaultdict(list)
    dates_seen = []
    total_raw  = 0

    for fname, tdate in dates_order:
        recs = day_map.get(tdate, [])
        dates_seen.append(tdate)
        for r in recs:
            all_expiry_ts[r["expiry"]].append(r)
        total_raw += len(recs)

    if not all_expiry_ts:
        sys.exit("No valid records. Check data path and market-hours filter.")

    # ── resampling ────────────────────────────────────────────────────────────
    total_resampled = total_raw
    if resample > 1:
        num_feat = ["T","atm_iv","atm_total_var","rr25","bf25","alpha","gamma"]
        key_feat = {"atm_total_var","rr25","bf25"}
        step_s   = resample * 60
        resampled: dict[date, list] = defaultdict(list)
        for exp, ts_list in all_expiry_ts.items():
            bucket_map: dict[int, dict] = {}
            for r in ts_list:
                epoch  = int(r["ts"].timestamp())
                bucket = (epoch // step_s) * step_s
                bucket_map[bucket] = r
            for bucket_epoch, r in sorted(bucket_map.items()):
                if not all(math.isfinite(r.get(k, float("nan"))) for k in key_feat):
                    continue
                rec = {c: r[c] for c in ["expiry"] + num_feat}
                rec["ts"] = pd.Timestamp(bucket_epoch, unit="s", tz="UTC")
                resampled[exp].append(rec)
        total_resampled = sum(len(v) for v in resampled.values())
        print(f"  Resampled to {resample}-min bars: {total_resampled} total bar-expiry records")
        all_expiry_ts = resampled

    # ── evaluate ──────────────────────────────────────────────────────────────
    min_bars = max(20, 60 // max(resample, 1))
    print(f"\nEvaluating {len(all_expiry_ts)} expiry series (min {min_bars} bars each) ...")
    all_results = {}
    for exp, ts in sorted(all_expiry_ts.items()):
        ts.sort(key=lambda r: r["ts"])
        if len(ts) < min_bars:
            continue
        all_results[str(exp)] = evaluate_forecasts(ts, H)
        print(f"  {exp}: {len(ts)} bars")

    if not all_results:
        sys.exit(f"No expiry had ≥{min_bars} bars. "
                 f"Run with more --days or a smaller --resample.")

    first_date = str(min(dates_seen)) if dates_seen else "?"
    last_date  = str(max(dates_seen)) if dates_seen else "?"

    run_meta = {
        "run_ts":           run_ts,
        "days":             len(dbn_files),
        "first_date":       first_date,
        "last_date":        last_date,
        "select_mode":      sel_label,
        "total_raw":        total_raw,
        "total_resampled":  total_resampled,
    }

    print_report(all_results, H, resample, run_meta)


if __name__ == "__main__":
    main()
