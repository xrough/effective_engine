#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
skew_scaling/benchmark.py
==========================
Experiment: Cross-sectional rr25 maturity power-law scaling

Research question (Gate 0A):
    Does the short-end SPY smile exhibit the maturity-scaling structure
    predicted by rough-volatility asymptotics?

Theoretical derivation:
    Rough vol predicts ATM skew scales as:
        ∂σ/∂k|_{k=0} ~ T^{H − 0.5}

    However rr25 is NOT the ATM skew. The 25-delta strikes sit at
    log-moneyness k ≈ ±0.674·σ·√T. Under a linear smile:

        rr25 ≈ skew × 2 × 0.674 × σ × √T
              ~ T^{H − 0.5} × √T = T^H

    Therefore the correct theoretical slope for the rr25 log-log regression is:

        log|rr25(T)| = intercept + beta · log(T),  beta ≈ H = 0.10

    (NOT H−0.5 = −0.40, which applies only to the true ATM derivative ∂σ/∂k.)

Design
------
1. Load DBN data using process_day_full (all expiries, no N_EXP cap).
2. Group records by timestamp.
3. Per timestamp with ≥3 expiries:
   fit: log|rr25| = a + beta · log(T)
4. Report distribution of beta across all qualifying timestamps.

Success criteria (PROCEED):
  - Median beta within ±0.10 of theoretical H (= 0.10 for H=0.1)
  - Mean R² > 0.70
  - Beta CV (std/|mean|) < 0.50

Usage
-----
  python benchmark.py [--days N] [--h H] [--quick] [--output]

  --days N    Number of trading days (default: all 127)
  --h H       Hurst exponent (default: 0.1)
  --quick     Alias for --days 5
  --output    Save full run log to output/<timestamp>.txt
"""

import argparse
import io
import math
import multiprocessing as mp
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

# ── shared pipeline ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "shared"))
from smile_pipeline import (
    RATE, DIV, MIN_DTE, MAX_DTE, MKT_OPEN_UTC, MKT_CLOSE_UTC,
    _Tee, process_day_full, get_device,
)


# ── parallel worker (module-level for pickling with spawn context) ─────────────
def _day_worker(args):
    """Open a fresh ZipFile per process — ZipFile handles are not picklable."""
    zip_path, fname, trade_date, H, device, min_dte, max_dte = args
    import zipfile as _zf
    try:
        with _zf.ZipFile(zip_path) as zf:
            return process_day_full(zf, fname, trade_date, H, device, min_dte, max_dte)
    except Exception:
        return []

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = _HERE.parents[4]   # .../MVP
OPRA_ZIP = ROOT / "demo" / "data" / "OPRA-20260208-3T68RYYKF9.zip"
OUT_DIR  = _HERE.parent / "output"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional log-log regression
# ─────────────────────────────────────────────────────────────────────────────

def fit_skew_scaling(ts_group: list) -> dict | None:
    """
    Fit log|rr25(T)| = intercept + beta·log(T) for one timestamp.

    Inputs:
      ts_group : list of records for one timestamp, each with keys T, rr25
    Requirements:
      ≥3 expiries with rr25 < 0 (SPY is negatively skewed; positive values
      at a timestamp are treated as data artifacts and excluded)
    Returns:
      {beta, intercept, r2, n_exp} or None if requirements not met.
    """
    pairs = [
        (r["T"], r["rr25"])
        for r in ts_group
        if math.isfinite(r.get("rr25", float("nan")))
        and r["rr25"] < 0
        and math.isfinite(r.get("T", float("nan")))
        and r["T"] > 1e-5
    ]
    if len(pairs) < 3:
        return None

    T_arr  = np.array([p[0] for p in pairs])
    rr_arr = np.array([p[1] for p in pairs])

    log_T  = np.log(T_arr)
    log_rr = np.log(np.abs(rr_arr))

    # OLS: [ones, log_T] @ [intercept, beta] = log_rr
    X = np.column_stack([np.ones(len(log_T)), log_T])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, log_rr, rcond=None)
    except Exception:
        return None

    intercept, beta = float(coeffs[0]), float(coeffs[1])

    resid  = log_rr - (intercept + beta * log_T)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((log_rr - log_rr.mean())**2))
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    return {"beta": beta, "intercept": intercept, "r2": r2, "n_exp": len(pairs)}


def evaluate_skew_scaling(all_ts_records: dict, H: float) -> dict:
    """
    Run fit_skew_scaling for every timestamp and aggregate the slope distribution.

    Parameters
    ----------
    all_ts_records : {ts → list of per-expiry records}
    H              : Hurst exponent (theoretical slope = H − 0.5)

    Returns
    -------
    dict with keys:
      n_ts_total, n_ts_fitted — total vs qualifying timestamps
      beta_mean, beta_std, beta_median, beta_p5, beta_p95, beta_cv
      r2_mean, r2_median
      theory_beta — H  (rr25 ~ T^H, not T^{H-0.5})
      frac_consistent — fraction of timestamps where |beta − theory| < 0.10
      verdict — PROCEED | WEAK | ABORT
    """
    betas, r2s, n_exps = [], [], []
    n_total = 0

    for ts in sorted(all_ts_records):
        n_total += 1
        result = fit_skew_scaling(all_ts_records[ts])
        if result is None:
            continue
        betas.append(result["beta"])
        r2s.append(result["r2"])
        n_exps.append(result["n_exp"])

    if not betas:
        return {"n_ts_total": n_total, "n_ts_fitted": 0, "verdict": "ABORT"}

    betas_arr = np.array(betas)
    r2_arr    = np.array([r for r in r2s if math.isfinite(r)])
    theory    = H          # rr25 ~ T^H (not T^{H-0.5}; see module docstring)

    beta_mean   = float(np.mean(betas_arr))
    beta_std    = float(np.std(betas_arr))
    beta_median = float(np.median(betas_arr))
    beta_cv     = beta_std / abs(beta_mean) if abs(beta_mean) > 1e-8 else float("nan")
    r2_mean     = float(np.mean(r2_arr)) if len(r2_arr) > 0 else float("nan")
    r2_median   = float(np.median(r2_arr)) if len(r2_arr) > 0 else float("nan")
    frac        = float(np.mean(np.abs(betas_arr - theory) < 0.10))

    # Verdict
    crit_beta = abs(beta_median - theory) < 0.10
    crit_r2   = math.isfinite(r2_mean) and r2_mean > 0.70
    crit_cv   = math.isfinite(beta_cv) and beta_cv < 0.50

    if crit_beta and crit_r2 and crit_cv:
        verdict = "PROCEED"
    elif crit_beta or (crit_r2 and crit_cv):
        verdict = "WEAK — partial evidence"
    else:
        verdict = "ABORT — no power-law scaling consistent with rough vol"

    return {
        "n_ts_total":    n_total,
        "n_ts_fitted":   len(betas),
        "beta_mean":     beta_mean,
        "beta_std":      beta_std,
        "beta_median":   beta_median,
        "beta_p5":       float(np.percentile(betas_arr, 5)),
        "beta_p95":      float(np.percentile(betas_arr, 95)),
        "beta_cv":       beta_cv,
        "r2_mean":       r2_mean,
        "r2_median":     r2_median,
        "mean_n_exp":    float(np.mean(n_exps)) if n_exps else float("nan"),
        "theory_beta":   theory,
        "frac_consistent": frac,
        "verdict":       verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(result: dict, H: float, run_meta: dict | None = None):
    W = 70

    print("\n" + "=" * W)
    print("  ATM SKEW POWER-LAW SCALING — Cross-Sectional Benchmark")
    print("  Experiment: skew_scaling")
    print(f"  H={H:.2f}  (theoretical slope for rr25 = H = {H:+.2f})")
    print("=" * W)

    if run_meta:
        print("\n── 0. Experiment metadata ──────────────────────────────────────\n")
        print(f"  Run:              {run_meta.get('run_ts','?')}")
        print(f"  Trading days:     {run_meta.get('days','?')}"
              f"  ({run_meta.get('first_date','?')} → {run_meta.get('last_date','?')})")
        print(f"  DTE window:       {MIN_DTE}–{MAX_DTE} days")
        print(f"  RATE={RATE:.3f}  DIV={DIV:.3f}")
        print(f"  Total timestamps evaluated:   {result.get('n_ts_total', '?')}")
        print(f"  Timestamps with ≥3 expiries:  {result.get('n_ts_fitted', '?')}")
        mean_n = result.get("mean_n_exp", float("nan"))
        if math.isfinite(mean_n):
            print(f"  Mean expiries per timestamp:  {mean_n:.1f}")

    theory = result.get("theory_beta", H)   # rr25 ~ T^H

    print("\n── 1. Slope (beta) distribution ────────────────────────────────\n")
    print(f"  Theoretical beta (H−0.5):  {theory:+.4f}")
    print()
    print(f"  {'Statistic':<20} {'Value':>10}")
    print(f"  {'-'*20} {'-'*10}")
    for label, key in [
        ("Mean",       "beta_mean"),
        ("Std",        "beta_std"),
        ("Median",     "beta_median"),
        ("p5",         "beta_p5"),
        ("p95",        "beta_p95"),
        ("CV",         "beta_cv"),
    ]:
        v = result.get(key, float("nan"))
        print(f"  {label:<20} {v:>+10.4f}" if math.isfinite(v) else
              f"  {label:<20} {'nan':>10}")
    frac = result.get("frac_consistent", float("nan"))
    print(f"\n  Fraction of timestamps with |beta − theory| < 0.10: "
          f"{frac*100:.1f}%" if math.isfinite(frac) else "  n/a")

    print("\n── 2. Log-log fit quality (R²) ─────────────────────────────────\n")
    r2m = result.get("r2_mean",   float("nan"))
    r2d = result.get("r2_median", float("nan"))
    print(f"  Mean R²:    {r2m:.4f}" if math.isfinite(r2m) else "  Mean R²:    nan")
    print(f"  Median R²:  {r2d:.4f}" if math.isfinite(r2d) else "  Median R²:  nan")

    print("\n── 3. Verdict ──────────────────────────────────────────────────\n")
    bm  = result.get("beta_median", float("nan"))
    bcv = result.get("beta_cv",    float("nan"))
    print(f"  Median beta within ±0.10 of H={theory:+.2f} (rr25~T^H): "
          f"{'YES' if math.isfinite(bm) and abs(bm-theory)<0.10 else 'NO'}"
          f"  (median={bm:+.4f})")
    print(f"  Mean R² > 0.70:   {'YES' if math.isfinite(r2m) and r2m>0.70 else 'NO'}"
          f"  ({r2m:.4f})")
    print(f"  Beta CV < 0.50:   {'YES' if math.isfinite(bcv) and bcv<0.50 else 'NO'}"
          f"  ({bcv:.4f})")
    print()
    verdict = result.get("verdict", "ABORT")
    print(f"  >>> {verdict}")
    if "PROCEED" in verdict:
        print("      rr25 scales as T^H consistent with rough-vol asymptotics.")
        print("      Cross-sectional geometry validation passes.")
    elif "WEAK" in verdict:
        print(f"      Partial evidence. Empirical slope ({bm:+.4f}) is in the right")
        print(f"      direction but deviates from theory (H={theory:+.2f}).")
        print("      Possible causes: non-linear smile, vol-of-vol contamination,")
        print("      or H is higher than assumed.")
    else:
        print("      rr25 does not scale as T^H. Rough-vol cross-sectional")
        print("      geometry is not supported at this DTE window.")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="ATM Skew Power-Law Scaling benchmark (skew_scaling)")
    ap.add_argument("--days",    type=int,   default=None)
    ap.add_argument("--h",       type=float, default=0.1)
    ap.add_argument("--quick",   action="store_true", help="--days 5")
    ap.add_argument("--workers", type=int,   default=1,
                    help="Parallel day-processing workers (default: 1)")
    ap.add_argument("--device",  type=str,   default="auto",
                    choices=["auto","cpu","cuda","mps"],
                    help="IV computation device (default: auto)")
    ap.add_argument("--output",  action="store_true",
                    help="Save full run log to output/<timestamp>.txt")
    args = ap.parse_args()

    H      = args.h
    n_days = 5 if args.quick else (args.days or None)
    device = get_device(args.device)

    if not OPRA_ZIP.exists():
        sys.exit(f"OPRA zip not found: {OPRA_ZIP}")

    buf = None
    if args.output:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        sys.stdout = _Tee(sys.__stdout__, buf)

    try:
        _run(H, n_days, args.workers, device)
    finally:
        if buf is not None:
            sys.stdout = sys.__stdout__
            stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = OUT_DIR / f"skew_scaling_{stamp}.txt"
            out_path.write_text(buf.getvalue())
            print(f"\nReport saved → {out_path}")


def _run(H: float, n_days: int | None, workers: int = 1, device: str = "cpu"):
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with zipfile.ZipFile(OPRA_ZIP) as zf_index:
        dbn_files = sorted(f for f in zf_index.namelist() if f.endswith(".dbn.zst"))
    if n_days:
        dbn_files = dbn_files[:n_days]

    print(f"\nskew_scaling benchmark")
    print(f"  Run:     {run_ts}")
    print(f"  Days:    {len(dbn_files)}  |  H={H}  |  DTE={MIN_DTE}–{MAX_DTE}")
    print(f"  Device:  {device}  |  Workers: {workers}")
    print(f"  Mode:    all expiries per timestamp (cross-sectional)\n")

    dates_order = []
    for fname in dbn_files:
        ds    = re.search(r"(\d{8})", fname).group(1)
        tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        dates_order.append((fname, tdate))

    task_args = [
        (str(OPRA_ZIP), fname, tdate, H, device, MIN_DTE, MAX_DTE)
        for fname, tdate in dates_order
    ]

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            day_results = pool.map(_day_worker, task_args)
    else:
        with zipfile.ZipFile(OPRA_ZIP) as zf:
            day_results = []
            for fname, tdate in dates_order:
                try:
                    day_results.append(
                        process_day_full(zf, fname, tdate, H, device, MIN_DTE, MAX_DTE))
                except Exception as e:
                    print(f"  SKIP {tdate}: {e}")
                    day_results.append([])

    # Collect records grouped by timestamp (across all days and expiries)
    all_ts_records: dict = defaultdict(list)
    dates_seen = []

    for (fname, tdate), recs in zip(dates_order, day_results):
        dates_seen.append(tdate)
        n_exp_ts: dict = defaultdict(set)
        for r in recs:
            all_ts_records[r["ts"]].append(r)
            n_exp_ts[r["ts"]].add(r["expiry"])
        qualifying = sum(1 for v in n_exp_ts.values() if len(v) >= 3)
        print(f"  {tdate}: {len(recs)} records, {len(n_exp_ts)} timestamps, "
              f"{qualifying} with ≥3 expiries")

    if not all_ts_records:
        sys.exit("No valid records.")

    print(f"\nTotal unique timestamps: {len(all_ts_records)}")
    print("Running cross-sectional skew scaling regression ...")

    result = evaluate_skew_scaling(all_ts_records, H)

    run_meta = {
        "run_ts":     run_ts,
        "days":       len(dbn_files),
        "first_date": str(min(dates_seen)) if dates_seen else "?",
        "last_date":  str(max(dates_seen)) if dates_seen else "?",
    }
    print_report(result, H, run_meta)


if __name__ == "__main__":
    main()
