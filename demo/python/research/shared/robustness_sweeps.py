"""
shared/robustness_sweeps.py
============================
Shared helpers for Gate 0 and Gate 0B robustness sweeps.

Public API
----------
SweepConfig          — sweep grid and cache settings
load_cache           — load pickled records from disk (returns None on miss)
save_cache           — pickle records to disk
resample_panel       — downsample a 1-min flat record list to N-min bars
apply_n_exp_selection — filter to N nearest/farthest expiries per timestamp
recompute_structural — recompute alpha/gamma for a new H without re-running IV
classify_gate0_cell  — PASS/MARGINAL/FAIL verdict for a (H, resample) cell
classify_gate0b_cell — PASS/MARGINAL/FAIL/SKIP verdict for a (H, resample, move_pct) cell
format_gate0_summary — text pivot table (rows=H, cols=resample)
format_gate0b_summary — text pivot table for a fixed move_pct slice
"""

from __future__ import annotations

import math
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ── default grids ─────────────────────────────────────────────────────────────
DEFAULT_H_GRID        = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
DEFAULT_RESAMPLE_GRID = [1, 5, 15, 30, 60, 120, 390]   # minutes
DEFAULT_MOVE_PCT_GRID = [0.10, 0.20, 0.30]

_HERE = Path(__file__).resolve()
_DEFAULT_CACHE_DIR = _HERE.parents[1] / "cache"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── SweepConfig ───────────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    gate_id: str                            # "gate0" | "gate0b"
    h_grid: list[float] = field(
        default_factory=lambda: list(DEFAULT_H_GRID))
    resample_grid: list[int] = field(
        default_factory=lambda: list(DEFAULT_RESAMPLE_GRID))
    move_pct_grid: list[float] = field(
        default_factory=lambda: list(DEFAULT_MOVE_PCT_GRID))
    n_days: int | None = None               # None = all available
    select_far: bool = False
    workers: int = 1
    device: str = "cpu"
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)


# ── Cache layer ───────────────────────────────────────────────────────────────

def _cache_path(cfg: SweepConfig) -> Path:
    """Cache key: gate_id + n_days + select_far (not H or resample)."""
    n = cfg.n_days if cfg.n_days is not None else "all"
    far = "T" if cfg.select_far else "F"
    return cfg.cache_dir / f"{cfg.gate_id}_days{n}_far{far}.pkl"


def load_cache(cfg: SweepConfig) -> list | None:
    """Return pickled records, or None if cache file does not exist."""
    p = _cache_path(cfg)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            records = pickle.load(f)
        _log(f"Cache hit: loaded {len(records)} records from {p.name}")
        return records
    except Exception as e:
        _log(f"Cache read failed ({p.name}): {e} — will re-extract")
        return None


def save_cache(cfg: SweepConfig, records: list) -> None:
    """Pickle records list to cache path."""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cfg)
    with open(p, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"Cache saved: {len(records)} records → {p.name}")


# ── Panel helpers ─────────────────────────────────────────────────────────────

def resample_panel(records: list, step_min: int) -> list:
    """
    Downsample 1-min records to step_min-bar by keeping the last record per bucket.

    Groups by (expiry, time_bucket). Buckets with non-finite key features are dropped.
    """
    if step_min <= 1:
        return list(records)

    step_s   = step_min * 60
    key_feat = {"atm_total_var", "rr25", "bf25"}

    bucket_map: dict[tuple, dict] = {}
    for r in records:
        epoch  = int(r["ts"].timestamp())
        bucket = (epoch // step_s) * step_s
        key    = (r["expiry"], bucket)
        bucket_map[key] = r   # last record in window wins

    out = []
    for (exp, bucket_epoch), r in sorted(bucket_map.items()):
        if not all(math.isfinite(r.get(k, float("nan"))) for k in key_feat):
            continue
        rec = dict(r)
        rec["ts"] = pd.Timestamp(bucket_epoch, unit="s", tz="UTC")
        out.append(rec)
    return out


def apply_n_exp_selection(records: list, n_exp: int = 2,
                          select_far: bool = False) -> list:
    """
    Per timestamp, keep the n_exp nearest (or farthest) expiries by DTE.

    Mirrors the N_EXP selection in process_day(). Used after loading Gate 0B
    cache (which stores all expiries) to make it comparable with Gate 0.
    """
    ts_exp: dict = defaultdict(set)
    for r in records:
        ts_exp[r["ts"]].add(r["expiry"])

    selected: dict = {}
    for ts, exps in ts_exp.items():
        sorted_exps = sorted(exps)
        selected[ts] = set(
            sorted_exps[-n_exp:] if select_far else sorted_exps[:n_exp]
        )

    return [r for r in records if r["expiry"] in selected.get(r["ts"], set())]


def recompute_structural(records: list, H: float) -> list:
    """
    Return a new list with alpha and gamma recomputed for a different H.

    rr25, bf25, atm_iv, atm_total_var, T are H-independent (they come from
    the IV solver). Only the structural coefficients change:

        alpha = rr25 / (T^{H-0.5} * atm_iv)
        gamma = bf25 / (T^{2H-1} * atm_total_var)
    """
    out = []
    for r in records:
        T   = r.get("T", float("nan"))
        atm = r.get("atm_iv", float("nan"))
        atv = r.get("atm_total_var", float("nan"))
        rr  = r.get("rr25", float("nan"))
        bf  = r.get("bf25", float("nan"))

        if T > 1e-6 and math.isfinite(atm) and atm > 1e-6 and math.isfinite(rr):
            alpha = rr / ((T ** (H - 0.5)) * atm)
        else:
            alpha = float("nan")

        denom = (T ** (2 * H - 1)) * atv if (T > 1e-6 and math.isfinite(atv)) else 0.0
        gamma = bf / denom if (denom > 1e-8 and math.isfinite(bf)) else float("nan")

        new_r = dict(r)
        new_r["alpha"] = alpha
        new_r["gamma"] = gamma
        out.append(new_r)
    return out


# ── Metric aggregation ────────────────────────────────────────────────────────

def _agg_metrics(ev: dict) -> dict:
    """
    Flatten an evaluate_forecasts() output to scalar metrics.

    Returns keys like: rr25_rmse_carry, rr25_rmse_rough, rr25_dir_rough,
    rr25_acf1, bf25_rmse_carry, bf25_rmse_rough, n_bars, alpha_cv, gamma_cv, ...
    """
    out: dict = {}
    for feat in ["rr25", "bf25", "atm_total_var"]:
        if feat not in ev:
            continue
        for method in ["carry", "rough", "ar1"]:
            md = ev[feat].get(method, {})
            out[f"{feat}_rmse_{method}"] = md.get("rmse", float("nan"))
            out[f"{feat}_dir_{method}"]  = md.get("dir",  float("nan"))

    meta = ev.get("_meta", {})
    for k in ["n_bars", "rr25_resid_acf", "rr25_resid_pval",
              "bf25_resid_acf", "bf25_resid_pval",
              "alpha_mean", "alpha_cv", "gamma_mean", "gamma_cv"]:
        out[k] = meta.get(k, float("nan"))
    return out


def _avg_metrics(metric_list: list[dict]) -> dict:
    """Average a list of _agg_metrics dicts across expiries."""
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    out  = {}
    for k in keys:
        vals = [m[k] for m in metric_list if math.isfinite(m.get(k, float("nan")))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


# ── Cell classification ───────────────────────────────────────────────────────

def classify_gate0_cell(agg: dict) -> str:
    """
    Verdict for one (H, resample) cell based on averaged metrics across expiries.

    agg: output of _avg_metrics([_agg_metrics(ev) for ev in expiry_evs])

    Returns:
      "PASS"     — rough beats carry by >5% on rr25 or bf25
      "MARGINAL" — rough is within ±5% of carry (either feature)
      "FAIL"     — rough is worse than carry by >5% on both features
      "SKIP"     — insufficient data
    """
    def _improvement(carry_key: str, rough_key: str) -> float:
        c = agg.get(carry_key, float("nan"))
        r = agg.get(rough_key, float("nan"))
        if not (math.isfinite(c) and math.isfinite(r) and c > 0):
            return float("nan")
        return (c - r) / c   # positive = rough better

    rr_imp = _improvement("rr25_rmse_carry", "rr25_rmse_rough")
    bf_imp = _improvement("bf25_rmse_carry", "bf25_rmse_rough")

    finite = [x for x in [rr_imp, bf_imp] if math.isfinite(x)]
    if not finite:
        return "SKIP"

    best = max(finite)
    if best > 0.05:
        return "PASS"
    if best > -0.05:
        return "MARGINAL"
    return "FAIL"


def classify_gate0b_cell(agg_active: dict | None,
                         agg_quiet:  dict | None) -> str:
    """
    Verdict for one (H, resample, move_pct) cell.

    Returns:
      "PASS"     — rough beats carry in active regime, NOT in quiet
      "MARGINAL" — rough advantage ≥3% in active but also present in quiet
      "FAIL"     — no rough advantage in active regime
      "SKIP"     — active or quiet partition is None/empty
    """
    if agg_active is None or agg_quiet is None:
        return "SKIP"

    def _beats(agg: dict) -> bool:
        for feat in ["rr25", "bf25"]:
            c = agg.get(f"{feat}_rmse_carry", float("nan"))
            r = agg.get(f"{feat}_rmse_rough", float("nan"))
            if math.isfinite(c) and math.isfinite(r) and r < c:
                return True
        return False

    def _best_improvement(agg: dict) -> float:
        best = float("nan")
        for feat in ["rr25", "bf25"]:
            c = agg.get(f"{feat}_rmse_carry", float("nan"))
            r = agg.get(f"{feat}_rmse_rough", float("nan"))
            if math.isfinite(c) and math.isfinite(r) and c > 0:
                imp = (c - r) / c
                if not math.isfinite(best) or imp > best:
                    best = imp
        return best

    active_beats = _beats(agg_active)
    quiet_beats  = _beats(agg_quiet)
    active_imp   = _best_improvement(agg_active)

    if active_beats and not quiet_beats:
        return "PASS"
    if math.isfinite(active_imp) and active_imp >= 0.03:
        return "MARGINAL"
    if active_beats:
        return "MARGINAL"
    return "FAIL"


# ── Summary formatters ────────────────────────────────────────────────────────

_VERDICT_ORDER = {"PASS": 0, "MARGINAL": 1, "FAIL": 2, "SKIP": 3}


def format_gate0_summary(results_df: pd.DataFrame) -> str:
    """
    Pivot table: rows=H, cols=resample_min, cell=verdict.

    results_df must have columns: H, resample_min, verdict
    Aggregate rows (expiry=="ALL") are used if present; otherwise first row per cell.
    """
    if results_df.empty:
        return "(no results)"

    # prefer summary rows
    if "expiry" in results_df.columns:
        df = results_df[results_df["expiry"] == "ALL"]
        if df.empty:
            df = results_df
    else:
        df = results_df

    pivot = df.pivot_table(
        index="H", columns="resample_min", values="verdict",
        aggfunc="first"
    )

    col_w  = max(9, *(len(str(c)) for c in pivot.columns))
    h_w    = 6
    sep    = "-" * (h_w + 3 + (col_w + 3) * len(pivot.columns))

    lines = ["\nGate 0 Robustness Sweep — verdict grid (PASS/MARGINAL/FAIL)",
             f"{'H':>{h_w}} | " + " | ".join(f"{c:>{col_w}}" for c in pivot.columns),
             sep]

    n_pass = 0
    n_total = 0
    for h_val, row in pivot.iterrows():
        cells = []
        for v in row:
            v_str = str(v) if pd.notna(v) else "SKIP"
            cells.append(f"{v_str:>{col_w}}")
            if v_str == "PASS":
                n_pass += 1
            if v_str != "SKIP":
                n_total += 1
        lines.append(f"{h_val:>{h_w}.2f} | " + " | ".join(cells))

    pct = n_pass / n_total * 100 if n_total > 0 else 0.0
    lines.append(sep)
    lines.append(f"\nRobust region: {n_pass}/{n_total} cells PASS ({pct:.0f}%)")
    if pct >= 50:
        lines.append("→ Edge appears ROBUST across parameter grid (≥50% PASS)")
    elif pct >= 25:
        lines.append("→ Partial evidence — robust in some (H, resample) combinations")
    else:
        lines.append("→ Edge NOT robust — fewer than 25% of cells PASS")
    return "\n".join(lines)


def format_gate0b_summary(results_df: pd.DataFrame,
                           move_pct: float) -> str:
    """Same as format_gate0_summary but for a fixed move_pct slice of Gate 0B results."""
    if results_df.empty:
        return f"(no results for move_pct={move_pct})"

    df = results_df[results_df["move_pct"] == move_pct]
    if df.empty:
        return f"(no rows for move_pct={move_pct})"

    if "expiry" in df.columns:
        agg = df[df["expiry"] == "ALL"]
        if not agg.empty:
            df = agg

    pivot = df.pivot_table(
        index="H", columns="resample_min", values="verdict",
        aggfunc="first"
    )

    col_w = max(9, *(len(str(c)) for c in pivot.columns))
    h_w   = 6
    sep   = "-" * (h_w + 3 + (col_w + 3) * len(pivot.columns))

    lines = [f"\nGate 0B Robustness Sweep — move_pct={move_pct:.0%} "
             f"(active = top {move_pct:.0%} |Δspot|)",
             f"{'H':>{h_w}} | " + " | ".join(f"{c:>{col_w}}" for c in pivot.columns),
             sep]

    n_pass = 0
    n_total = 0
    for h_val, row in pivot.iterrows():
        cells = []
        for v in row:
            v_str = str(v) if pd.notna(v) else "SKIP"
            cells.append(f"{v_str:>{col_w}}")
            if v_str == "PASS":
                n_pass += 1
            if v_str != "SKIP":
                n_total += 1
        lines.append(f"{h_val:>{h_w}.2f} | " + " | ".join(cells))

    pct = n_pass / n_total * 100 if n_total > 0 else 0.0
    lines.append(sep)
    lines.append(f"  {n_pass}/{n_total} cells PASS ({pct:.0f}%)")
    return "\n".join(lines)
