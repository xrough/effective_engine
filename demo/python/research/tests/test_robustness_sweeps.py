#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
tests/test_robustness_sweeps.py
================================
Unit tests for shared/robustness_sweeps.py — no data files required.

Run:
    python tests/test_robustness_sweeps.py [-v]
"""

from __future__ import annotations

import math
import sys
import tempfile
import traceback
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "shared"))

from robustness_sweeps import (
    SweepConfig,
    _cache_path, load_cache, save_cache,
    resample_panel, apply_n_exp_selection, pool_by_tenor_bucket, recompute_structural,
    _agg_metrics, _avg_metrics,
    classify_gate2_cell, classify_gate3_cell,
    classify_gate4_cell, classify_gate5_cell,
    format_gate2_summary,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, str]] = []   # (name, PASS|FAIL|ERROR)


def _run(name: str, fn, verbose: bool):
    try:
        fn()
        _results.append((name, "PASS"))
        if verbose:
            print(f"  PASS  {name}")
    except AssertionError as e:
        _results.append((name, "FAIL"))
        print(f"  FAIL  {name}: {e}")
    except Exception:
        _results.append((name, "ERROR"))
        print(f"  ERROR {name}:")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for building fake records
# ─────────────────────────────────────────────────────────────────────────────

def _make_record(ts_epoch: int, expiry: date, T: float = 0.10,
                 rr25: float = -0.02, bf25: float = 0.003,
                 atm_iv: float = 0.18, H: float = 0.10) -> dict:
    atv   = atm_iv * atm_iv * T
    alpha = rr25 / ((T ** (H - 0.5)) * atm_iv) if atm_iv > 1e-6 else float("nan")
    denom = (T ** (2 * H - 1)) * atv
    gamma = bf25 / denom if denom > 1e-8 else float("nan")
    return {
        "ts":            pd.Timestamp(ts_epoch, unit="s", tz="UTC"),
        "expiry":        expiry,
        "T":             T,
        "atm_iv":        atm_iv,
        "atm_total_var": atv,
        "rr25":          rr25,
        "bf25":          bf25,
        "alpha":         alpha,
        "gamma":         gamma,
        "forward":       580.0,
    }


def _make_panel(n_bars: int = 30, step_s: int = 60,
                expiries: list | None = None,
                H: float = 0.10) -> list:
    """Build a fake 1-min panel with multiple expiries."""
    if expiries is None:
        expiries = [date(2026, 2, 20), date(2026, 3, 20)]
    records = []
    for i in range(n_bars):
        epoch = 1_740_000_000 + i * step_s   # arbitrary start
        for exp in expiries:
            T = (exp - date(2026, 1, 5)).days / 365.0
            records.append(_make_record(epoch, exp, T=T, H=H))
    return records


def _make_ev(rr25_carry: float, rr25_rough: float,
             bf25_carry: float, bf25_rough: float,
             n_bars: int = 50) -> dict:
    """Build a minimal evaluate_forecasts()-style result dict."""
    return {
        "rr25": {
            "carry": {"rmse": rr25_carry, "dir": 0.5},
            "rough": {"rmse": rr25_rough, "dir": 0.5},
            "ar1":   {"rmse": rr25_carry, "dir": 0.5},
        },
        "bf25": {
            "carry": {"rmse": bf25_carry, "dir": 0.5},
            "rough": {"rmse": bf25_rough, "dir": 0.5},
            "ar1":   {"rmse": bf25_carry, "dir": 0.5},
        },
        "_meta": {
            "n_bars": n_bars,
            "rr25_resid_acf": -0.10, "rr25_resid_pval": 0.02,
            "bf25_resid_acf": -0.08, "bf25_resid_pval": 0.05,
            "alpha_mean": -0.17, "alpha_cv": 0.30,
            "gamma_mean":  0.55, "gamma_cv": 0.25,
        },
    }


def _make_gate4_agg(rr25_carry: float,
                    rr25_cond: float,
                    rr25_recent: float,
                    bf25_carry: float,
                    bf25_cond: float,
                    bf25_recent: float) -> dict:
    """Minimal aggregated metrics dict for Gate 4 classification tests."""
    return {
        "rr25_rmse_carry": rr25_carry,
        "rr25_rmse_rough_cond_carry": rr25_cond,
        "rr25_rmse_rough_recent": rr25_recent,
        "bf25_rmse_carry": bf25_carry,
        "bf25_rmse_rough_cond_carry": bf25_cond,
        "bf25_rmse_rough_recent": bf25_recent,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_resample_panel():
    """1-min records resampled to 5-min → correct bar count per expiry."""
    panel = _make_panel(n_bars=60, step_s=60)   # 60 one-minute bars, 2 expiries
    resampled = resample_panel(panel, step_min=5)

    by_exp: dict = {}
    for r in resampled:
        by_exp.setdefault(r["expiry"], []).append(r)

    for exp, recs in by_exp.items():
        # 60 bars / 5 = 12 buckets, but bucket boundaries may clip edges
        assert 10 <= len(recs) <= 12, (
            f"Expiry {exp}: expected ~12 bars after 5-min resample, got {len(recs)}"
        )
    # passthrough for step_min=1
    same = resample_panel(panel, step_min=1)
    assert len(same) == len(panel)


def test_apply_n_exp_selection_near():
    """Keeps 2 nearest expiries per timestamp."""
    exp1 = date(2026, 2, 1)
    exp2 = date(2026, 2, 15)
    exp3 = date(2026, 3, 1)
    panel = _make_panel(n_bars=5, expiries=[exp1, exp2, exp3])

    selected = apply_n_exp_selection(panel, n_exp=2, select_far=False)
    expiries_seen = {r["expiry"] for r in selected}
    assert exp1 in expiries_seen
    assert exp2 in expiries_seen
    assert exp3 not in expiries_seen, "Farthest expiry should be excluded"


def test_apply_n_exp_selection_far():
    """Keeps 2 farthest expiries per timestamp."""
    exp1 = date(2026, 2, 1)
    exp2 = date(2026, 2, 15)
    exp3 = date(2026, 3, 1)
    panel = _make_panel(n_bars=5, expiries=[exp1, exp2, exp3])

    selected = apply_n_exp_selection(panel, n_exp=2, select_far=True)
    expiries_seen = {r["expiry"] for r in selected}
    assert exp2 in expiries_seen
    assert exp3 in expiries_seen
    assert exp1 not in expiries_seen, "Nearest expiry should be excluded"


def test_pool_by_tenor_bucket_creates_long_lived_series():
    """Tenor buckets should collapse exact expiries into front/mid roles."""
    expiries = [
        date(2026, 1, 19),  # ~14 DTE from 2026-01-05 -> front
        date(2026, 2, 2),   # ~28 DTE -> mid
        date(2026, 2, 20),  # ~46 DTE -> back
    ]
    panel = _make_panel(n_bars=8, expiries=expiries)

    pooled = pool_by_tenor_bucket(panel, select_far=False)
    series_ids = {r["series_id"] for r in pooled}
    assert series_ids == {"front", "mid"}, f"Unexpected tenor series: {series_ids}"

    by_series = {}
    for rec in pooled:
        by_series.setdefault(rec["series_id"], []).append(rec)

    assert len(by_series["front"]) == 8, "Front bucket should keep one record per timestamp"
    assert len(by_series["mid"]) == 8, "Mid bucket should keep one record per timestamp"


def test_recompute_structural_H_change():
    """alpha and gamma update correctly when H changes; rr25/bf25 unchanged."""
    panel = _make_panel(n_bars=10, H=0.10)
    recomputed = recompute_structural(panel, H=0.20)

    assert len(recomputed) == len(panel)

    for orig, new in zip(panel, recomputed):
        # rr25/bf25 unchanged
        assert orig["rr25"] == new["rr25"]
        assert orig["bf25"] == new["bf25"]
        # alpha/gamma should differ
        assert orig["alpha"] != new["alpha"] or not math.isfinite(orig["alpha"]), (
            "alpha should change when H changes"
        )
        # verify formula: alpha = rr25 / (T^(H-0.5) * atm_iv)
        T   = new["T"]
        atm = new["atm_iv"]
        rr  = new["rr25"]
        if T > 1e-6 and atm > 1e-6:
            expected_alpha = rr / ((T ** (0.20 - 0.5)) * atm)
            assert abs(new["alpha"] - expected_alpha) < 1e-10


def test_agg_metrics_keys():
    """_agg_metrics returns the expected keys."""
    ev  = _make_ev(0.01, 0.009, 0.002, 0.0018)
    agg = _agg_metrics(ev)

    for feat in ["rr25", "bf25"]:
        for method in ["carry", "rough", "ar1"]:
            assert f"{feat}_rmse_{method}" in agg, f"Missing key: {feat}_rmse_{method}"
    assert "n_bars" in agg
    assert "alpha_cv" in agg
    assert math.isfinite(agg["rr25_rmse_rough"])


def test_classify_gate2_cell_pass():
    """PASS when rough beats carry by >5% on rr25."""
    agg = _agg_metrics(_make_ev(rr25_carry=0.010, rr25_rough=0.009,   # 10% improvement
                                bf25_carry=0.002, bf25_rough=0.0021))  # slightly worse
    verdict = classify_gate2_cell(agg)
    assert verdict == "PASS", f"Expected PASS, got {verdict}"


def test_classify_gate2_cell_marginal():
    """MARGINAL when rough is within ±5% of carry."""
    agg = _agg_metrics(_make_ev(rr25_carry=0.010, rr25_rough=0.0097,  # ~3% improvement
                                bf25_carry=0.002, bf25_rough=0.0021))
    verdict = classify_gate2_cell(agg)
    assert verdict == "MARGINAL", f"Expected MARGINAL, got {verdict}"


def test_classify_gate2_cell_fail():
    """FAIL when rough is worse than carry by >5% on both features."""
    agg = _agg_metrics(_make_ev(rr25_carry=0.010, rr25_rough=0.0115,  # 15% worse
                                bf25_carry=0.002, bf25_rough=0.0023))
    verdict = classify_gate2_cell(agg)
    assert verdict == "FAIL", f"Expected FAIL, got {verdict}"


def test_classify_gate3_cell_pass():
    """PASS when active regime beats carry, quiet does not."""
    active_ev = _make_ev(0.010, 0.0088, 0.002, 0.0017)  # rough beats carry by >5%
    quiet_ev  = _make_ev(0.010, 0.0115, 0.002, 0.0023)  # rough loses

    agg_act = _agg_metrics(active_ev)
    agg_qui = _agg_metrics(quiet_ev)
    verdict = classify_gate3_cell(agg_act, agg_qui)
    assert verdict == "PASS", f"Expected PASS, got {verdict}"


def test_classify_gate3_cell_fail():
    """FAIL when rough loses in active regime."""
    active_ev = _make_ev(0.010, 0.0115, 0.002, 0.0023)  # rough worse
    quiet_ev  = _make_ev(0.010, 0.0115, 0.002, 0.0023)
    verdict = classify_gate3_cell(_agg_metrics(active_ev), _agg_metrics(quiet_ev))
    assert verdict == "FAIL", f"Expected FAIL, got {verdict}"


def test_classify_gate3_cell_skip():
    """SKIP when active partition is None."""
    qui_ev = _make_ev(0.010, 0.009, 0.002, 0.0018)
    verdict = classify_gate3_cell(None, _agg_metrics(qui_ev))
    assert verdict == "SKIP", f"Expected SKIP, got {verdict}"


def test_classify_gate4_cell_pass():
    """PASS when a Gate 4 method improves on carry by >2%."""
    agg = _make_gate4_agg(0.010, 0.0096, 0.0101, 0.0020, 0.0021, 0.0018)
    verdict = classify_gate4_cell(agg)
    assert verdict == "PASS", f"Expected PASS, got {verdict}"


def test_classify_gate5_cell_pass():
    """PASS when active improves and quiet does not."""
    agg_act = _make_gate4_agg(0.010, 0.0097, 0.0102, 0.0020, 0.0021, 0.0020)
    agg_qui = _make_gate4_agg(0.010, 0.0102, 0.0101, 0.0020, 0.0021, 0.0020)
    verdict = classify_gate5_cell(agg_act, agg_qui)
    assert verdict == "PASS", f"Expected PASS, got {verdict}"


def test_cache_roundtrip():
    """save_cache → load_cache returns identical list."""
    records = _make_panel(n_bars=20)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = SweepConfig(gate_id="gate2", n_days=5, cache_dir=Path(tmp))
        save_cache(cfg, records)
        loaded = load_cache(cfg)

    assert loaded is not None, "Cache load returned None"
    assert len(loaded) == len(records)
    assert loaded[0]["rr25"] == records[0]["rr25"]


def test_format_gate2_summary_shape():
    """format_gate2_summary produces a non-empty string with expected labels."""
    rows = []
    for H in [0.05, 0.10]:
        for res in [1, 30]:
            rows.append({"H": H, "resample_min": res, "expiry": "ALL",
                         "n_bars": 100, "verdict": "PASS"})
    df = pd.DataFrame(rows)
    text = format_gate2_summary(df)
    assert "PASS" in text
    assert "0.05" in text
    assert "0.10" in text


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    tests = [
        ("resample_panel",                   test_resample_panel),
        ("apply_n_exp_selection_near",        test_apply_n_exp_selection_near),
        ("apply_n_exp_selection_far",         test_apply_n_exp_selection_far),
        ("recompute_structural_H_change",     test_recompute_structural_H_change),
        ("agg_metrics_keys",                  test_agg_metrics_keys),
        ("classify_gate2_cell_pass",          test_classify_gate2_cell_pass),
        ("classify_gate2_cell_marginal",      test_classify_gate2_cell_marginal),
        ("classify_gate2_cell_fail",          test_classify_gate2_cell_fail),
        ("classify_gate3_cell_pass",         test_classify_gate3_cell_pass),
        ("classify_gate3_cell_fail",         test_classify_gate3_cell_fail),
        ("classify_gate3_cell_skip",         test_classify_gate3_cell_skip),
        ("classify_gate4_cell_pass",          test_classify_gate4_cell_pass),
        ("classify_gate5_cell_pass",         test_classify_gate5_cell_pass),
        ("cache_roundtrip",                   test_cache_roundtrip),
        ("format_gate2_summary_shape",        test_format_gate2_summary_shape),
    ]

    print(f"\nrobustness_sweeps unit tests ({len(tests)} tests)")
    print("-" * 50)
    for name, fn in tests:
        _run(name, fn, args.verbose)

    n_pass  = sum(1 for _, s in _results if s == "PASS")
    n_fail  = sum(1 for _, s in _results if s in ("FAIL", "ERROR"))
    print("-" * 50)
    print(f"Results: {n_pass}/{len(tests)} passed"
          + (f"  ({n_fail} failed)" if n_fail else ""))

    if not args.verbose and n_pass == len(tests):
        print("All tests PASSED.")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
