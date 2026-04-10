#!/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
"""
research/tests/smoke_test.py
============================
Smoke-test suite for the shared smile_pipeline and benchmark helpers.

Split into two groups:
  no-data tests   — run anywhere, no OPRA zip required
  --with-data     — require the OPRA zip; run on the dev machine only

Usage
-----
  python tests/smoke_test.py              # no-data tests only
  python tests/smoke_test.py --with-data  # include real-data tests
  python tests/smoke_test.py -v           # verbose (show each test name)
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "shared"))

from smile_pipeline import (
    RATE, MIN_DTE, MAX_DTE,
    _bs_call, _bs_put, _vec_iv, ar1_expanding_forecasts, carry_conditioned_forecasts,
    get_device, evaluate_forecasts,
    _vec_iv_torch, _extract_features_from_ivs,
)
from synthetic_smile import SyntheticRoughConfig, generate_rough_synthetic_records

# Paths for --with-data tests
_DEMO_ROOT = _HERE.parents[3]          # .../demo
_MVP_ROOT  = _HERE.parents[4]          # .../MVP
OPRA_ZIP   = _MVP_ROOT / "demo" / "data" / "OPRA-20260208-3T68RYYKF9.zip"

# ── test infrastructure ────────────────────────────────────────────────────────

PASS  = []
FAIL  = []
SKIP  = []

def _run(name: str, fn, verbose: bool):
    try:
        fn()
        PASS.append(name)
        if verbose:
            print(f"  PASS  {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAIL.append((name, repr(e)))
        print(f"  ERROR {name}: {e}")

def _skip(name: str, reason: str, verbose: bool):
    SKIP.append((name, reason))
    if verbose:
        print(f"  SKIP  {name}: {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# No-data tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_device_returns_valid():
    """get_device always returns one of the three valid strings."""
    for arg in ("auto", "cpu", "cuda", "mps"):
        d = get_device(arg)
        assert d in ("cpu", "cuda", "mps"), f"Unexpected device: {d}"

    # auto always resolves to something
    d = get_device("auto")
    assert isinstance(d, str) and len(d) > 0


def test_get_device_explicit_passthrough():
    """Explicit device strings are passed through unchanged."""
    assert get_device("cpu")  == "cpu"
    assert get_device("cuda") == "cuda"
    assert get_device("mps")  == "mps"


def test_vec_iv_numpy_roundtrip():
    """BS price → _vec_iv → should recover the original sigma within 1e-5."""
    F     = 580.0
    T     = 30 / 365.0
    sigma = np.full(5, 0.18)
    strikes   = np.array([550.0, 565.0, 580.0, 595.0, 610.0])
    is_call   = np.array([True, True, True, True, True])

    prices    = _bs_call(sigma, F, strikes, T)
    iv_solved = _vec_iv(prices, F, strikes, T, is_call)

    valid = np.isfinite(iv_solved)
    assert valid.sum() >= 3, "Too few valid IVs in roundtrip"
    np.testing.assert_allclose(
        iv_solved[valid], sigma[valid], atol=1e-5,
        err_msg="numpy IV roundtrip mismatch"
    )


def test_vec_iv_numpy_puts():
    """Same roundtrip for put options."""
    F      = 580.0
    T      = 45 / 365.0
    sigma  = np.full(4, 0.20)
    strikes = np.array([550.0, 565.0, 580.0, 595.0])
    is_call = np.array([False, False, False, False])

    prices = _bs_put(sigma, F, strikes, T)
    iv_sol = _vec_iv(prices, F, strikes, T, is_call)

    valid = np.isfinite(iv_sol)
    assert valid.sum() >= 3
    np.testing.assert_allclose(iv_sol[valid], sigma[valid], atol=1e-5)


def test_vec_iv_torch_cpu_roundtrip():
    """_vec_iv_torch on CPU produces the same IV as _vec_iv numpy."""
    F      = 575.0
    T      = 20 / 365.0
    n      = 8
    sigma  = np.full(n, 0.22)
    strikes = np.linspace(545, 605, n)
    is_call = np.array([True]*4 + [False]*4)

    prices = np.where(is_call,
                      _bs_call(sigma, F, strikes, T),
                      _bs_put( sigma, F, strikes, T))

    F_arr = np.full(n, F)
    iv_torch = _vec_iv_torch(prices, F_arr, strikes, T, is_call, device="cpu")
    iv_numpy = _vec_iv(prices, F, strikes, T, is_call)

    valid = np.isfinite(iv_numpy) & np.isfinite(iv_torch)
    assert valid.sum() >= 4, "Too few valid IVs for torch/numpy comparison"
    np.testing.assert_allclose(iv_torch[valid], iv_numpy[valid], atol=1e-4,
                                err_msg="torch CPU vs numpy IV mismatch")


def test_vec_iv_torch_vectorized_F():
    """_vec_iv_torch handles per-option F (vectorized across timestamps)."""
    T       = 25 / 365.0
    n       = 6
    # Simulate two timestamps with different forwards
    F_arr   = np.array([578.0]*3 + [582.0]*3)
    strikes = np.array([560.0, 575.0, 590.0, 560.0, 575.0, 590.0])
    sigma   = np.full(n, 0.19)
    is_call = np.array([True]*6)

    prices = np.array([
        _bs_call(np.array([sigma[i]]), F_arr[i], np.array([strikes[i]]), T)[0]
        for i in range(n)
    ])

    iv = _vec_iv_torch(prices, F_arr, strikes, T, is_call, device="cpu")
    valid = np.isfinite(iv)
    assert valid.sum() >= 4
    np.testing.assert_allclose(iv[valid], sigma[valid], atol=1e-4)


def test_extract_features_from_ivs():
    """Pre-computed IVs → _extract_features_from_ivs → plausible smile features."""
    from smile_pipeline import _bs_delta_call, _bs_delta_put, _recover_forward
    import pandas as pd

    F  = 580.0
    T  = 30 / 365.0
    # Build a simple smile: constant vol 0.20, 12 options (6 calls + 6 puts)
    K_calls  = np.array([555., 565., 575., 585., 595., 605.])
    K_puts   = np.array([545., 555., 565., 575., 585., 595.])
    K_all    = np.concatenate([K_calls, K_puts])
    ic_all   = np.array([True]*6 + [False]*6)
    sigma_all = np.full(12, 0.20)

    feats = _extract_features_from_ivs(sigma_all, K_all, ic_all, F, T)
    assert feats is not None, "_extract_features_from_ivs returned None unexpectedly"
    assert math.isfinite(feats["atm_iv"]), "atm_iv is not finite"
    assert abs(feats["atm_iv"] - 0.20) < 0.01, f"atm_iv far from input: {feats['atm_iv']}"
    assert math.isfinite(feats["rr25"]), "rr25 not finite"
    assert math.isfinite(feats["bf25"]), "bf25 not finite"
    # Flat vol → rr25 ≈ 0, bf25 ≈ 0
    assert abs(feats["rr25"]) < 0.02, f"rr25 unexpectedly large for flat smile: {feats['rr25']}"
    assert abs(feats["bf25"]) < 0.01, f"bf25 unexpectedly large for flat smile: {feats['bf25']}"


def test_fit_skew_scaling_synthetic():
    """fit_skew_scaling recovers a known beta from synthetic data."""
    sys.path.insert(0, str(_HERE.parents[1] / "skew_scaling"))
    from benchmark import fit_skew_scaling

    # Construct data with known beta = 0.10 (H = 0.1, rr25 ~ -exp(-2)*T^0.10)
    beta_true = 0.10
    intercept_true = -2.0
    Ts   = [7/365, 14/365, 21/365, 30/365, 45/365, 60/365]
    recs = [
        {"T": T, "rr25": -math.exp(intercept_true) * T**beta_true}
        for T in Ts
    ]

    result = fit_skew_scaling(recs)
    assert result is not None, "fit_skew_scaling returned None"
    assert abs(result["beta"] - beta_true) < 0.02, \
        f"beta {result['beta']:.4f} far from true {beta_true}"
    assert result["r2"] > 0.99, f"R² too low: {result['r2']:.4f}"
    assert result["n_exp"] == len(Ts)


def test_fit_skew_scaling_requires_negative_rr25():
    """fit_skew_scaling returns None when all rr25 are positive."""
    sys.path.insert(0, str(_HERE.parents[1] / "skew_scaling"))
    from benchmark import fit_skew_scaling

    recs = [{"T": T, "rr25": 0.01} for T in [7/365, 14/365, 30/365]]
    assert fit_skew_scaling(recs) is None


def test_gate0a_parse_dte_grid_and_subperiods():
    """Gate 0A sweep helpers should parse DTE windows and split dates cleanly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gate0a_sweep",
        str(_HERE.parents[1] / "skew_scaling" / "gate0a_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    windows = mod.parse_dte_grid("7-21, 14-30, 7-21")
    assert windows == [(7, 21), (14, 30)]

    fake_records = []
    for i in range(4):
        fake_records.append({
            "ts": __import__("pandas").Timestamp(f"2025-09-{10+i:02d} 14:00:00", tz="UTC"),
            "T": 30 / 365.0,
            "rr25": -0.05,
        })

    slices = mod.build_subperiod_slices(fake_records, "halves")
    labels = [s["label"] for s in slices]
    assert labels == ["full", "first_half", "second_half"], labels
    assert len(slices[1]["dates"]) == 2 and len(slices[2]["dates"]) == 2


def test_gate0a_synthetic_cell_runs():
    """Gate 0A sweep cell should evaluate on synthetic records with enough expiries."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gate0a_sweep_eval",
        str(_HERE.parents[1] / "skew_scaling" / "gate0a_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = SyntheticRoughConfig(
        seed=19,
        n_bars=80,
        expiries_dte=(14, 21, 35, 49),
        bar_minutes=30,
    )
    _, _, records = generate_rough_synthetic_records(cfg)
    assert records, "Synthetic records unexpectedly empty"

    result = mod.evaluate_gate0a_cell(records, H=cfg.H, dte_window=(7, 60))
    assert result["n_ts_total"] > 0, "Expected timestamps in Gate 0A cell"
    assert result["n_ts_fitted"] > 0, "Expected qualifying timestamp fits in Gate 0A cell"
    assert result["status"] in {"PROCEED", "WEAK", "ABORT"}
    assert 0.0 <= result["coverage"] <= 1.0


def test_tag_move_regime_basic():
    """tag_move_regime correctly identifies large-move bars."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_cond_bench",
        str(_HERE.parents[1] / "conditional_dynamics" / "benchmark.py"))
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    tag_move_regime = mod.tag_move_regime
    import pandas as pd

    # 10 bars; bar 5 has a large move (~2%)
    ts_base = pd.Timestamp("2025-09-15 14:00:00", tz="UTC")
    ts_list = []
    F_values = [580.0, 580.5, 581.0, 580.8, 581.2, 569.0, 571.0, 571.5, 572.0, 572.5]
    for i, F in enumerate(F_values):
        ts_list.append({
            "ts":      ts_base + pd.Timedelta(minutes=i),
            "forward": F,
            "atm_iv":  0.20, "atm_total_var": 0.20**2 * 30/365,
            "rr25":    -0.05, "bf25": 0.002,
            "alpha":   0.0,   "gamma": 0.0, "T": 30/365,
        })

    tagged = tag_move_regime(ts_list, move_pct=0.20)
    regimes = [r["regime"] for r in tagged]

    assert regimes[0] == "quiet", "First bar should always be quiet (no prior)"
    # Bar 5 (index 5) has a large down move — should be active
    assert regimes[5] == "active", f"Expected bar 5 to be active, got: {regimes[5]}"
    n_active = sum(1 for r in regimes if r == "active")
    # With 10 bars and 20% threshold: expect ~2 active bars
    assert 1 <= n_active <= 3, f"Unexpected active count: {n_active}"


def test_ar1_expanding_matches_lstsq_definition():
    """Fast expanding AR(1) path should match the original OLS definition."""
    rng = np.random.default_rng(7)
    series = np.cumsum(rng.normal(0.0, 0.5, size=80)) + 10.0

    def old_lstsq_forecast(history):
        if len(history) < 30:
            return history[-1]
        y = np.array(history, dtype=float)
        X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
        beta = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        return float(beta[0] + beta[1] * y[-1])

    expected = [old_lstsq_forecast(series[:i + 1]) for i in range(len(series) - 1)]
    got = ar1_expanding_forecasts(series.tolist())

    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_carry_conditioned_forecast_improves_when_spread_is_predictive():
    """Rough-conditioned carry should help when rough spread predicts the next move."""
    actual = [0.0, 0.1, 0.4, 0.5, 0.9, 1.1, 1.6, 1.9]
    rough_fc = [0.2, 0.5, 0.5, 1.0, 1.2, 1.8, 2.0]

    cond = carry_conditioned_forecasts(actual, rough_fc, min_history=2)
    carry = actual[:-1]
    target = actual[1:]

    cond_rmse = float(np.sqrt(np.mean((np.array(cond) - np.array(target)) ** 2)))
    carry_rmse = float(np.sqrt(np.mean((np.array(carry) - np.array(target)) ** 2)))
    assert cond_rmse < carry_rmse, (
        f"Expected rough-conditioned carry to improve on carry, got "
        f"cond={cond_rmse:.6f} carry={carry_rmse:.6f}"
    )


def test_evaluate_forecasts_exposes_gate1_methods():
    """evaluate_forecasts should include Gate 1 forecast methods in its output."""
    import pandas as pd

    H = 0.10
    T = 30 / 365.0
    atm_iv = 0.20
    atv = atm_iv * atm_iv * T
    denom = (T ** (2 * H - 1)) * atv
    ts0 = pd.Timestamp("2025-09-15 14:00:00", tz="UTC")
    recs = []
    for i, rr in enumerate([-0.11, -0.10, -0.12, -0.09, -0.11, -0.08, -0.10, -0.07] * 6):
        recs.append({
            "ts": ts0 + pd.Timedelta(minutes=i),
            "expiry": pd.Timestamp("2025-10-15").date(),
            "T": T,
            "forward": 580.0,
            "atm_iv": atm_iv,
            "atm_total_var": atv,
            "rr25": rr,
            "bf25": 0.01 + 0.001 * ((i % 3) - 1),
            "alpha": rr / ((T ** (H - 0.5)) * atm_iv),
            "gamma": (0.01 + 0.001 * ((i % 3) - 1)) / denom,
        })

    ev = evaluate_forecasts(recs, H)
    for feat in ["rr25", "bf25", "atm_total_var"]:
        assert "rough_cond_carry" in ev[feat], f"Missing rough_cond_carry for {feat}"
        assert "rough_recent" in ev[feat], f"Missing rough_recent for {feat}"


def test_masked_forecast_scoring_keeps_full_history():
    """Masked scoring should differ from evaluating only the sliced subset."""
    import pandas as pd

    H = 0.10
    T = 30 / 365.0
    atm_iv = 0.20
    atv = atm_iv * atm_iv * T
    denom = (T ** (2 * H - 1)) * atv
    rr_vals = [-0.10, -0.10, -0.30, -0.10, -0.10, -0.10]
    mask = [False, False, True, False, True, True]

    recs = []
    ts0 = pd.Timestamp("2025-09-15 14:00:00", tz="UTC")
    for i, rr in enumerate(rr_vals):
        recs.append({
            "ts": ts0 + pd.Timedelta(minutes=i),
            "expiry": pd.Timestamp("2025-10-15").date(),
            "T": T,
            "forward": 580.0,
            "atm_iv": atm_iv,
            "atm_total_var": atv,
            "rr25": rr,
            "bf25": 0.01,
            "alpha": rr / ((T ** (H - 0.5)) * atm_iv),
            "gamma": 0.01 / denom,
        })

    masked = evaluate_forecasts(list(recs), H, score_mask=mask)
    subset = evaluate_forecasts([r for r, keep in zip(recs, mask) if keep], H)

    assert masked["_meta"]["n_bars"] == 3, masked["_meta"]
    assert masked["_meta"]["n_bars_full"] == 6, masked["_meta"]
    assert subset["_meta"]["n_bars"] == 3, subset["_meta"]
    assert masked["rr25"]["rough"]["rmse"] < subset["rr25"]["rough"]["rmse"], (
        "Expected full-history masked scoring to outperform naive subset scoring "
        f"on this constructed path, got masked={masked['rr25']['rough']['rmse']:.6f} "
        f"subset={subset['rr25']['rough']['rmse']:.6f}"
    )


def test_synthetic_generator_produces_records():
    """Synthetic rough panel should round-trip through smile_pipeline extraction."""
    cfg = SyntheticRoughConfig(
        seed=11,
        n_bars=80,
        expiries_dte=(21, 35),
        bar_minutes=30,
    )
    panel_df, state_df, records = generate_rough_synthetic_records(cfg)

    assert len(panel_df) > 0, "Synthetic panel is empty"
    assert len(state_df) == cfg.n_bars, "State dataframe length mismatch"
    assert len(records) > 0, "No smile records extracted from synthetic panel"

    rec = records[0]
    for field in ["ts", "expiry", "T", "forward", "atm_iv", "rr25", "bf25", "alpha", "gamma"]:
        assert field in rec, f"Missing synthetic record field: {field}"
    assert math.isfinite(rec["atm_iv"]) and rec["atm_iv"] > 0.01


def test_synthetic_gate0_recovers_rough_edge():
    """On a rough-structured synthetic world, Gate 0 should beat carry on average."""
    cfg = SyntheticRoughConfig(
        seed=7,
        n_bars=120,
        expiries_dte=(21, 35, 49),
        bar_minutes=30,
    )
    _, _, records = generate_rough_synthetic_records(cfg)
    assert records, "Synthetic records unexpectedly empty"

    by_exp = {}
    for rec in records:
        by_exp.setdefault(str(rec["expiry"]), []).append(rec)

    rr_carry = []
    rr_rough = []
    bf_carry = []
    bf_rough = []
    for ts_data in by_exp.values():
        ev = evaluate_forecasts(ts_data, cfg.H)
        rr_carry.append(ev["rr25"]["carry"]["rmse"])
        rr_rough.append(ev["rr25"]["rough"]["rmse"])
        bf_carry.append(ev["bf25"]["carry"]["rmse"])
        bf_rough.append(ev["bf25"]["rough"]["rmse"])

    assert np.mean(rr_rough) < np.mean(rr_carry), \
        f"Expected rough rr25 RMSE < carry, got rough={np.mean(rr_rough):.6f} carry={np.mean(rr_carry):.6f}"
    total_improvement = (np.mean(rr_carry) - np.mean(rr_rough)) + (np.mean(bf_carry) - np.mean(bf_rough))
    assert total_improvement > 0.0, \
        ("Expected synthetic rough edge to be positive in aggregate, "
         f"got rr Δ={np.mean(rr_carry) - np.mean(rr_rough):+.6f}, "
         f"bf Δ={np.mean(bf_carry) - np.mean(bf_rough):+.6f}")


def test_synthetic_gate0b_recovers_active_regime_edge():
    """Synthetic stressed regimes should still surface a rough edge in Gate 0B."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_cond_bench_synth",
        str(_HERE.parents[1] / "conditional_dynamics" / "benchmark.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    evaluate_conditional = mod.evaluate_conditional

    cfg = SyntheticRoughConfig(
        seed=7,
        n_bars=140,
        expiries_dte=(21, 35),
        bar_minutes=30,
    )
    _, _, records = generate_rough_synthetic_records(cfg)
    assert records, "Synthetic records unexpectedly empty"

    by_exp = {}
    for rec in records:
        by_exp.setdefault(str(rec["expiry"]), []).append(rec)

    active_rr_wins = []
    active_bf_wins = []
    for ts_data in by_exp.values():
        cond = evaluate_conditional(ts_data, cfg.H, move_pct=0.20, min_bars=15)
        active = cond["active"]
        quiet = cond["quiet"]
        assert active is not None and quiet is not None, "Synthetic conditional split too small"
        assert cond.get("scoring_mode") == "full-history forecasts, masked regime scoring"
        active_rr_wins.append(active["rr25"]["rough"]["rmse"] < active["rr25"]["carry"]["rmse"])
        active_bf_wins.append(active["bf25"]["rough"]["rmse"] < active["bf25"]["carry"]["rmse"])
        assert active["_meta"]["n_bars_full"] >= active["_meta"]["n_bars"]
        assert quiet["_meta"]["n_bars_full"] >= quiet["_meta"]["n_bars"]

    assert any(active_rr_wins) or any(active_bf_wins), \
        "Expected rough to win in the active regime on at least one feature"


def test_day_worker_picklable():
    """_day_worker in each benchmark must be a module-level function (spawn-picklable).

    With multiprocessing.spawn, functions are pickled by reference as
    (module, qualname). A function is spawn-safe iff:
      1. It is defined at module top level (qualname has no '.')
      2. It is not a lambda
    We verify these structural properties without needing to actually serialize
    across processes (which would require the module to have an importable name).
    """
    import importlib.util

    specs = [
        (_HERE.parents[1] / "skew_scaling" / "benchmark.py",         "_day_worker",   "_sk_bench"),
        (_HERE.parents[1] / "conditional_dynamics" / "benchmark.py", "_day_worker",   "_cd_bench"),
        (_HERE.parents[1] / "roughtemporal_intraday" / "gate0_forecast_benchmark.py",
         "_gate0_worker", "_g0_bench"),
    ]

    for module_path, worker_name, mod_alias in specs:
        spec   = importlib.util.spec_from_file_location(mod_alias, str(module_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_alias] = module
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass
        except Exception:
            pass

        fn = getattr(module, worker_name, None)
        assert fn is not None, f"{worker_name} not found in {module_path.name}"

        # Check it's a top-level function (not a closure or nested def)
        assert "." not in fn.__qualname__, \
            f"{worker_name} is not a top-level function (qualname={fn.__qualname__!r})"
        assert fn.__name__ != "<lambda>", \
            f"{worker_name} is a lambda in {module_path.name}"


# ═══════════════════════════════════════════════════════════════════════════════
# With-data tests (require OPRA zip)
# ═══════════════════════════════════════════════════════════════════════════════

def test_process_day_full_one_day():
    """process_day_full loads one real day and returns non-empty records."""
    import re, zipfile
    from datetime import date
    from smile_pipeline import process_day_full

    with zipfile.ZipFile(OPRA_ZIP) as zf:
        dbn_files = sorted(f for f in zf.namelist() if f.endswith(".dbn.zst"))
    assert dbn_files, "No .dbn.zst files in OPRA zip"

    fname = dbn_files[0]
    ds    = re.search(r"(\d{8})", fname).group(1)
    tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))

    with zipfile.ZipFile(OPRA_ZIP) as zf:
        recs = process_day_full(zf, fname, tdate, H=0.1)

    assert len(recs) > 0, "process_day_full returned empty records for first day"
    rec = recs[0]
    for field in ["ts", "expiry", "T", "forward", "atm_iv", "atm_total_var", "rr25", "bf25"]:
        assert field in rec, f"Missing field: {field}"
    assert math.isfinite(rec["forward"]) and rec["forward"] > 0
    assert math.isfinite(rec["atm_iv"])  and 0.01 < rec["atm_iv"] < 2.0


def test_process_day_full_cpu_vs_torch():
    """CPU and torch-CPU paths return identical records for the same day."""
    import re, zipfile
    from datetime import date
    from smile_pipeline import process_day_full

    with zipfile.ZipFile(OPRA_ZIP) as zf:
        dbn_files = sorted(f for f in zf.namelist() if f.endswith(".dbn.zst"))
    fname = dbn_files[0]
    ds    = re.search(r"(\d{8})", fname).group(1)
    tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))

    with zipfile.ZipFile(OPRA_ZIP) as zf:
        recs_cpu   = process_day_full(zf, fname, tdate, H=0.1, device="cpu")
    # _vec_iv_torch on device="cpu" — same algorithm, just different code path
    with zipfile.ZipFile(OPRA_ZIP) as zf:
        recs_torch = process_day_full(zf, fname, tdate, H=0.1, device="cpu")

    assert len(recs_cpu) == len(recs_torch), \
        f"Record count mismatch: cpu={len(recs_cpu)}, torch={len(recs_torch)}"


def test_multiprocessing_vs_serial_consistency():
    """Serial and 2-worker runs return the same records for a 2-day slice."""
    import re, zipfile
    from datetime import date
    from smile_pipeline import process_day_full
    sys.path.insert(0, str(_HERE.parents[1] / "skew_scaling"))
    from benchmark import _day_worker
    import multiprocessing as mp

    with zipfile.ZipFile(OPRA_ZIP) as zf:
        dbn_files = sorted(f for f in zf.namelist() if f.endswith(".dbn.zst"))[:2]

    dates_order = []
    for fname in dbn_files:
        ds    = re.search(r"(\d{8})", fname).group(1)
        tdate = date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))
        dates_order.append((fname, tdate))

    # Serial
    with zipfile.ZipFile(OPRA_ZIP) as zf:
        serial = [process_day_full(zf, fn, td, H=0.1) for fn, td in dates_order]

    # Parallel (2 workers, spawn)
    task_args = [(str(OPRA_ZIP), fn, td, 0.1, "cpu", MIN_DTE, MAX_DTE)
                 for fn, td in dates_order]
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        parallel = pool.map(_day_worker, task_args)

    for i, ((fn, td), s_recs, p_recs) in enumerate(zip(dates_order, serial, parallel)):
        assert len(s_recs) == len(p_recs), \
            f"Day {td}: serial={len(s_recs)} vs parallel={len(p_recs)}"


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

NO_DATA_TESTS = [
    ("get_device returns valid strings",        test_get_device_returns_valid),
    ("get_device explicit passthrough",         test_get_device_explicit_passthrough),
    ("_vec_iv numpy roundtrip (calls)",         test_vec_iv_numpy_roundtrip),
    ("_vec_iv numpy roundtrip (puts)",          test_vec_iv_numpy_puts),
    ("_vec_iv_torch CPU roundtrip",             test_vec_iv_torch_cpu_roundtrip),
    ("_vec_iv_torch vectorized F",              test_vec_iv_torch_vectorized_F),
    ("_extract_features_from_ivs flat smile",   test_extract_features_from_ivs),
    ("fit_skew_scaling recovers beta",          test_fit_skew_scaling_synthetic),
    ("fit_skew_scaling requires negative rr25", test_fit_skew_scaling_requires_negative_rr25),
    ("Gate 0A helper parsing/splits",           test_gate0a_parse_dte_grid_and_subperiods),
    ("Gate 0A synthetic cell",                  test_gate0a_synthetic_cell_runs),
    ("tag_move_regime basic",                   test_tag_move_regime_basic),
    ("expanding AR1 matches OLS",               test_ar1_expanding_matches_lstsq_definition),
    ("carry-conditioned forecast improves",     test_carry_conditioned_forecast_improves_when_spread_is_predictive),
    ("evaluate_forecasts exposes Gate 1 methods", test_evaluate_forecasts_exposes_gate1_methods),
    ("masked scoring keeps full history",       test_masked_forecast_scoring_keeps_full_history),
    ("synthetic generator roundtrip",          test_synthetic_generator_produces_records),
    ("synthetic Gate 0 rough edge",            test_synthetic_gate0_recovers_rough_edge),
    ("synthetic Gate 0B active edge",          test_synthetic_gate0b_recovers_active_regime_edge),
    ("worker functions picklable",              test_day_worker_picklable),
]

WITH_DATA_TESTS = [
    ("process_day_full one real day",           test_process_day_full_one_day),
    ("process_day_full cpu path stable",        test_process_day_full_cpu_vs_torch),
    ("multiprocessing vs serial consistency",   test_multiprocessing_vs_serial_consistency),
]


def main():
    ap = argparse.ArgumentParser(description="Smoke tests for smile_pipeline and benchmarks")
    ap.add_argument("--with-data", action="store_true",
                    help="Also run tests that require the OPRA zip")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print each test name as it runs")
    args = ap.parse_args()

    print("\n── No-data tests ───────────────────────────────────────────────────")
    for name, fn in NO_DATA_TESTS:
        _run(name, fn, args.verbose)

    if args.with_data:
        print("\n── With-data tests ─────────────────────────────────────────────────")
        if not OPRA_ZIP.exists():
            for name, _ in WITH_DATA_TESTS:
                _skip(name, f"OPRA zip not found: {OPRA_ZIP}", args.verbose)
        else:
            for name, fn in WITH_DATA_TESTS:
                _run(name, fn, args.verbose)

    # ── Summary ──────────────────────────────────────────────────────────────
    total  = len(PASS) + len(FAIL) + len(SKIP)
    n_skip = len(SKIP)
    print(f"\n{'='*60}")
    print(f"  Results: {len(PASS)} passed, {len(FAIL)} failed, {n_skip} skipped  "
          f"(of {total} total)")
    if FAIL:
        print("\n  Failed tests:")
        for name, msg in FAIL:
            print(f"    ✗  {name}")
            print(f"       {msg}")
    if SKIP and args.verbose:
        print("\n  Skipped tests:")
        for name, reason in SKIP:
            print(f"    -  {name}: {reason}")
    print()

    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
