"""
shared/smile_pipeline.py
========================
Reusable smile extraction and forecast evaluation primitives shared across
all rough-vol research benchmarks (skew_scaling, conditional_dynamics,
roughtemporal_intraday).

Public API
----------
Constants : RATE, DIV, MIN_DTE, MAX_DTE, MKT_OPEN_UTC, MKT_CLOSE_UTC
BS math   : _d1d2, _bs_call, _bs_put, _bs_delta_call, _bs_delta_put
IV solver : _vec_iv
Parsing   : _parse_sym_batch, _recover_forward
Features  : extract_features
Forecasts : ar1_forecast, evaluate_forecasts
Data I/O  : process_day_full
Utility   : _Tee
"""

import math
import re
import warnings
import zipfile
from datetime import date

import databento as db
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import pearsonr

# ── constants ──────────────────────────────────────────────────────────────────
RATE          = 0.053    # approx 3m T-bill 2025
DIV           = 0.013    # SPY annual dividend yield
MIN_DTE       = 7
MAX_DTE       = 60
MKT_OPEN_UTC  = 13 * 60 + 35    # 9:35 ET in UTC minutes
MKT_CLOSE_UTC = 19 * 60 + 55    # 15:55 ET in UTC minutes


# ── stdout tee ─────────────────────────────────────────────────────────────────
class _Tee:
    """Mirrors writes to multiple file-like objects simultaneously."""
    def __init__(self, *files):
        self.files = files
    def write(self, s):
        for f in self.files: f.write(s)
    def flush(self):
        for f in self.files: f.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised Black-Scholes (forward-price form)
# ─────────────────────────────────────────────────────────────────────────────

def _d1d2(sigma: np.ndarray, F: float, K: np.ndarray, T: float) -> tuple:
    sqrtT = math.sqrt(T)
    lmk   = np.log(F / K)
    d1    = (lmk + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2    = d1 - sigma * sqrtT
    return d1, d2

def _bs_call(sigma: np.ndarray, F: float, K: np.ndarray, T: float) -> np.ndarray:
    d1, d2 = _d1d2(sigma, F, K, T)
    return math.exp(-RATE * T) * (F * ndtr(d1) - K * ndtr(d2))

def _bs_put(sigma: np.ndarray, F: float, K: np.ndarray, T: float) -> np.ndarray:
    d1, d2 = _d1d2(sigma, F, K, T)
    return math.exp(-RATE * T) * (K * ndtr(-d2) - F * ndtr(-d1))

def _bs_delta_call(sigma: np.ndarray, F: float, K: np.ndarray, T: float) -> np.ndarray:
    d1, _ = _d1d2(sigma, F, K, T)
    return math.exp(-DIV * T) * ndtr(d1)

def _bs_delta_put(sigma: np.ndarray, F: float, K: np.ndarray, T: float) -> np.ndarray:
    d1, _ = _d1d2(sigma, F, K, T)
    return math.exp(-DIV * T) * (ndtr(d1) - 1.0)


def _vec_iv(prices: np.ndarray, F: float, K: np.ndarray, T: float,
            is_call: np.ndarray) -> np.ndarray:
    """
    Vectorised implied vol via bisection over sigma ∈ [1e-4, 4.0].
    60 halvings → machine-epsilon accuracy. Returns NaN outside no-arb bounds.
    """
    disc   = math.exp(-RATE * T)
    lo_arr = np.full(len(prices), 1e-4)
    hi_arr = np.full(len(prices), 4.0)

    fwd_disc = F * disc
    K_disc   = K * disc
    lb = np.where(is_call, np.maximum(0.0, fwd_disc - K_disc),
                            np.maximum(0.0, K_disc - fwd_disc))
    ub = np.where(is_call, fwd_disc, K_disc)
    valid = (prices > lb) & (prices < ub) & (prices > 0) & (T > 1e-5)

    sigma = np.full(len(prices), float("nan"))
    if not valid.any():
        return sigma

    p   = prices[valid]
    Kv  = K[valid]
    icv = is_call[valid]
    lo  = lo_arr[valid]
    hi  = hi_arr[valid]

    for _ in range(60):
        mid  = 0.5 * (lo + hi)
        pmid = np.where(icv,
                        _bs_call(mid, F, Kv, T),
                        _bs_put( mid, F, Kv, T))
        too_high = pmid > p
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo,  mid)

    sigma[valid] = 0.5 * (lo + hi)
    return sigma


# ─────────────────────────────────────────────────────────────────────────────
# Symbol parsing
# ─────────────────────────────────────────────────────────────────────────────
_SYM_RE = re.compile(r"SPY\s+(\d{6})([CP])(\d{8})")

def _parse_sym_batch(syms: pd.Series):
    """Vectorised OCC symbol parse → (expiry array, is_call array, strike array)."""
    expiry  = [None] * len(syms)
    is_call = [None] * len(syms)
    strike  = [None] * len(syms)
    for i, s in enumerate(syms):
        m = _SYM_RE.match(s.strip())
        if not m:
            continue
        ds, cp, ks = m.groups()
        expiry[i]  = date(2000 + int(ds[:2]), int(ds[2:4]), int(ds[4:6]))
        is_call[i] = cp == "C"
        strike[i]  = int(ks) / 1000.0
    return expiry, is_call, strike


# ─────────────────────────────────────────────────────────────────────────────
# Forward recovery
# ─────────────────────────────────────────────────────────────────────────────
def _recover_forward(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    """Median forward from C - P = disc·(F - K) across near-ATM strike pairs."""
    common = np.intersect1d(calls_df["strike"].values, puts_df["strike"].values)
    if len(common) < 1:
        return float("nan")
    c_by_k = calls_df.set_index("strike")["mid"]
    p_by_k = puts_df.set_index("strike")["mid"]
    fwds = []
    for K in common:
        if K not in c_by_k.index or K not in p_by_k.index:
            continue
        fwds.append(K + (c_by_k[K] - p_by_k[K]))
    if not fwds:
        return float("nan")
    return float(np.median(fwds))


# ─────────────────────────────────────────────────────────────────────────────
# Smile feature extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(slice_df: pd.DataFrame, T: float):
    """
    Per (timestamp, expiry) slice → smile features.

    Input columns: strike, is_call, mid, spread
    Returns dict {forward, atm_iv, atm_total_var, rr25, bf25} or None.
    """
    if T < 1e-4 or len(slice_df) < 6:
        return None

    calls = slice_df[slice_df["is_call"]].copy()
    puts  = slice_df[~slice_df["is_call"]].copy()
    if len(calls) < 3 or len(puts) < 3:
        return None

    F = _recover_forward(calls, puts)
    if not math.isfinite(F) or F <= 0:
        return None

    K_all  = slice_df["strike"].values.astype(float)
    p_all  = slice_df["mid"].values.astype(float)
    ic_all = slice_df["is_call"].values.astype(bool)
    iv_all = _vec_iv(p_all, F, K_all, T, ic_all)

    valid = np.isfinite(iv_all) & (iv_all > 0.02) & (iv_all < 3.0)
    if valid.sum() < 6:
        return None

    K_v  = K_all[valid]
    iv_v = iv_all[valid]
    ic_v = ic_all[valid]

    nearest_K = K_v[np.argmin(np.abs(K_v - F))]
    atm_mask  = K_v == nearest_K
    if atm_mask.sum() == 0:
        return None
    atm_iv = float(iv_v[atm_mask].mean())
    if atm_iv <= 0 or not math.isfinite(atm_iv):
        return None

    sigma_grid = np.where(iv_v > 0, iv_v, 0.01)
    deltas = np.where(ic_v,
                      _bs_delta_call(sigma_grid, F, K_v, T),
                      _bs_delta_put( sigma_grid, F, K_v, T))

    c_mask = ic_v & np.isfinite(deltas) & (deltas > 0.05) & (deltas < 0.70)
    p_mask = (~ic_v) & np.isfinite(deltas) & (deltas > -0.70) & (deltas < -0.05)
    if c_mask.sum() < 2 or p_mask.sum() < 2:
        return None

    def interp_at_delta(ivs, deltas, target):
        order = np.argsort(deltas)
        d = deltas[order]; v = ivs[order]
        if target < d[0] or target > d[-1]:
            return float("nan")
        idx = int(np.searchsorted(d, target))
        idx = min(max(idx, 1), len(d) - 1)
        t   = (target - d[idx-1]) / (d[idx] - d[idx-1] + 1e-12)
        return float(v[idx-1] + t * (v[idx] - v[idx-1]))

    iv_25c = interp_at_delta(iv_v[c_mask], deltas[c_mask],  0.25)
    iv_25p = interp_at_delta(iv_v[p_mask], deltas[p_mask], -0.25)

    if not (math.isfinite(iv_25c) and math.isfinite(iv_25p)):
        return None

    return {
        "forward":       F,
        "atm_iv":        atm_iv,
        "atm_total_var": atm_iv**2 * T,
        "rr25":          iv_25c - iv_25p,
        "bf25":          (iv_25c + iv_25p) / 2.0 - atm_iv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — full expiry window (no N_EXP cap)
# ─────────────────────────────────────────────────────────────────────────────
def process_day_full(zf: zipfile.ZipFile, fname: str, trade_date: date,
                     H: float, min_dte: int = MIN_DTE,
                     max_dte: int = MAX_DTE) -> list:
    """
    Load one DBN file and extract smile features for ALL expiries in
    [min_dte, max_dte] DTE — no N_EXP cap.

    Includes 'forward' in each record (needed for spot-return computation).

    Record fields: ts, expiry, T, forward, atm_iv, atm_total_var, rr25, bf25,
                   alpha, gamma
    """
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

    # ALL expiries in DTE window — no N_EXP limit
    all_exp  = np.array(sorted(valid["expiry"].unique()))
    selected = [e for e in all_exp
                if min_dte <= (e - trade_date).days <= max_dte]
    if not selected:
        return []

    valid = valid[valid["expiry"].isin(selected)]
    valid = valid[valid["spread"] < 2.0]

    records = []
    grouped = valid.groupby([valid.index, "expiry"])
    for (ts, exp), grp in grouped:
        T_cal = (exp - trade_date).days / 365.0
        if T_cal < min_dte / 365.0:
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
            "forward":       feats["forward"],   # included for spot-return computation
            "atm_iv":        feats["atm_iv"],
            "atm_total_var": atv,
            "rr25":          feats["rr25"],
            "bf25":          feats["bf25"],
            "alpha":         alpha,
            "gamma":         gamma,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Forecast evaluation (temporal)
# ─────────────────────────────────────────────────────────────────────────────
def ar1_forecast(history: list) -> float:
    if len(history) < 30:
        return history[-1]
    y = np.array(history)
    X = np.column_stack([np.ones(len(y)-1), y[:-1]])
    try:
        beta = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        return float(beta[0] + beta[1]*y[-1])
    except Exception:
        return history[-1]


def evaluate_forecasts(ts_data: list, H: float) -> dict:
    """
    1-step-ahead forecast evaluation for one expiry time series.

    Forecasters: carry, rough-structural (α/γ rolling median), AR(1)
    Metrics: RMSE, directional accuracy, residual ACF(1), structural stability

    Returns nested dict with keys:
      atm_total_var / rr25 / bf25 → {carry/rough/ar1 → {rmse, dir}}
      _meta → {n_bars, rr25_resid_acf, rr25_resid_pval, bf25_resid_acf,
                bf25_resid_pval, alpha_mean, alpha_std, alpha_cv,
                gamma_mean, gamma_std, gamma_cv}
      _dist_{feat} → {n, mean, std, min, p25, p50, p75, max}
    """
    ts_data.sort(key=lambda r: r["ts"])
    n = len(ts_data)

    atv  = [r["atm_total_var"] for r in ts_data]
    rr25 = [r["rr25"]          for r in ts_data]
    bf25 = [r["bf25"]          for r in ts_data]
    aiv  = [r["atm_iv"]        for r in ts_data]
    T_   = [r["T"]             for r in ts_data]
    alph = [r["alpha"]         for r in ts_data]
    gamm = [r["gamma"]         for r in ts_data]

    fc  = {m: {f: [] for f in ["atm_total_var","rr25","bf25"]}
           for m in ["carry","rough","ar1"]}
    act = {f: [] for f in ["atm_total_var","rr25","bf25"]}

    alpha_hist, gamma_hist = [], []

    for i in range(n - 1):
        act["atm_total_var"].append(atv[i+1])
        act["rr25"].append(rr25[i+1])
        act["bf25"].append(bf25[i+1])

        fc["carry"]["atm_total_var"].append(atv[i])
        fc["carry"]["rr25"].append(rr25[i])
        fc["carry"]["bf25"].append(bf25[i])

        if math.isfinite(alph[i]):
            alpha_hist.append(alph[i])
        if math.isfinite(gamm[i]):
            gamma_hist.append(gamm[i])

        Ti = T_[i]
        fc["rough"]["atm_total_var"].append(atv[i])
        fc["rough"]["rr25"].append(
            np.median(alpha_hist) * Ti**(H-0.5) * aiv[i]
            if alpha_hist else rr25[i])
        fc["rough"]["bf25"].append(
            np.median(gamma_hist) * Ti**(2*H-1) * atv[i]
            if gamma_hist else bf25[i])

        fc["ar1"]["atm_total_var"].append(ar1_forecast(atv[:i+1]))
        fc["ar1"]["rr25"].append(ar1_forecast(rr25[:i+1]))
        fc["ar1"]["bf25"].append(ar1_forecast(bf25[:i+1]))

    def rmse(errs):
        return float(np.sqrt(np.mean(np.array(errs)**2))) if errs else float("nan")

    def dir_acc(actual_list, forecast_list):
        correct = tot = 0
        for i in range(1, len(actual_list)):
            da = actual_list[i] - actual_list[i-1]
            df = forecast_list[i-1] - actual_list[i-1]
            if abs(da) < 1e-9 or abs(df) < 1e-9:
                continue
            correct += int((da > 0) == (df > 0))
            tot += 1
        return correct / tot if tot else float("nan")

    out = {}
    for feat in ["atm_total_var","rr25","bf25"]:
        a_arr = np.array(act[feat])
        out[feat] = {
            m: {
                "rmse": rmse((np.array(fc[m][feat]) - a_arr).tolist()),
                "dir":  dir_acc(act[feat], fc[m][feat]),
            }
            for m in ["carry","rough","ar1"]
        }

    def acf1(x):
        x = np.array(x); x = x[np.isfinite(x)]
        if len(x) < 10:
            return float("nan"), float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r, p = pearsonr(x[:-1], x[1:])
        return float(r), float(p)

    rr_resid = np.array(act["rr25"]) - np.array(fc["rough"]["rr25"])
    bf_resid = np.array(act["bf25"]) - np.array(fc["rough"]["bf25"])
    rr_acf, rr_p = acf1(rr_resid)
    bf_acf, bf_p = acf1(bf_resid)

    af = [v for v in alph if math.isfinite(v)]
    gf = [v for v in gamm if math.isfinite(v)]
    acv = np.std(af)/abs(np.mean(af)) if af and abs(np.mean(af)) > 1e-8 else float("nan")
    gcv = np.std(gf)/abs(np.mean(gf)) if gf and abs(np.mean(gf)) > 1e-8 else float("nan")

    for feat, vals in [("atm_total_var", atv), ("rr25", rr25), ("bf25", bf25)]:
        arr = np.array([v for v in vals if math.isfinite(v)])
        if len(arr) >= 4:
            out[f"_dist_{feat}"] = {
                "n":    len(arr),
                "mean": float(np.mean(arr)),
                "std":  float(np.std(arr)),
                "min":  float(np.min(arr)),
                "p25":  float(np.percentile(arr, 25)),
                "p50":  float(np.median(arr)),
                "p75":  float(np.percentile(arr, 75)),
                "max":  float(np.max(arr)),
            }

    out["_meta"] = {
        "n_bars":          n,
        "rr25_resid_acf":  rr_acf,  "rr25_resid_pval": rr_p,
        "bf25_resid_acf":  bf_acf,  "bf25_resid_pval": bf_p,
        "alpha_mean":      float(np.mean(af)) if af else float("nan"),
        "alpha_std":       float(np.std(af))  if af else float("nan"),
        "alpha_cv":        float(acv),
        "gamma_mean":      float(np.mean(gf)) if gf else float("nan"),
        "gamma_std":       float(np.std(gf))  if gf else float("nan"),
        "gamma_cv":        float(gcv),
    }
    return out
