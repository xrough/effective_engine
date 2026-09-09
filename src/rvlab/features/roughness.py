"""
rvlab.features.roughness
========================
Two independent ways to measure H, which is the point of notebook 04.

1. **Time-series** (`hurst_structure_function`) — the physical-measure estimate
   from Gatheral-Jaisson-Rosenbaum. Fit

       log E|log sigma(t+lag) - log sigma(t)|  ~  H * log(lag)

   Rough volatility claims H ~ 0.1 here, far below the H = 0.5 of a diffusion.

2. **Cross-sectional** (`skew_term_structure_fit`) — the risk-neutral estimate
   from the option smile. At one instant, fit

       log|RR25(T)|  ~  beta * log(T),      H = beta + 1/2

   across the maturities quoted at that instant.

They need not agree, and in this project they do not: the fitted cross-sectional
slope is around +0.21 against a prior implying about -0.40. Reporting both, and
the disagreement, is the honest result.

> A caution that belongs with every number these functions return: the
> structure-function estimator is biased downward by measurement noise in
> sigma. Some of the "roughness" in the literature is estimation error, and
> notebook 04 demonstrates that bias on synthetic data with a known H.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import compat


@dataclass
class HurstEstimate:
    """Result of a structure-function fit."""

    hurst: float
    intercept: float
    r_squared: float
    lags: np.ndarray
    structure_function: np.ndarray
    n_obs: int

    def __str__(self) -> str:
        return (f"H = {self.hurst:.4f}  (R2 = {self.r_squared:.3f}, "
                f"{len(self.lags)} lags, {self.n_obs:,} obs)")


def hurst_structure_function(series, min_lag: int = 1, max_lag: int = 32,
                             log_transform: bool = True) -> HurstEstimate:
    """Estimate H from E|X(t+lag) - X(t)| ~ lag^H.

    `series` should be a volatility path. With `log_transform=True` the
    increments are taken of log sigma, which is the quantity the rough
    literature models as fractional Brownian motion.

    Delegates to `roughvol.analytics.roughness.estimate_hurst_exponent` when
    available — it is session-aware, so increments never straddle an overnight
    gap, which otherwise contaminates every lag.
    """
    s = pd.Series(np.asarray(series, dtype=float)).dropna()

    if compat.ensure_roughvol():
        try:
            rn = compat.roughvol_module("roughvol.analytics.roughness")
            est = rn.estimate_hurst_exponent(s, min_lag=min_lag, max_lag=max_lag)
            return HurstEstimate(
                hurst=float(est.hurst), intercept=float(est.intercept),
                r_squared=float(est.r_squared), lags=np.asarray(est.lags),
                structure_function=np.asarray(est.structure_function), n_obs=len(s))
        except Exception:
            pass                                    # fall through to the native fit

    x = np.log(s.to_numpy()) if log_transform else s.to_numpy()
    x = x[np.isfinite(x)]
    lags = np.arange(min_lag, min(max_lag, len(x) // 4) + 1)
    sf = np.array([np.mean(np.abs(x[lag:] - x[:-lag])) for lag in lags])

    ok = sf > 0
    slope, intercept = np.polyfit(np.log(lags[ok]), np.log(sf[ok]), 1)
    pred = intercept + slope * np.log(lags[ok])
    resid = np.log(sf[ok]) - pred
    ss_tot = np.sum((np.log(sf[ok]) - np.log(sf[ok]).mean()) ** 2)
    r2 = 1.0 - np.sum(resid**2) / ss_tot if ss_tot > 0 else np.nan

    return HurstEstimate(hurst=float(slope), intercept=float(intercept),
                         r_squared=float(r2), lags=lags, structure_function=sf,
                         n_obs=len(x))


def skew_term_structure_fit(cross_section: pd.DataFrame, skew_col: str = "rr25",
                            T_col: str = "T", min_points: int = 3) -> dict | None:
    """Fit log|RR25| ~ beta * log(T) on one timestamp's term structure.

    Returns {slope, intercept, r_squared, implied_hurst, n_points} or None when
    the slice has too few usable maturities. `implied_hurst = slope + 0.5`.
    """
    df = cross_section[[T_col, skew_col]].dropna()
    df = df[(df[T_col] > 0) & (df[skew_col].abs() > 1e-8)]
    if len(df) < min_points:
        return None

    x = np.log(df[T_col].to_numpy(dtype=float))
    y = np.log(np.abs(df[skew_col].to_numpy(dtype=float)))
    if np.ptp(x) < 1e-9:                    # all one maturity: slope undefined
        return None

    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan

    return {"slope": float(slope), "intercept": float(intercept), "r_squared": r2,
            "implied_hurst": float(slope) + 0.5, "n_points": len(df)}


def skew_scaling_panel(df: pd.DataFrame, skew_col: str = "rr25",
                       min_points: int = 3, max_timestamps: int | None = None
                       ) -> pd.DataFrame:
    """Run `skew_term_structure_fit` at every timestamp; return one row each.

    This is the Gate 1 computation. `max_timestamps` subsamples for speed —
    notebooks use it to stay inside the 60-second budget, and say so.
    """
    stamps = df["ts"].drop_duplicates()
    if max_timestamps and len(stamps) > max_timestamps:
        stamps = stamps.iloc[:: max(1, len(stamps) // max_timestamps)]
    keep = df[df["ts"].isin(set(stamps))]

    rows = []
    for ts, grp in keep.groupby("ts", observed=True, sort=True):
        fit = skew_term_structure_fit(grp, skew_col=skew_col, min_points=min_points)
        if fit:
            rows.append({"ts": ts, **fit})

    out = pd.DataFrame(rows)
    out.attrs["skew_col"] = skew_col
    return out


def summarise_skew_scaling(fits: pd.DataFrame) -> pd.Series:
    """Headline numbers for a skew-scaling panel — comparable to Gate 1.

    Reports the slope's coefficient of variation, because a power law that fits
    well at every instant but whose exponent wanders is a description, not a
    structural constant.
    """
    slope = fits["slope"].dropna()
    return pd.Series({
        "n_timestamps": len(fits),
        "median_slope": float(slope.median()),
        "mean_r_squared": float(fits["r_squared"].mean()),
        "median_r_squared": float(fits["r_squared"].median()),
        "slope_cv": float(slope.std() / abs(slope.mean())) if slope.mean() else np.nan,
        "median_implied_hurst": float(fits["implied_hurst"].median()),
    })


__all__ = [
    "HurstEstimate", "hurst_structure_function", "skew_term_structure_fit",
    "skew_scaling_panel", "summarise_skew_scaling",
]
