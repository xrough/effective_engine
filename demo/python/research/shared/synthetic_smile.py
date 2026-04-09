"""
shared/synthetic_smile.py
=========================
Synthetic rough-style option-chain generator for research sanity checks.

The goal is not to be a full rough-Heston simulator. Instead, this module
creates a latent smile state where:

  rr25 ~= alpha_t * T^(H-0.5) * atm_iv_t
  bf25 ~= gamma_t * T^(2H-1) * atm_total_var_t

and then generates full option-chain slices that must be re-extracted through
the same smile_pipeline logic used on Databento data. This lets Gate 0 / 0B
answer a sharper question:

  "If the world really had a stable rough-style smile manifold, would our
   benchmark recover that edge?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtri

from smile_pipeline import RATE, DIV, process_panel_df


@dataclass(frozen=True)
class SyntheticRoughConfig:
    seed: int = 7
    H: float = 0.10
    n_bars: int = 180
    bar_minutes: int = 30
    start_date: str = "2026-01-05"
    expiries_dte: tuple[int, ...] = (21, 35, 49)

    spot0: float = 580.0
    atm_iv0: float = 0.18
    alpha0: float = -0.17
    gamma0: float = 0.55

    ret_sigma: float = 0.006
    shock_prob: float = 0.12
    shock_sigma: float = 0.03
    active_move_threshold: float = 0.015

    atm_revert: float = 0.12
    atm_absret_beta: float = 0.70
    atm_noise_sigma: float = 0.0015
    atm_long_run: float = 0.18

    alpha_revert: float = 0.008
    alpha_noise_sigma: float = 0.0010
    alpha_long_run: float = -0.17

    gamma_revert: float = 0.008
    gamma_noise_sigma: float = 0.0040
    gamma_long_run: float = 0.55

    quiet_iv_noise: float = 0.0010
    active_iv_noise: float = 0.0120


def _market_timestamps(start_date: str, n_bars: int, bar_minutes: int) -> list[pd.Timestamp]:
    """Generate business-day timestamps in the same UTC session window as OPRA tests."""
    ts_list: list[pd.Timestamp] = []
    day = pd.Timestamp(start_date, tz="UTC")
    market_open = 13 * 60 + 35
    market_close = 19 * 60 + 55

    while len(ts_list) < n_bars:
        if day.weekday() < 5:
            cur = day.normalize() + pd.Timedelta(minutes=market_open)
            close = day.normalize() + pd.Timedelta(minutes=market_close)
            while cur <= close and len(ts_list) < n_bars:
                ts_list.append(cur)
                cur += pd.Timedelta(minutes=bar_minutes)
        day += pd.Timedelta(days=1)

    return ts_list


def _strike_from_delta(F: float, T: float, sigma: float,
                       delta: float, is_call: bool) -> float:
    """
    Invert Black-Scholes delta to a strike using the same RATE/DIV setup as the
    smile pipeline. delta is signed, e.g. +0.25 for calls, -0.25 for puts.
    """
    if is_call:
        target = delta * math.exp(DIV * T)
    else:
        target = 1.0 + delta * math.exp(DIV * T)

    target = min(max(target, 1e-6), 1.0 - 1e-6)
    d1 = float(ndtri(target))
    log_fk = sigma * math.sqrt(T) * d1 - 0.5 * sigma * sigma * T
    return float(F * math.exp(-log_fk))


def _build_synthetic_slice(F: float, T: float, atm_iv: float,
                           rr25: float, bf25: float,
                           iv_noise: float,
                           rng: np.random.Generator) -> pd.DataFrame:
    """
    Build a small option-chain slice with common strikes for calls and puts.

    The smile is anchored on ATM, 25d put, and 25d call vols, then extended to
    10d wings with a mild convexity lift. Mid prices are internally consistent,
    so the smile extractor still has to recover the forward and IVs from prices.
    """
    iv25c = max(atm_iv + bf25 + 0.5 * rr25, 0.05)
    iv25p = max(atm_iv + bf25 - 0.5 * rr25, 0.05)
    wing_lift = max(0.003, 0.45 * abs(bf25) + 0.20 * abs(rr25))

    k10p = _strike_from_delta(F, T, iv25p + wing_lift, -0.10, False)
    k25p = _strike_from_delta(F, T, iv25p,            -0.25, False)
    k50 = F
    k25c = _strike_from_delta(F, T, iv25c,             0.25, True)
    k10c = _strike_from_delta(F, T, iv25c + wing_lift, 0.10, True)

    anchor_logk = np.log(np.array([k10p, k25p, k50, k25c, k10c]) / F)
    anchor_iv = np.array([
        iv25p + wing_lift,
        iv25p,
        atm_iv,
        iv25c,
        iv25c + wing_lift,
    ])

    strike_grid = np.exp(np.linspace(anchor_logk[0], anchor_logk[-1], 9)) * F
    strike_grid = np.unique(np.round(strike_grid, 4))
    smile_ivs = np.interp(np.log(strike_grid / F), anchor_logk, anchor_iv)
    if iv_noise > 0.0:
        smile_ivs = np.maximum(smile_ivs + rng.normal(0.0, iv_noise, size=smile_ivs.shape), 0.05)

    disc = math.exp(-RATE * T)
    rows = []
    for K, sigma in zip(strike_grid, smile_ivs):
        d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        Nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        call_mid = disc * (F * Nd1 - K * Nd2)
        put_mid = disc * (K * (1.0 - Nd2) - F * (1.0 - Nd1))
        spread = 0.02 + 0.02 * abs(math.log(K / F))

        rows.append({
            "strike": float(K),
            "is_call": True,
            "mid": float(call_mid),
            "spread": float(spread),
        })
        rows.append({
            "strike": float(K),
            "is_call": False,
            "mid": float(put_mid),
            "spread": float(spread),
        })

    return pd.DataFrame(rows)


def generate_rough_synthetic_panel(cfg: SyntheticRoughConfig = SyntheticRoughConfig()):
    """
    Generate a synthetic option-chain panel and the latent state used to create it.

    Returns
    -------
    panel_df : DataFrame
      Columns: ts, expiry, strike, is_call, mid, spread
    state_df : DataFrame
      Latent spot / atm_iv / alpha / gamma / abs_ret at each timestamp
    """
    rng = np.random.default_rng(cfg.seed)
    timestamps = _market_timestamps(cfg.start_date, cfg.n_bars, cfg.bar_minutes)
    expiry_dates = [timestamps[0].date() + pd.Timedelta(days=d) for d in cfg.expiries_dte]

    n = len(timestamps)
    spot = np.empty(n)
    atm_iv = np.empty(n)
    alpha = np.empty(n)
    gamma = np.empty(n)
    abs_ret = np.zeros(n)

    spot[0] = cfg.spot0
    atm_iv[0] = cfg.atm_iv0
    alpha[0] = cfg.alpha0
    gamma[0] = cfg.gamma0

    for i in range(1, n):
        ret = rng.normal(0.0, cfg.ret_sigma)
        if rng.random() < cfg.shock_prob:
            ret += rng.normal(0.0, cfg.shock_sigma)

        abs_ret[i] = abs(ret)
        spot[i] = spot[i - 1] * math.exp(ret)

        atm_iv[i] = max(
            0.08,
            (1.0 - cfg.atm_revert) * atm_iv[i - 1]
            + cfg.atm_revert * cfg.atm_long_run
            + cfg.atm_absret_beta * abs_ret[i]
            + rng.normal(0.0, cfg.atm_noise_sigma),
        )
        alpha[i] = (
            (1.0 - cfg.alpha_revert) * alpha[i - 1]
            + cfg.alpha_revert * cfg.alpha_long_run
            + rng.normal(0.0, cfg.alpha_noise_sigma)
        )
        gamma[i] = (
            (1.0 - cfg.gamma_revert) * gamma[i - 1]
            + cfg.gamma_revert * cfg.gamma_long_run
            + rng.normal(0.0, cfg.gamma_noise_sigma)
        )

    state_df = pd.DataFrame({
        "ts": timestamps,
        "spot": spot,
        "atm_iv": atm_iv,
        "alpha": alpha,
        "gamma": gamma,
        "abs_ret": abs_ret,
        "is_active": abs_ret > cfg.active_move_threshold,
    })

    rows = []
    for i, ts in enumerate(timestamps):
        for exp in expiry_dates:
            dte_days = (exp - ts.date()).days
            if dte_days < 7 or dte_days > 60:
                continue

            T = dte_days / 365.0
            rr25 = alpha[i] * (T ** (cfg.H - 0.5)) * atm_iv[i]
            bf25 = gamma[i] * (T ** (2.0 * cfg.H - 1.0)) * (atm_iv[i] * atm_iv[i] * T)
            iv_noise = cfg.active_iv_noise if abs_ret[i] > cfg.active_move_threshold else cfg.quiet_iv_noise
            forward = spot[i] * math.exp((RATE - DIV) * T)

            slice_df = _build_synthetic_slice(
                F=forward,
                T=T,
                atm_iv=atm_iv[i],
                rr25=rr25,
                bf25=bf25,
                iv_noise=iv_noise,
                rng=rng,
            )
            slice_df["ts"] = ts
            slice_df["expiry"] = exp
            rows.extend(slice_df.to_dict("records"))

    panel_df = pd.DataFrame(rows)
    return panel_df, state_df


def generate_rough_synthetic_records(cfg: SyntheticRoughConfig = SyntheticRoughConfig()):
    """
    Convenience wrapper: generate a synthetic panel and immediately run the same
    feature extraction logic used by the real-data gates.
    """
    panel_df, state_df = generate_rough_synthetic_panel(cfg)
    records = process_panel_df(panel_df, cfg.H)
    return panel_df, state_df, records

