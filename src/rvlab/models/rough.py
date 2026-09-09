"""
rvlab.models.rough
==================
The rough-volatility model, in the two forms this project actually uses.

1. **The scaling laws** (`structural_alpha`, `structural_gamma`) — the empirical
   claim. Under a rough model with Hurst H, the 25-delta risk reversal and
   butterfly scale with maturity as

       RR25(T) = alpha * T^(H - 0.5) * atm_iv
       BF25(T) = gamma * T^(2H - 1) * atm_total_var

   so alpha and gamma should be *maturity-invariant* if H is right. That is the
   testable content, and inverting the relation is how notebook 04 estimates H.

2. **The closed-form smile** (`bergomi_guyon_smile`) — the second-order
   Bergomi-Guyon / Fukasawa expansion. A direct port of
   `src/core/analytics/RoughVolPricingEngine.cpp:44-72`, so the notebooks and the
   C++ engine cannot drift apart:

       sigma_ATM = sqrt(xi0)
       psi(T)    = rho * eta * Gamma(H + 0.5) / (2*sqrt(pi)) * T^(H - 0.5)
       chi(T)    = psi(T)^2 / sigma_ATM
       sigma(k)  = sigma_ATM + psi*k + (chi/2)*k^2,     k = log(K/S)

Note the sign structure: with rho < 0, psi < 0, so the smile slopes down — and
because H < 1/2 the exponent T^(H-0.5) is negative, meaning the slope *blows up*
as T -> 0. That explosion is the whole empirical signature of roughness, and
notebook 04 checks whether SPY actually shows it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import blackscholes as bs


@dataclass(frozen=True)
class RoughVolParams:
    """Matches the C++ `omm::domain::RoughVolParams` field for field."""

    H: float = 0.10        # Hurst exponent; H < 0.5 is "rough"
    eta: float = 1.50      # vol-of-vol
    rho: float = -0.70     # spot-vol correlation
    xi0: float = 0.0625    # flat forward variance at t=0 (= 0.25^2)

    @property
    def sigma_atm(self) -> float:
        return math.sqrt(self.xi0)


# ── 1. the scaling laws ────────────────────────────────────────────────────────
def skew_exponent(hurst: float) -> float:
    """The maturity exponent of the skew: H - 1/2. Negative whenever H < 1/2."""
    return hurst - 0.5


def curvature_exponent(hurst: float) -> float:
    """The maturity exponent of the butterfly: 2H - 1."""
    return 2.0 * hurst - 1.0


def structural_alpha(rr25, T, atm_iv, hurst: float):
    """Invert RR25 = alpha * T^(H-0.5) * atm_iv for alpha.

    A maturity-invariant alpha is the signature of a correct H. Identical to
    `_append_record` in demo/python/research/shared/smile_pipeline.py.
    """
    T = np.asarray(T, dtype=float)
    denom = T ** skew_exponent(hurst) * np.asarray(atm_iv, dtype=float)
    return np.where(np.abs(denom) > 1e-12, np.asarray(rr25, dtype=float) / denom, np.nan)


def structural_gamma(bf25, T, atm_total_var, hurst: float):
    """Invert BF25 = gamma * T^(2H-1) * atm_total_var for gamma."""
    T = np.asarray(T, dtype=float)
    denom = T ** curvature_exponent(hurst) * np.asarray(atm_total_var, dtype=float)
    return np.where(np.abs(denom) > 1e-12, np.asarray(bf25, dtype=float) / denom, np.nan)


def implied_hurst_from_slope(slope: float) -> float:
    """H implied by an observed log|RR25| vs log(T) regression slope.

    Because slope = H - 1/2, this is just slope + 1/2. Kept as a named function
    because notebook 04 needs to compare an *empirical* slope against the prior
    and the arithmetic is exactly where sign errors happen.
    """
    return slope + 0.5


# ── 2. the closed-form smile ───────────────────────────────────────────────────
def psi(T, params: RoughVolParams = RoughVolParams()):
    """The leading skew coefficient psi(T). Port of the C++ expansion."""
    T = np.asarray(T, dtype=float)
    return (params.rho * params.eta * math.gamma(params.H + 0.5)
            / (2.0 * math.sqrt(math.pi)) * T ** (params.H - 0.5))


def chi(T, params: RoughVolParams = RoughVolParams()):
    """The leading curvature coefficient chi(T) = psi(T)^2 / sigma_ATM."""
    return psi(T, params) ** 2 / params.sigma_atm


def bergomi_guyon_smile(K, S, T, params: RoughVolParams = RoughVolParams()):
    """Second-order implied vol at strike K. Broadcasts over K and T.

    Floored at 0.001 exactly as the C++ engine does — the quadratic expansion
    goes negative in the far wings and that floor is what keeps it usable.
    """
    k = np.log(np.asarray(K, dtype=float) / np.asarray(S, dtype=float))
    sig = params.sigma_atm + psi(T, params) * k + 0.5 * chi(T, params) * k**2
    return np.maximum(sig, 0.001)


def rough_delta(K, S, T, params: RoughVolParams = RoughVolParams(),
                is_call: bool = True, rate: float | None = None):
    """Minimum-variance delta: BS delta at the smile vol, plus the skew term.

        Delta_rough = Delta_BS(sigma_K) + Vega(sigma_K) * dsigma_K/dS
        dsigma_K/dS = -(psi + chi*k) / S

    The second term is the whole point. A delta-hedger that ignores how implied
    vol moves with spot is hedging the wrong sensitivity whenever the smile has
    slope — which, under a rough model, is always and increasingly so at short
    maturity.
    """
    from ..config import RATE
    rate = RATE if rate is None else rate

    S_a = np.asarray(S, dtype=float)
    k = np.log(np.asarray(K, dtype=float) / S_a)
    sigma_k = bergomi_guyon_smile(K, S, T, params)

    bs_d = bs.delta(sigma_k, S_a, K, T, is_call, rate)
    bs_v = bs.vega(sigma_k, S_a, K, T, rate)
    dsigma_dS = -(psi(T, params) + chi(T, params) * k) / S_a
    return bs_d + bs_v * dsigma_dS


def smile_grid(params: RoughVolParams = RoughVolParams(),
               moneyness=None, maturities=None, spot: float = 1.0):
    """Tidy (T, k, implied_vol) frame — the input to `charts.term_structure_plot`."""
    import pandas as pd

    moneyness = np.linspace(-0.15, 0.15, 61) if moneyness is None else np.asarray(moneyness)
    maturities = np.array([7, 14, 30, 60, 120]) / 365.0 if maturities is None \
        else np.asarray(maturities)

    kk, TT = np.meshgrid(moneyness, maturities)
    iv = bergomi_guyon_smile(spot * np.exp(kk), spot, TT, params)
    return pd.DataFrame({"T": TT.ravel(), "k": kk.ravel(), "implied_vol": iv.ravel()})


__all__ = [
    "RoughVolParams", "skew_exponent", "curvature_exponent", "structural_alpha",
    "structural_gamma", "implied_hurst_from_slope", "psi", "chi",
    "bergomi_guyon_smile", "rough_delta", "smile_grid",
]
