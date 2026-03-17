"""
RoughPricingClient — thin gRPC wrapper for RoughPricingService.

Replaces the local roughvol/ and calibration/ imports that previously lived
inside Effective_Engine.  Call sites change minimally:

  Before:
    from calibration.model_calibrator import BSCalibrator, make_gbm_calibrator, ...
    from roughvol.analytics.black_scholes_formula import bs_price, implied_vol

  After:
    from grpc_client.rough_pricing_client import RoughPricingClient, CalibResult
    client = RoughPricingClient()
    price = client.bs_price(spot=..., ...)
    result = client.calibrate_bs(spot, options_df, rate, div)

Usage:
    # As a context manager (preferred):
    with RoughPricingClient() as client:
        price = client.bs_price(spot=100, strike=100, maturity=1.0,
                                 rate=0.05, div=0.0, vol=0.2, is_call=True)

    # Long-lived instance:
    client = RoughPricingClient(address="localhost:50051")
    ...
    client.close()

The server address defaults to localhost:50051 and can be overridden via
the MODEL_SERVICE_ADDR environment variable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import grpc
import pandas as pd

# Make generated stubs importable from MVP/generated/python
_GEN_DIR = Path(__file__).resolve().parents[2] / "generated" / "python"
if str(_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN_DIR))

import rough_pricing_pb2 as pb2
import rough_pricing_pb2_grpc as pb2_grpc


# ── Result container (mirrors server-side CalibResult) ─────────────────────

@dataclass
class CalibResult:
    """Calibration result returned to callers."""
    model_name: str
    params: dict
    mse: float
    per_option_ivols: list = field(default_factory=list)
    elapsed_s: float = 0.0

    def __str__(self) -> str:
        param_str = "  ".join(f"{k}={v:.4f}" for k, v in self.params.items())
        return (f"{self.model_name:<12} │ {param_str:<40} │ "
                f"MSE={self.mse:.3e}  │ {self.elapsed_s:.2f}s")


# ── MC price result ────────────────────────────────────────────────────────

@dataclass
class MCPriceResult:
    price: float
    stderr: float
    ci95_lo: float
    ci95_hi: float
    n_paths: int
    n_steps: int


# ── Client ─────────────────────────────────────────────────────────────────

class RoughPricingClient:
    """Synchronous gRPC client for RoughPricingService."""

    def __init__(self, address: Optional[str] = None) -> None:
        addr = address or os.environ.get("MODEL_SERVICE_ADDR", "localhost:50051")
        self._channel = grpc.insecure_channel(addr)
        self._stub = pb2_grpc.RoughPricingServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "RoughPricingClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Black-Scholes price ────────────────────────────────────────────────

    def bs_price(
        self,
        *,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        vol: float,
        is_call: bool,
    ) -> float:
        req = pb2.BSPriceRequest(
            spot=spot, strike=strike, maturity=maturity,
            rate=rate, div=div, vol=vol, is_call=is_call,
        )
        return self._stub.BSPrice(req).price

    # ── Implied volatility ─────────────────────────────────────────────────

    def implied_vol(
        self,
        *,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        div: float,
        is_call: bool,
    ) -> float:
        req = pb2.ImpliedVolRequest(
            price=price, spot=spot, strike=strike, maturity=maturity,
            rate=rate, div=div, is_call=is_call,
        )
        return self._stub.ImpliedVol(req).vol

    # ── MC price (GBM vanilla) ─────────────────────────────────────────────

    def mc_price_vanilla_gbm(
        self,
        *,
        sigma: float,
        strike: float,
        maturity: float,
        is_call: bool,
        spot: float,
        rate: float,
        div_yield: float,
        n_paths: int = 20_000,
        n_steps: int = 50,
        seed: int = 42,
        antithetic: bool = True,
    ) -> MCPriceResult:
        req = pb2.MCPriceRequest(
            gbm=pb2.GBMParams(sigma=sigma),
            vanilla=pb2.VanillaInstrument(
                strike=strike, maturity=maturity, is_call=is_call
            ),
            market=pb2.MarketData(spot=spot, rate=rate, div_yield=div_yield),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            antithetic=antithetic,
        )
        resp = self._stub.MCPrice(req)
        return MCPriceResult(
            price=resp.price,
            stderr=resp.stderr,
            ci95_lo=resp.ci95_lo,
            ci95_hi=resp.ci95_hi,
            n_paths=resp.n_paths,
            n_steps=resp.n_steps,
        )

    # ── MC price (Heston vanilla) ──────────────────────────────────────────

    def mc_price_vanilla_heston(
        self,
        *,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
        strike: float,
        maturity: float,
        is_call: bool,
        spot: float,
        rate: float,
        div_yield: float,
        n_paths: int = 20_000,
        n_steps: int = 50,
        seed: int = 42,
        antithetic: bool = True,
    ) -> MCPriceResult:
        req = pb2.MCPriceRequest(
            heston=pb2.HestonParams(
                kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0
            ),
            vanilla=pb2.VanillaInstrument(
                strike=strike, maturity=maturity, is_call=is_call
            ),
            market=pb2.MarketData(spot=spot, rate=rate, div_yield=div_yield),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            antithetic=antithetic,
        )
        resp = self._stub.MCPrice(req)
        return MCPriceResult(
            price=resp.price,
            stderr=resp.stderr,
            ci95_lo=resp.ci95_lo,
            ci95_hi=resp.ci95_hi,
            n_paths=resp.n_paths,
            n_steps=resp.n_steps,
        )

    # ── Calibration helpers ────────────────────────────────────────────────

    def _calibrate(
        self,
        model_type_name: str,
        spot: float,
        options_df: pd.DataFrame,
        rate: float,
        div: float,
        x0: Optional[list] = None,
        n_paths: int = 20_000,
        n_steps: int = 50,
        seed: int = 42,
        antithetic: bool = True,
    ) -> CalibResult:
        quotes = [
            pb2.OptionQuote(
                strike=float(row["strike"]),
                maturity_years=float(row["maturity_years"]),
                is_call=bool(row["is_call"]),
                market_price=float(row["market_price"]),
            )
            for _, row in options_df.iterrows()
        ]
        req = pb2.CalibrateRequest(
            model_type=pb2.ModelType.Value(model_type_name),
            spot=spot,
            rate=rate,
            div=div,
            quotes=quotes,
            x0=x0 or [],
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            antithetic=antithetic,
        )
        resp = self._stub.Calibrate(req)
        return CalibResult(
            model_name=resp.model_name,
            params=dict(resp.params),
            mse=resp.mse,
            per_option_ivols=list(resp.per_option_ivols),
            elapsed_s=resp.elapsed_s,
        )

    def calibrate_bs(
        self,
        spot: float,
        options_df: pd.DataFrame,
        rate: float = 0.05,
        div: float = 0.0,
    ) -> CalibResult:
        return self._calibrate("BS", spot, options_df, rate, div)

    def calibrate_gbm(
        self,
        spot: float,
        options_df: pd.DataFrame,
        rate: float = 0.05,
        div: float = 0.0,
        x0_sigma: float = 0.20,
        n_paths: int = 20_000,
        n_steps: int = 50,
        seed: int = 42,
        antithetic: bool = True,
    ) -> CalibResult:
        return self._calibrate(
            "GBM_MC", spot, options_df, rate, div,
            x0=[x0_sigma],
            n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=antithetic,
        )

    def calibrate_heston(
        self,
        spot: float,
        options_df: pd.DataFrame,
        rate: float = 0.05,
        div: float = 0.0,
        x0_sigma: float = 0.20,
        n_paths: int = 20_000,
        n_steps: int = 50,
        seed: int = 42,
        antithetic: bool = True,
    ) -> CalibResult:
        return self._calibrate(
            "HESTON", spot, options_df, rate, div,
            x0=[x0_sigma],
            n_paths=n_paths, n_steps=n_steps, seed=seed, antithetic=antithetic,
        )
