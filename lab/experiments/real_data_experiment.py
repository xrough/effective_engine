#!/usr/bin/env python3
"""
Real Market Data Experiment
============================
Fetches live data from Yahoo Finance, calibrates and compares pricing models
against real option prices, and produces diagnostic visualizations.

  Phase 0  Data download      yfinance → spot CSV + options CSV
  Phase 1  Calibration        BS (closed-form) | GBM-MC | Heston-MC  (via gRPC)
  Phase 2  Visualization      4 matplotlib figures saved to data/plots/

The live event-driven simulation (GBM/BS pipeline) runs on the C++ platform:
  ./build/market_maker

Usage (from MVP/lab/):
  MODEL_SERVICE_ADDR=localhost:50051 python3 experiments/real_data_experiment.py --ticker AAPL
  python3 experiments/real_data_experiment.py --ticker SPY --start 2024-01-01 --end 2024-09-30
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── sys.path setup ────────────────────────────────────────────────────────────
_LAB_DIR   = Path(__file__).resolve().parent.parent   # MVP/lab/
_DATA_DIR  = _LAB_DIR.parent / "data"                 # MVP/data/
_PLOTS_DIR = _DATA_DIR / "plots"

if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.yfinance_fetcher import YFinanceFetcher
from grpc_client.rough_pricing_client import RoughPricingClient, CalibResult

_CLIENT = RoughPricingClient(os.environ.get("MODEL_SERVICE_ADDR", "localhost:50051"))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-data model calibration experiment")
    p.add_argument("--ticker",       default="AAPL",       help="Yahoo Finance ticker (default: AAPL)")
    p.add_argument("--start",        default="2024-01-01", help="Spot history start date (YYYY-MM-DD)")
    p.add_argument("--end",          default="2024-06-30", help="Spot history end date (YYYY-MM-DD)")
    p.add_argument("--rate",         type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    p.add_argument("--div",          type=float, default=0.0,  help="Continuous dividend yield (default: 0)")
    p.add_argument("--max-expiries", type=int,   default=2,    help="Number of option expiries to include")
    p.add_argument("--skip-heston",  action="store_true",      help="Skip slow Heston calibration")
    return p.parse_args()


# ── Phase 0: Data download ────────────────────────────────────────────────────

def phase0_fetch(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Download spot + options, save CSVs, return (spot_df, options_df, latest_spot)."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Phase 0: Fetching Market Data                           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    fetcher = YFinanceFetcher()

    spot_df = fetcher.fetch_spot_history(args.ticker, args.start, args.end)
    spot_csv = _DATA_DIR / f"spot_{args.ticker}.csv"
    fetcher.save_spot_csv(spot_df, spot_csv)

    latest_spot = float(spot_df["underlying_price"].iloc[-1])
    print(f"[Data] Latest spot: ${latest_spot:.2f}")

    options_df = fetcher.fetch_options_chain(
        args.ticker,
        spot=latest_spot,
        max_expiries=args.max_expiries,
    )
    options_csv = _DATA_DIR / f"options_{args.ticker}.csv"
    fetcher.save_options_csv(options_df, options_csv)

    return spot_df, options_df, latest_spot


# ── Phase 1: Multi-model calibration ─────────────────────────────────────────

def phase1_calibration(
    spot: float,
    options_df: pd.DataFrame,
    args: argparse.Namespace,
) -> list[CalibResult]:
    """Calibrate BS, GBM-MC, and (optionally) Heston against real options data."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Phase 1: Multi-Model Calibration                        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    print(f"  Spot:    ${spot:.4f}")
    print(f"  Rate:     {args.rate:.4f}")
    print(f"  Div:      {args.div:.4f}")
    print(f"  Options: {len(options_df)} contracts across {options_df['expiry'].nunique()} expiry(s)\n")

    results: list[CalibResult] = []

    # 1. BS (closed-form, fast)
    print("── BS Calibration (closed-form) ──")
    result_bs = _CLIENT.calibrate_bs(spot, options_df, args.rate, args.div)
    results.append(result_bs)
    sigma_bs = result_bs.params["sigma_atm"]
    print(f"   σ_ATM = {sigma_bs:.4f}  MSE = {result_bs.mse:.3e}  elapsed = {result_bs.elapsed_s:.3f}s\n")

    # 2. GBM-MC (warm-started from BS)
    print("── GBM Monte Carlo Calibration (1D, warm-started from BS) ──")
    result_gbm = _CLIENT.calibrate_gbm(spot, options_df, args.rate, args.div, x0_sigma=sigma_bs)
    results.append(result_gbm)
    print()

    # 3. Heston-MC (optional)
    if not args.skip_heston:
        print("── Heston Monte Carlo Calibration (5D, warm-started from BS) ──")
        print("   (This may take 1–5 minutes depending on option count)")
        result_heston = _CLIENT.calibrate_heston(spot, options_df, args.rate, args.div, x0_sigma=sigma_bs)
        results.append(result_heston)
        print()

    return results


# ── Phase 2: Visualisation ────────────────────────────────────────────────────

def phase2_visualize(
    ticker: str,
    spot_df: pd.DataFrame,
    options_df: pd.DataFrame,
    spot: float,
    results: list[CalibResult],
    rate: float,
    div: float,
) -> None:
    """Produce and save four diagnostic plots."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Phase 2: Visualization                                  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Plot 1: Spot price history ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(spot_df["timestamp"], spot_df["underlying_price"],
            linewidth=1.2, color="#1f77b4")
    n_ticks = min(8, len(spot_df))
    step = max(1, len(spot_df) // n_ticks)
    ax.set_xticks(spot_df["timestamp"].values[::step])
    ax.set_xticklabels(spot_df["timestamp"].values[::step], rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.set_title(f"{ticker} — Spot Price History")
    fig.tight_layout()
    p1 = _PLOTS_DIR / f"spot_{ticker}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[Plot 1] Spot history → {p1}")

    # ── Plot 2: Implied vol smile ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors

    for i, (expiry, grp) in enumerate(options_df.groupby("expiry")):
        ivols = []
        moneyness = []
        for _, row in grp.iterrows():
            try:
                iv = _CLIENT.implied_vol(
                    price=float(row["market_price"]),
                    spot=spot,
                    strike=float(row["strike"]),
                    maturity=float(row["maturity_years"]),
                    rate=rate,
                    div=div,
                    is_call=bool(row["is_call"]),
                )
                ivols.append(iv)
                moneyness.append(float(row["strike"]) / spot)
            except Exception:
                pass
        if ivols:
            pairs = sorted(zip(moneyness, ivols))
            m_sorted, iv_sorted = zip(*pairs)
            ax.plot(m_sorted, iv_sorted, marker="o", markersize=5,
                    color=colors[i % len(colors)], label=f"Expiry {expiry}")

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, label="ATM (K/S=1)")
    ax.set_xlabel("Moneyness (K / S)")
    ax.set_ylabel("BS Implied Volatility")
    ax.set_title(f"{ticker} — Implied Volatility Smile")
    ax.legend()
    fig.tight_layout()
    p2 = _PLOTS_DIR / f"vol_smile_{ticker}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[Plot 2] Vol smile → {p2}")

    # ── Plot 3: Calibration MSE comparison ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [r.model_name for r in results]
    mses  = [r.mse        for r in results]
    colors_bar = ["#2196F3", "#FF9800", "#E91E63"][:len(names)]
    bars = ax.barh(names, mses, color=colors_bar, edgecolor="white")
    ax.bar_label(bars, fmt="%.2e", padding=4, fontsize=9)
    ax.set_xlabel("MSE (model price vs market price)")
    ax.set_title(f"{ticker} — Calibration MSE by Model")
    ax.invert_yaxis()
    fig.tight_layout()
    p3 = _PLOTS_DIR / f"calib_mse_{ticker}.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[Plot 3] MSE comparison → {p3}")

    # ── Plot 4: Model price fit ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    market_prices = options_df["market_price"].values.astype(float)
    scatter_colors = ["#2196F3", "#FF9800", "#E91E63"]

    for idx, result in enumerate(results):
        model_prices = _reprice(result, spot, options_df, rate, div)
        ax.scatter(market_prices, model_prices,
                   label=result.model_name, alpha=0.65, s=30,
                   color=scatter_colors[idx % len(scatter_colors)])

    lo = min(market_prices.min(), 0)
    hi = market_prices.max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Market Price ($)")
    ax.set_ylabel("Model Price ($)")
    ax.set_title(f"{ticker} — Model vs Market Option Prices")
    ax.legend()
    fig.tight_layout()
    p4 = _PLOTS_DIR / f"price_fit_{ticker}.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"[Plot 4] Price fit → {p4}")


# ── Results table ─────────────────────────────────────────────────────────────

def _print_table(ticker: str, spot: float, rate: float, options_df: pd.DataFrame,
                 results: list[CalibResult]) -> None:
    print("\n" + "═" * 72)
    print("  Calibration Results")
    print("═" * 72)
    print(f"  Ticker  : {ticker}")
    print(f"  Spot    : ${spot:.4f}")
    print(f"  Rate    : {rate:.4f}")
    print(f"  Options : {len(options_df)} contracts  ({options_df['expiry'].nunique()} expiry(s))\n")

    header = f"  {'Model':<12} │ {'Parameters':<40} │ {'MSE':>10}  │ {'Time (s)':>8}"
    sep    = "  " + "─" * 12 + "─┼─" + "─" * 40 + "─┼─" + "─" * 10 + "──┼─" + "─" * 8
    print(header)
    print(sep)

    for r in results:
        param_str = "  ".join(f"{k}={v:.4f}" for k, v in r.params.items())
        print(f"  {r.model_name:<12} │ {param_str:<40} │ {r.mse:>10.3e}  │ {r.elapsed_s:>8.2f}")

    print("═" * 72)
    if len(results) >= 2:
        mses = [r.mse for r in results]
        best = results[int(np.argmin(mses))]
        print(f"\n  Best fit: {best.model_name}  (MSE = {best.mse:.3e})")
    print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reprice(
    result: CalibResult,
    spot: float,
    options_df: pd.DataFrame,
    rate: float,
    div: float,
) -> np.ndarray:
    """Reprice all options using the calibrated model parameters."""
    if result.model_name == "BS":
        sigma = result.params["sigma_atm"]
        return np.array([
            _CLIENT.bs_price(
                spot=spot,
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                rate=rate,
                div=div,
                vol=sigma,
                is_call=bool(row["is_call"]),
            )
            for _, row in options_df.iterrows()
        ])

    if result.model_name == "GBM-MC":
        sigma = result.params["sigma"]
        return np.array([
            _CLIENT.mc_price_vanilla_gbm(
                sigma=sigma,
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                is_call=bool(row["is_call"]),
                spot=spot,
                rate=rate,
                div_yield=div,
                n_paths=20_000,
                n_steps=50,
                seed=42,
                antithetic=True,
            ).price
            for _, row in options_df.iterrows()
        ])

    if result.model_name == "Heston":
        p = result.params
        return np.array([
            _CLIENT.mc_price_vanilla_heston(
                kappa=p["kappa"],
                theta=p["theta"],
                xi=p["xi"],
                rho=p["rho"],
                v0=p["v0"],
                strike=float(row["strike"]),
                maturity=float(row["maturity_years"]),
                is_call=bool(row["is_call"]),
                spot=spot,
                rate=rate,
                div_yield=div,
                n_paths=20_000,
                n_steps=50,
                seed=42,
                antithetic=True,
            ).price
            for _, row in options_df.iterrows()
        ])

    return np.zeros(len(options_df))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Options Market Maker — Real Data Experiment             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Ticker: {args.ticker}   Period: {args.start} → {args.end}")
    print(f"  Rate: {args.rate:.4f}   Div: {args.div:.4f}   Expiries: {args.max_expiries}")
    print(f"  Model service: {os.environ.get('MODEL_SERVICE_ADDR', 'localhost:50051')}\n")

    spot_df, options_df, latest_spot = phase0_fetch(args)
    results = phase1_calibration(latest_spot, options_df, args)
    _print_table(args.ticker, latest_spot, args.rate, options_df, results)
    phase2_visualize(args.ticker, spot_df, options_df, latest_spot, results, args.rate, args.div)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Experiment complete!                                    ║")
    print(f"║  Plots saved to:  {str(_PLOTS_DIR):<38}║")
    print("╚══════════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
