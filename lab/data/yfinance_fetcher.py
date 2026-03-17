"""
yfinance data fetcher — downloads real market data and formats it for the MVP pipeline.

Produces two CSV files:
  spot_<TICKER>.csv    : timestamp,underlying_price          (feeds MarketDataAdapter)
  options_<TICKER>.csv : strike,maturity_years,is_call,market_price,expiry
                         (feeds calibration layer)
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


class YFinanceFetcher:
    """Download and format market data from Yahoo Finance."""

    # ------------------------------------------------------------------ #
    # Spot history                                                          #
    # ------------------------------------------------------------------ #

    def fetch_spot_history(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download OHLCV history and return a cleaned DataFrame.

        Returns
        -------
        DataFrame with columns: [timestamp (str ISO-8601), underlying_price (float)]
        Ready to be written by save_spot_csv().
        """
        raw = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
        if raw.empty:
            raise ValueError(f"No data returned for {ticker} [{start} – {end}]")

        # Use Close price as the underlying_price
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]   # multi-ticker edge case

        df = pd.DataFrame({
            "timestamp": close.index.strftime("%Y-%m-%dT%H:%M:%S"),
            "underlying_price": close.values.astype(float),
        })
        df = df.dropna(subset=["underlying_price"])
        print(f"[Data] Fetched {len(df)} spot ticks for {ticker}  "
              f"(range ${df['underlying_price'].min():.2f} – ${df['underlying_price'].max():.2f})")
        return df

    def get_latest_spot(self, ticker: str) -> float:
        """Return the most recent closing price."""
        raw = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return float(close.dropna().iloc[-1])

    # ------------------------------------------------------------------ #
    # Options chain                                                         #
    # ------------------------------------------------------------------ #

    def fetch_options_chain(
        self,
        ticker: str,
        spot: float,
        max_expiries: int = 2,
        moneyness_lo: float = 0.80,
        moneyness_hi: float = 1.20,
    ) -> pd.DataFrame:
        """Download near-the-money options for the nearest expiries.

        Parameters
        ----------
        ticker         : equity symbol (e.g. "AAPL")
        spot           : current underlying price (used for moneyness filter)
        max_expiries   : number of expiry dates to include
        moneyness_lo/hi: K/S range to keep (default 0.80–1.20)

        Returns
        -------
        DataFrame with columns:
            strike, maturity_years, is_call, market_price, expiry (str date)
        """
        t = yf.Ticker(ticker)
        all_expiries = t.options
        if not all_expiries:
            raise ValueError(f"No options found for {ticker}")

        today = date.today()
        rows: list[dict] = []

        for expiry_str in all_expiries[:max_expiries]:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            maturity_years = max((expiry_date - today).days / 365.0, 1e-4)

            chain = t.option_chain(expiry_str)

            for is_call, df_opts in [(True, chain.calls), (False, chain.puts)]:
                # Filter: near-the-money; require a usable price
                df_opts = df_opts[
                    (df_opts["strike"] / spot >= moneyness_lo) &
                    (df_opts["strike"] / spot <= moneyness_hi)
                ].copy()
                for _, row in df_opts.iterrows():
                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    last = float(row.get("lastPrice", 0) or 0)
                    if bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                    elif last > 0:
                        mid = last          # fallback: last traded price
                    else:
                        continue            # no usable price — skip
                    rows.append({
                        "strike": float(row["strike"]),
                        "maturity_years": maturity_years,
                        "is_call": is_call,
                        "market_price": mid,
                        "expiry": expiry_str,
                    })

        if not rows:
            raise ValueError(f"No valid options found for {ticker} in moneyness range [{moneyness_lo}, {moneyness_hi}]")

        df = pd.DataFrame(rows)
        n_calls = int(df["is_call"].sum())
        n_puts  = int((~df["is_call"]).sum())
        expiries = df["expiry"].unique().tolist()
        print(f"[Data] Fetched {len(df)} options for {ticker}  "
              f"({n_calls} calls, {n_puts} puts, expiries: {expiries})")
        return df

    # ------------------------------------------------------------------ #
    # CSV writers                                                           #
    # ------------------------------------------------------------------ #

    def save_spot_csv(self, df: pd.DataFrame, path: str | Path) -> None:
        """Write spot DataFrame to CSV in the format expected by MarketDataAdapter.

        Format: timestamp,underlying_price
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df[["timestamp", "underlying_price"]].to_csv(path, index=False)
        print(f"[Data] Spot CSV saved → {path}  ({len(df)} rows)")

    def save_options_csv(self, df: pd.DataFrame, path: str | Path) -> None:
        """Write options DataFrame to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[Data] Options CSV saved → {path}  ({len(df)} rows)")
