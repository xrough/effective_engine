#!/usr/bin/env python3
"""
prepare_spy_data.py
===================
Extracts rolling ATM straddle chain from the OPRA CBBO-1m Databento zip and
writes a flat CSV consumed by HistoricalChainAdapter.

Output: demo/data/spy_atm_chain.csv
Columns:
    timestamp_utc       — bar timestamp (UTC, ISO-8601)
    underlying_price    — SPY spot recovered via put-call parity
    atm_strike          — strike used for this bar
    expiry_date         — YYYY-MM-DD of the front weekly expiry
    time_to_expiry      — calendar days to expiry / 365 (annualised)
    call_mid            — (bid+ask)/2 for the ATM call
    put_mid             — (bid+ask)/2 for the ATM put
    call_bid            — best bid for the ATM call (from Databento OPRA)
    call_ask            — best ask for the ATM call (from Databento OPRA)
    put_bid             — best bid for the ATM put  (from Databento OPRA)
    put_ask             — best ask for the ATM put  (from Databento OPRA)
    rv5_ann             — rolling 5-bar annualised realised vol from spot log-returns
                          (NaN → 0.0 for first 4 bars; annualised as σ × √(252×390))

Algorithm per day:
  1. Load DBN/zstd via databento, filter bid > 0 for both legs.
  2. Select "front weekly" expiry: nearest expiry with T in [5, 21] calendar days.
  3. For each 1-min bar (group by ts_recv):
       a. Find K* = argmin |K - S_prev| among available strikes at that expiry.
       b. call_mid = (bid + ask) / 2 for K* call.
       c. put_mid  = (bid + ask) / 2 for K* put.
       d. S_new ≈ K* + call_mid - put_mid  (put-call parity, r≈0 for T < 3w).
       e. Skip bar if either leg has mid < 0.05 (illiquid) or NaN.
  4. Roll to next expiry when the current one drops below 5 calendar days.
  5. Append raw bid/ask for each leg so C++ can compute realistic fills.
  6. Compute rolling 5-bar realised vol from recovered spot log-returns.

Usage:
    python3 demo/scripts/prepare_spy_data.py [--days N] [--zip PATH] [--out PATH]

Defaults:
    --days  5
    --zip   demo/data/OPRA-20260208-3T68RYYKF9.zip
    --out   demo/data/spy_atm_chain.csv
"""

import argparse
import io
import re
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
T_MIN_DAYS  = 10   # minimum days to expiry (~2 weeks)
T_MAX_DAYS  = 35   # maximum days to expiry (~5 weeks)
MID_FLOOR   = 0.05 # minimum mid-price to accept a leg as liquid
S_INIT      = 637.0  # warm-start spot estimate (used only for the very first bar)

# ── helpers ──────────────────────────────────────────────────────────────────
_OSI_RE = re.compile(r'^(\w+)\s+(\d{6})([CP])(\d{8})$')

def parse_osi(sym: str):
    """Return (expiry_date, opt_type, strike) or None."""
    m = _OSI_RE.match(sym.strip())
    if not m:
        return None
    _, exp_str, typ, strike_str = m.groups()
    exp = date(2000 + int(exp_str[:2]), int(exp_str[2:4]), int(exp_str[4:6]))
    strike = int(strike_str) / 1000.0
    return exp, typ, strike


def select_front_expiry(bar_date: date, available_expiries: list[date]) -> date | None:
    """
    Pick the nearest expiry with T in [T_MIN_DAYS, T_MAX_DAYS] calendar days.
    Returns None if no suitable expiry exists.
    """
    candidates = [
        e for e in available_expiries
        if T_MIN_DAYS <= (e - bar_date).days <= T_MAX_DAYS
    ]
    return min(candidates) if candidates else None


def load_day(zf: zipfile.ZipFile, day_name: str) -> pd.DataFrame:
    """
    Load one day's DBN/zstd file from the zip into a DataFrame.
    Adds parsed columns: expiry, opt_type, strike.
    Filters to rows with valid bid/ask (bid > 0).
    """
    raw = zf.read(day_name)
    buf = io.BytesIO(raw)

    # databento requires a file path or file-like object
    # write to a temp file-like with a .name attribute
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.dbn.zst', delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        dbn = db.DBNStore.from_file(tmp_path)
        df = dbn.to_df()
    finally:
        os.unlink(tmp_path)

    # filter valid bid/ask
    df = df[(df['bid_px_00'] > 0) & (df['ask_px_00'] > 0)].copy()

    # parse OSI symbols
    parsed = df['symbol'].apply(parse_osi)
    valid = parsed.notna()
    df = df[valid].copy()
    df['expiry']   = parsed[valid].apply(lambda x: x[0])
    df['opt_type'] = parsed[valid].apply(lambda x: x[1])
    df['strike']   = parsed[valid].apply(lambda x: x[2])

    return df


def process_day(df: pd.DataFrame, bar_date: date, s_prev: float) -> tuple[list[dict], float]:
    """
    Process one day's option data.
    Returns (rows, s_last) where rows is a list of dicts (one per valid bar).
    """
    rows = []
    available_expiries = sorted(df['expiry'].unique())

    # group by 1-min bar
    for ts, bar_df in df.groupby(df.index):
        bar_d = ts.date()

        # select front expiry for this bar
        expiry = select_front_expiry(bar_d, available_expiries)
        if expiry is None:
            continue

        # filter to chosen expiry
        exp_df = bar_df[bar_df['expiry'] == expiry]
        calls  = exp_df[exp_df['opt_type'] == 'C'].set_index('strike')
        puts   = exp_df[exp_df['opt_type'] == 'P'].set_index('strike')

        if calls.empty or puts.empty:
            continue

        # find ATM strike nearest to s_prev
        common_strikes = calls.index.intersection(puts.index)
        if common_strikes.empty:
            continue

        k_atm = min(common_strikes, key=lambda k: abs(k - s_prev))

        call_row = calls.loc[k_atm]
        put_row  = puts.loc[k_atm]

        call_bid = call_row['bid_px_00']
        call_ask = call_row['ask_px_00']
        put_bid  = put_row['bid_px_00']
        put_ask  = put_row['ask_px_00']

        call_mid = (call_bid + call_ask) / 2.0
        put_mid  = (put_bid  + put_ask)  / 2.0

        # liquidity guard
        if call_mid < MID_FLOOR or put_mid < MID_FLOOR:
            continue

        # recover spot via put-call parity (r≈0 for T < 3w)
        s_new = k_atm + call_mid - put_mid
        s_prev = s_new

        T = (expiry - bar_d).days / 365.0

        rows.append({
            'timestamp_utc':    ts.isoformat(),
            'underlying_price': round(s_new, 4),
            'atm_strike':       k_atm,
            'expiry_date':      expiry.isoformat(),
            'time_to_expiry':   round(T, 6),
            'call_mid':         round(call_mid, 4),
            'put_mid':          round(put_mid, 4),
            'call_bid':         round(call_bid, 4),
            'call_ask':         round(call_ask, 4),
            'put_bid':          round(put_bid,  4),
            'put_ask':          round(put_ask,  4),
        })

    return rows, s_prev


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Preprocess OPRA SPY options to ATM chain CSV')
    parser.add_argument('--days', type=int, default=5,
                        help='Number of trading days to process (default: 5)')
    parser.add_argument('--zip',  type=str,
                        default='demo/data/OPRA-20260208-3T68RYYKF9.zip',
                        help='Path to OPRA zip file')
    parser.add_argument('--out',  type=str,
                        default='demo/data/spy_atm_chain.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_path = Path(args.out)

    if not zip_path.exists():
        print(f'[ERROR] zip not found: {zip_path}', file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'[prepare_spy_data] 打开 zip: {zip_path}')
    zf = zipfile.ZipFile(zip_path)

    # list all per-day DBN files, sorted chronologically
    day_files = sorted(
        n for n in zf.namelist()
        if re.match(r'opra-pillar-\d{8}\.cbbo-1m\.dbn\.zst', n)
    )
    day_files = day_files[:args.days]
    print(f'[prepare_spy_data] 处理 {len(day_files)} 个交易日: '
          f'{day_files[0]} … {day_files[-1]}')

    all_rows = []
    s_prev = S_INIT

    for i, day_name in enumerate(day_files):
        date_str = re.search(r'(\d{8})', day_name).group(1)
        bar_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        print(f'  [{i+1}/{len(day_files)}] {date_str}', end='', flush=True)

        df = load_day(zf, day_name)
        rows, s_prev = process_day(df, bar_date, s_prev)
        all_rows.extend(rows)
        print(f'  → {len(rows)} bars  S_last={s_prev:.2f}')

    zf.close()

    result = pd.DataFrame(all_rows)

    # ── 滚动已实现波动率（5 bar，年化）────────────────────────────
    log_ret = np.log(result['underlying_price'] / result['underlying_price'].shift(1))
    # annualise: 252 trading days × 390 bars/day ≈ 1-min bar count per year
    result['rv5_ann'] = (log_ret.rolling(5).std() * np.sqrt(252 * 390)).fillna(0.0).round(6)

    result.to_csv(out_path, index=False)
    print(f'\n[prepare_spy_data] 写入 {len(result)} 行 → {out_path}')
    print(f'  行情区间: {result["timestamp_utc"].iloc[0]}  …  {result["timestamp_utc"].iloc[-1]}')
    print(f'  SPY 区间: {result["underlying_price"].min():.2f} – {result["underlying_price"].max():.2f}')
    print(f'  到期日集: {sorted(result["expiry_date"].unique())}')
    call_spread = (result['call_ask'] - result['call_bid']).mean()
    put_spread  = (result['put_ask']  - result['put_bid']).mean()
    rv_mean = result.loc[result['rv5_ann'] > 0, 'rv5_ann'].mean()
    print(f'  平均买卖价差: 认购 ${call_spread:.4f}  认沽 ${put_spread:.4f}')
    print(f'  平均 RV5 (年化): {rv_mean:.4f} ({rv_mean*100:.2f}%)')


if __name__ == '__main__':
    main()
