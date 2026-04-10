#!/usr/bin/env python3
"""
prepare_spy_data.py
===================
Extracts rolling ATM straddle chain from the OPRA CBBO-1m Databento zip and
writes a flat CSV consumed by HistoricalChainAdapter.

Output: demo/data/spy_chain_panel.csv   (default, multi-day panel)
        demo/data/spy_atm_chain.csv     (legacy single-day output, unchanged)

Columns — base ATM straddle (same as before, cols 0-11):
    timestamp_utc       — bar timestamp (UTC, ISO-8601)
    underlying_price    — SPY spot recovered via put-call parity
    atm_strike          — strike used for this bar
    expiry_date         — YYYY-MM-DD of the front weekly expiry
    time_to_expiry      — calendar days to expiry / 365 (annualised)
    call_mid            — (bid+ask)/2 for the ATM call
    put_mid             — (bid+ask)/2 for the ATM put
    call_bid            — best bid for the ATM call
    call_ask            — best ask for the ATM call
    put_bid             — best bid for the ATM put
    put_ask             — best ask for the ATM put
    rv5_ann             — rolling 5-bar annualised realised vol (session-reset)

New columns (cols 12-21):
    atm_iv              — BS implied vol at ATM strike (bisection on call_mid)
    call25d_strike      — call strike nearest to 25-delta
    call25d_mid         — market mid for that call
    call25d_bid
    call25d_ask
    put25d_strike       — put strike nearest to |25-delta|
    put25d_mid          — market mid for that put
    put25d_bid
    put25d_ask
    rr25_iv             — IV(call25d) - IV(put25d)    (skew proxy)
    bf25_iv             — (IV(call25d)+IV(put25d))/2 - atm_iv  (curvature proxy)

Algorithm per day:
  1. Load DBN/zstd via databento, filter bid > 0 for both legs.
  2. Select "front weekly" expiry: nearest expiry with T in [T_MIN_DAYS, T_MAX_DAYS].
  3. For each 1-min bar:
       a. Find K* = argmin |K - S_prev| among available strikes.
       b. Recover S via put-call parity.
       c. Compute atm_iv via BS bisection on call_mid.
       d. For each common strike compute BS delta using atm_iv.
       e. Find call25d / put25d as nearest strikes to ±25Δ.
       f. Compute rr25_iv, bf25_iv using bisection on those legs.
  4. Roll expiry when current drops below T_MIN_DAYS.
  5. rv5_ann reset to NaN at each new day (no cross-session leakage).

Usage:
    python3 demo/python/data_prep/prepare_spy_data.py [--days N] [--zip PATH] [--out PATH]

Defaults:
    --days  127
    --zip   demo/data/OPRA-20260208-3T68RYYKF9.zip
    --out   demo/data/spy_chain_panel.csv
"""

import argparse
import io
import math
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
T_MIN_DAYS  = 10
T_MAX_DAYS  = 35
MID_FLOOR   = 0.05   # minimum mid-price for liquid leg
S_INIT      = 637.0  # warm-start spot (first bar only)

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


def select_front_expiry(bar_date: date, available_expiries: list) -> date | None:
    candidates = [
        e for e in available_expiries
        if T_MIN_DAYS <= (e - bar_date).days <= T_MAX_DAYS
    ]
    return min(candidates) if candidates else None


# ── Black-Scholes helpers ─────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return F * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_put_price(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - F, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return K * _norm_cdf(-d2) - F * _norm_cdf(-d1)


def implied_vol(price: float, F: float, K: float, T: float, is_call: bool,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """
    Standard bisection IV. Returns NaN if price is outside no-arb bounds.
    Uses forward price F (put-call parity recovery, r≈0 for T < 3w).
    """
    if T <= 1e-6:
        return float('nan')
    pricer = bs_call_price if is_call else bs_put_price
    lo_price = pricer(F, K, T, lo)
    hi_price = pricer(F, K, T, hi)
    if price <= lo_price or price >= hi_price:
        return float('nan')
    for _ in range(60):
        mid_sigma = 0.5 * (lo + hi)
        mid_price = pricer(F, K, T, mid_sigma)
        if mid_price < price:
            lo = mid_sigma
        else:
            hi = mid_sigma
    return 0.5 * (lo + hi)


def bs_call_delta(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if F >= K else 0.0
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


# ── per-day loading ──────────────────────────────────────────────────────────
def load_day(zf: zipfile.ZipFile, day_name: str) -> pd.DataFrame:
    raw = zf.read(day_name)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix='.dbn.zst', delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        dbn = db.DBNStore.from_file(tmp_path)
        df = dbn.to_df()
    finally:
        os.unlink(tmp_path)
    df = df[(df['bid_px_00'] > 0) & (df['ask_px_00'] > 0)].copy()
    parsed = df['symbol'].apply(parse_osi)
    valid  = parsed.notna()
    df = df[valid].copy()
    df['expiry']   = parsed[valid].apply(lambda x: x[0])
    df['opt_type'] = parsed[valid].apply(lambda x: x[1])
    df['strike']   = parsed[valid].apply(lambda x: x[2])
    return df


def process_day(df: pd.DataFrame, bar_date: date, s_prev: float) -> tuple[list, float]:
    rows = []
    available_expiries = sorted(df['expiry'].unique())
    prev_log_s = math.log(s_prev) if s_prev > 0 else math.log(S_INIT)

    for ts, bar_df in df.groupby(df.index):
        bar_d = ts.date()
        expiry = select_front_expiry(bar_d, available_expiries)
        if expiry is None:
            continue

        exp_df = bar_df[bar_df['expiry'] == expiry]
        calls  = exp_df[exp_df['opt_type'] == 'C'].set_index('strike')
        puts   = exp_df[exp_df['opt_type'] == 'P'].set_index('strike')

        if calls.empty or puts.empty:
            continue

        common_strikes = sorted(calls.index.intersection(puts.index))
        if not common_strikes:
            continue

        # ── ATM strike ──────────────────────────────────────────
        k_atm = min(common_strikes, key=lambda k: abs(k - s_prev))
        call_row = calls.loc[k_atm]
        put_row  = puts.loc[k_atm]

        call_bid = float(call_row['bid_px_00'])
        call_ask = float(call_row['ask_px_00'])
        put_bid  = float(put_row['bid_px_00'])
        put_ask  = float(put_row['ask_px_00'])

        call_mid = (call_bid + call_ask) / 2.0
        put_mid  = (put_bid  + put_ask)  / 2.0

        if call_mid < MID_FLOOR or put_mid < MID_FLOOR:
            continue

        # Forward / spot recovery
        F = k_atm + call_mid - put_mid   # put-call parity (r≈0)
        s_prev = F
        T = (expiry - bar_d).days / 365.0

        # ── ATM implied vol ──────────────────────────────────────
        atm_iv = implied_vol(call_mid, F, k_atm, T, is_call=True)

        # ── 25Δ strike selection ─────────────────────────────────
        call25d_strike = call25d_mid = call25d_bid = call25d_ask = float('nan')
        put25d_strike  = put25d_mid  = put25d_bid  = put25d_ask  = float('nan')
        rr25_iv = bf25_iv = float('nan')

        if not math.isnan(atm_iv) and atm_iv > 0:
            # For calls, delta decreases with K; find K closest to delta=0.25
            call_deltas = {k: bs_call_delta(F, k, T, atm_iv) for k in common_strikes}
            # 25Δ call: call delta = 0.25 (OTM call)
            k_c25 = min(common_strikes, key=lambda k: abs(call_deltas[k] - 0.25))
            # 25Δ put: |put delta| = 0.25, i.e. call_delta = 0.75 (OTM put)
            k_p25 = min(common_strikes, key=lambda k: abs(call_deltas[k] - 0.75))

            if k_c25 in calls.index and k_c25 in puts.index:
                cr = calls.loc[k_c25]
                call25d_bid  = float(cr['bid_px_00'])
                call25d_ask  = float(cr['ask_px_00'])
                call25d_mid  = (call25d_bid + call25d_ask) / 2.0
                call25d_strike = k_c25
                if call25d_mid < MID_FLOOR:
                    call25d_strike = call25d_mid = call25d_bid = call25d_ask = float('nan')

            if k_p25 in calls.index and k_p25 in puts.index:
                pr = puts.loc[k_p25]
                put25d_bid  = float(pr['bid_px_00'])
                put25d_ask  = float(pr['ask_px_00'])
                put25d_mid  = (put25d_bid + put25d_ask) / 2.0
                put25d_strike = k_p25
                if put25d_mid < MID_FLOOR:
                    put25d_strike = put25d_mid = put25d_bid = put25d_ask = float('nan')

            # IVs at 25Δ strikes for skew/curvature
            iv_c25 = iv_p25 = float('nan')
            if not math.isnan(call25d_mid) and not math.isnan(call25d_strike):
                iv_c25 = implied_vol(call25d_mid, F, call25d_strike, T, is_call=True)
            if not math.isnan(put25d_mid) and not math.isnan(put25d_strike):
                iv_p25 = implied_vol(put25d_mid, F, put25d_strike, T, is_call=False)

            if not math.isnan(iv_c25) and not math.isnan(iv_p25):
                rr25_iv = iv_c25 - iv_p25
                bf25_iv = 0.5 * (iv_c25 + iv_p25) - atm_iv

        rows.append({
            'timestamp_utc':    ts.isoformat(),
            'underlying_price': round(F, 4),
            'atm_strike':       k_atm,
            'expiry_date':      expiry.isoformat(),
            'time_to_expiry':   round(T, 6),
            'call_mid':         round(call_mid, 4),
            'put_mid':          round(put_mid, 4),
            'call_bid':         round(call_bid, 4),
            'call_ask':         round(call_ask, 4),
            'put_bid':          round(put_bid, 4),
            'put_ask':          round(put_ask, 4),
            # new columns
            'atm_iv':           round(atm_iv, 6) if not math.isnan(atm_iv) else float('nan'),
            'call25d_strike':   call25d_strike,
            'call25d_mid':      round(call25d_mid, 4) if not math.isnan(call25d_mid) else float('nan'),
            'call25d_bid':      round(call25d_bid, 4) if not math.isnan(call25d_bid) else float('nan'),
            'call25d_ask':      round(call25d_ask, 4) if not math.isnan(call25d_ask) else float('nan'),
            'put25d_strike':    put25d_strike,
            'put25d_mid':       round(put25d_mid, 4) if not math.isnan(put25d_mid) else float('nan'),
            'put25d_bid':       round(put25d_bid, 4) if not math.isnan(put25d_bid) else float('nan'),
            'put25d_ask':       round(put25d_ask, 4) if not math.isnan(put25d_ask) else float('nan'),
            'rr25_iv':          round(rr25_iv, 6) if not math.isnan(rr25_iv) else float('nan'),
            'bf25_iv':          round(bf25_iv, 6) if not math.isnan(bf25_iv) else float('nan'),
        })

    return rows, s_prev


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Preprocess OPRA SPY options to multi-day chain panel CSV')
    parser.add_argument('--days', type=int, default=127,
                        help='Number of trading days to process (default: 127)')
    parser.add_argument('--zip',  type=str,
                        default='demo/data/OPRA-20260208-3T68RYYKF9.zip',
                        help='Path to OPRA zip file')
    parser.add_argument('--out',  type=str,
                        default='demo/data/spy_chain_panel.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_path = Path(args.out)

    if not zip_path.exists():
        print(f'[ERROR] zip not found: {zip_path}', file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'[prepare_spy_data] Opening zip: {zip_path}')
    zf = zipfile.ZipFile(zip_path)

    day_files = sorted(
        n for n in zf.namelist()
        if re.match(r'opra-pillar-\d{8}\.cbbo-1m\.dbn\.zst', n)
    )
    day_files = day_files[:args.days]
    print(f'[prepare_spy_data] Processing {len(day_files)} trading days: '
          f'{day_files[0]} … {day_files[-1]}')

    all_rows = []
    s_prev = S_INIT

    for i, day_name in enumerate(day_files):
        date_str = re.search(r'(\d{8})', day_name).group(1)
        bar_date = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        print(f'  [{i+1}/{len(day_files)}] {date_str}', end='', flush=True)

        df = load_day(zf, day_name)
        day_rows, s_prev = process_day(df, bar_date, s_prev)
        all_rows.extend(day_rows)
        n_with_iv    = sum(1 for r in day_rows if not math.isnan(r.get('atm_iv', float('nan'))))
        n_with_25d   = sum(1 for r in day_rows if not math.isnan(r.get('rr25_iv', float('nan'))))
        print(f'  → {len(day_rows)} bars  '
              f'atm_iv={n_with_iv}/{len(day_rows)}  '
              f'25Δ={n_with_25d}/{len(day_rows)}  '
              f'S_last={s_prev:.2f}')

    zf.close()

    result = pd.DataFrame(all_rows)

    # ── rolling realised vol (session-reset: computed per-day block) ──────────
    # Assign session labels so rv5 doesn't bleed across day boundaries
    result['_session'] = result['timestamp_utc'].str[:10]  # YYYY-MM-DD prefix
    log_ret = np.log(result['underlying_price'] / result['underlying_price'].shift(1))
    # Zero out cross-session log-returns (first bar of each new day)
    session_change = result['_session'] != result['_session'].shift(1)
    log_ret[session_change] = float('nan')
    result['rv5_ann'] = (
        log_ret.rolling(5).std() * np.sqrt(252 * 390)
    ).fillna(0.0).round(6)
    result.drop(columns=['_session'], inplace=True)

    result.to_csv(out_path, index=False)
    print(f'\n[prepare_spy_data] Written {len(result)} rows → {out_path}')
    print(f'  Date range:    {result["timestamp_utc"].iloc[0][:10]}  …  {result["timestamp_utc"].iloc[-1][:10]}')
    print(f'  SPY range:     {result["underlying_price"].min():.2f} – {result["underlying_price"].max():.2f}')
    print(f'  Expiries used: {sorted(result["expiry_date"].unique())}')
    print(f'  atm_iv null:   {result["atm_iv"].isna().mean():.1%}')
    print(f'  rr25_iv null:  {result["rr25_iv"].isna().mean():.1%}')
    call_spread = (result['call_ask'] - result['call_bid']).mean()
    put_spread  = (result['put_ask']  - result['put_bid']).mean()
    print(f'  Avg spread:    call ${call_spread:.4f}  put ${put_spread:.4f}')
    rv_mean = result.loc[result['rv5_ann'] > 0, 'rv5_ann'].mean()
    print(f'  Avg RV5 (ann): {rv_mean:.4f} ({rv_mean*100:.2f}%)')


if __name__ == '__main__':
    main()
