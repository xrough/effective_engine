# Effective Engine — MVP

C++ options trading engine. Event-driven, layered DDD architecture. Focus on a buy-side variance alpha demo, with a separate deep BSDE neural hedging demo, while capable also of the seller-side initialization. Built while learning C++, with assistance from Claude Code.

## Demo (Rough Volatility Models)
**Buyer — variance alpha demo (`./build/alpha_runner`)**

Extracts ATM implied variance from synthetic option quotes via BS IV bisection, computes a rolling z-score against the rough-model forward variance forecast $\xi_0 T$, and runs a Flat → Live → Cooldown state machine to trade ATM front straddles when the spread is statistically significant. Delta-hedges the resulting position and reports a PnL breakdown (option MTM, delta hedge PnL, transaction cost).

**Deep BSDE hedging demo ([demo/](demo/))**

Generates lifted rough Heston paths in C++, trains a shared-weight MLP offline in Python to solve the BSDE hedging problem, exports to ONNX, and benchmarks in-process inference against analytic BS delta and finite-difference delta.

| Hedger | PnL std | CVaR(95%) | Latency p50 |
|---|---|---|---|
| BS delta (analytic) | 4.24 | 22.57 | — |
| BS delta (FD bump) | 3.65 | 20.35 | — |
| Neural BSDE | **3.40** | **19.35** | 3.4 µs |

**Seller — live simulation + calibration (`./build/market_maker`)**

Quotes bid/ask spreads (Rough Bergomi skew), simulates a probabilistic counterparty (30% fill), runs threshold-based delta hedging, enforces live risk limits (max loss $1M, max delta 10,000), then replays on an isolated bus to calibrate implied volatility via golden-section search and hot-inject the result.

## Research gates

The research stack in `demo/python/research/` is organized as a sequence of
falsification gates. Each gate tests a different claim about rough-volatility
smile geometry or rough-volatility alpha on SPY OPRA intraday data.

The shared pipeline recovers the forward from call-put parity, then extracts
the smile features:

```text
rr25 = IV_25c - IV_25p
bf25 = (IV_25c + IV_25p)/2 - IV_ATM
```

and the rough structural coefficients:

```text
alpha = rr25 / (T^(H-1/2) * sigma_ATM)
gamma = bf25 / (T^(2H-1) * ATV)
ATV   = sigma_ATM^2 * T
```

These are then fed into the temporal and conditional forecast benchmarks.

### Gate 1: Skew Structure

Implemented in `skew_scaling/`.

**Hypothesis:** the short-end skew follows a stable power-law term structure
across maturities, so a cross-sectional regression
`log |rr25(T)| = a + beta * log(T)` should reveal a persistent rough-style
maturity slope.

**Latest full benchmark:** 127 days, 44,483 timestamps, mean 11.5 expiries per timestamp.

| Statistic | Value |
|---|---|
| Median `beta` | +0.2123 |
| Mean `R^2` | 0.8606 |
| Median `R^2` | 0.9068 |
| `beta` CV | 0.4210 |

**Verdict: WEAK / PARTIAL SUPPORT.** The cross-sectional power-law shape is
clearly present and statistically stable, but the implied slope is materially
above the original `H = 0.10` prior. So the smile geometry looks rough-like,
but the fitted exponent is not a clean confirmation of the original parameter
choice.

### Gate 2: Temporal Forecast

Implemented in `roughtemporal_intraday/`.

**Hypothesis:** the raw rough structural forecasts
`rr25_hat_rough(t+1)` and `bf25_hat_rough(t+1)` should beat naive carry on
one-step-ahead smile prediction.

**Latest robustness sweep:** `gate0_sweep_20260409_174933`

- 127 trading days
- `H in {0.03, 0.05, 0.07, 0.10, 0.15, 0.20}`
- `resample in {1, 5, 15, 30, 60}` min
- `0/30` evaluable cells PASS

```
H \ resample |  1 min |  5 min | 15 min | 30 min | 60 min
----------------------------------------------------------
     all H   |   FAIL |   FAIL | MARG.  | MARG.  | MARG.
```

**Verdict: REJECTED.** The raw rough forecast loses to carry at 1–5 min and
remains worse on aggregate `rr25` RMSE even when the gap narrows at
coarser bars.

### Gate 3: Regime Dynamics

Implemented in `conditional_dynamics/`.

**Hypothesis:** even if raw rough loses unconditionally, it may still add value
after large spot or forward moves. ACTIVE bars are defined by:

```text
|r_t| > percentile_{1-move_pct}(|r|)
r_t  = log(F_t / F_{t-1})
```

**Latest fair-scoring sweep:** `gate0b_sweep_20260410_082811`

- 90 trading days
- full-history forecasts with active/quiet masked scoring
- surviving region: only `390m`
- surviving `move_pct`: `10%` and `20%`
- surviving feature: `rr25`

**Verdict: NARROW SUPPORT.** Raw rough conditional alpha is not broadly
present, but a narrow daily-ish skew signal survives at `390m` in stressed
regimes.

### Gate 4: Incremental Edge Over Carry

Implemented in `roughtemporal_intraday/gate1_sweep.py`.

**Hypothesis:** rough does not need to replace carry; it only needs to improve
it. Gate 4 tests two hybrids:

```text
x_hat_cond(t+1) = x_t + a + b * (x_hat_rough(t) - x_t)
```

This is the carry-conditioned rough correction. Gate 4 also tests a
recency-weighted rough forecaster built from EWMA estimates of `alpha_t` and
`gamma_t`.

**Latest robustness sweep:** `gate1_sweep_20260410_090610`

- 90 trading days
- `24/30` evaluable cells PASS
- `1m`, `5m`, `15m`, and `60m` pass across the full `H` grid
- `30m` is only marginal
- the winning feature is overwhelmingly `bf25`
- `rough_cond_carry` leads at `1m` to `15m`
- `rough_recent` leads at `60m`

```
H \ resample |  1 min |  5 min | 15 min | 30 min | 60 min
----------------------------------------------------------
     all H   |   PASS |   PASS |   PASS |  MARG. |   PASS
```

**Verdict: PASS.** Rough geometry adds useful information to carry, mainly on
smile curvature (`bf25`) rather than skew.

### Gate 5: Edge Concentration

Implemented in `conditional_dynamics/gate1b_sweep.py`.

**Hypothesis:** the Gate 4 improvement should be stronger in ACTIVE than in
QUIET, so that the rough enhancement is genuinely concentrated in stressed
regimes:

```text
Delta_active > 0
Delta_quiet <= 0
```

**Latest sweep:** `gate1b_sweep_20260410_095254`

- 90 trading days
- `move_pct = 10\%`: `0/42` PASS
- `move_pct = 20\%`: `0/42` PASS
- `move_pct = 30\%`: `6/42` PASS
- all 6 PASS cells are:
  - `5m`
  - all tested `H`
  - `rough_cond_carry`
  - `bf25`

**Verdict: WEAK / NARROW SUPPORT.** The regime-specific improvement exists,
but only in a narrow conditional curvature pocket. Most of the Gate 4 gain
appears to be unconditional rather than uniquely stress-driven.
