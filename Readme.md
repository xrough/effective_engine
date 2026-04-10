# Effective Engine — MVP

C++ options trading engine. Event-driven, layered DDD architecture. Focus on a buy-side variance alpha demo, with a separate deep BSDE neural hedging demo, while capable also of the seller-side initialization. Built while learning C++, with assistance from Claude Code.

## What it does
**Buyer — variance alpha demo (`./build/alpha_runner`)**

Extracts ATM implied variance from synthetic option quotes via BS IV bisection, computes a rolling z-score against the rough-model forward variance forecast (`xi0 * T`), and runs a Flat → Live → Cooldown state machine to trade ATM front straddles when the spread is statistically significant. Delta-hedges the resulting position and reports a PnL breakdown (option MTM, delta hedge PnL, transaction cost).

**Deep BSDE hedging demo ([demo/](demo/Readme.md))**

Generates lifted rough Heston paths in C++, trains a shared-weight MLP offline in Python to solve the BSDE hedging problem, exports to ONNX, and benchmarks in-process inference against analytic BS delta and finite-difference delta.

| Hedger | PnL std | CVaR(95%) | Latency p50 |
|---|---|---|---|
| BS delta (analytic) | 4.24 | 22.57 | — |
| BS delta (FD bump) | 3.65 | 20.35 | — |
| Neural BSDE | **3.40** | **19.35** | 3.4 µs |

**Seller — live simulation + calibration (`./build/market_maker`)**

Quotes bid/ask spreads (Rough Bergomi skew), simulates a probabilistic counterparty (30% fill), runs threshold-based delta hedging, enforces live risk limits (max loss $1M, max delta 10,000), then replays on an isolated bus to calibrate implied volatility via golden-section search and hot-inject the result.

## Rough volatility research (`demo/python/research/`)

A sequential gate system testing whether SPY intraday option data (Databento OPRA, 127 trading days, Aug 2025 – Feb 2026) contains the structural signatures predicted by rough-volatility theory (H ≈ 0.10).

The shared pipeline recovers the forward from call-put parity, then extracts ATM IV, 25-delta risk reversal (rr25 = IV_25c − IV_25p), and butterfly (bf25) via bisection IV. Rough-vol structural coefficients α = rr25 / (T^{H−0.5} · σ_ATM) and γ = bf25 / (T^{2H−1} · ATV) are computed per bar and used as 1-step-ahead forecasters.

### Gate 1: Skew Structure (`skew_scaling/`)

**Hypothesis:** |rr25(T)| ∝ T^H across maturities at any snapshot — `log|rr25| = a + β·log(T)` should give β ≈ H = 0.10.

**Result (127 days, ~48 000 timestamps, ≥3 expiries each):**

| Statistic | Value | Target |
|---|---|---|
| Median β | +0.21 | +0.10 |
| Mean R² | 0.86 | > 0.70 |
| β CV | 0.42 | < 0.50 |

**Verdict: WEAK.** A genuine power-law term structure exists (R² = 0.86, stable), but the slope β ≈ +0.21 is systematically above the rough-Bergomi prediction. The scaling shape is confirmed; the exponent does not match the H = 0.10 prior.

### Gate 2: Temporal Forecast (`roughtemporal_intraday/`)

**Hypothesis:** The rough-vol structural coefficients produce mean-reverting residuals and beat naive carry on 1-step-ahead rr25/bf25 RMSE.

**Robustness sweep (90 days, H ∈ {0.03–0.20}, resample ∈ {1–60 min}, 30 cells):**

```
H \ resample |  1 min |  5 min | 15 min | 30 min | 60 min
----------------------------------------------------------
     all H   |   FAIL |   FAIL | MARG.  | MARG.  | MARG.
```

**Verdict: 0/30 PASS.** Rough-struct underperforms carry at 1–5 min. The gap narrows at coarser bars but never crosses the PASS threshold.

### Gate 3: Regime Dynamics (`conditional_dynamics/`)

**Hypothesis:** Any rough-vol edge concentrates in the ACTIVE regime (top 20% of |spot returns|) and is absent in QUIET.

**Robustness sweep (90 days, 126 cells: H × resample × move_pct):** All cells FAIL or MARGINAL. No regime-specific advantage detected. **Verdict: FAIL.**

### Gate 4: Incremental Edge (`roughtemporal_intraday/gate1_sweep.py`)

**Hypothesis:** A composite rough-vol forecaster (carry + recent structural shift) beats carry by >2% on bf25 RMSE.

**Robustness sweep (90 days, H ∈ {0.03–0.20}, resample ∈ {1–60 min}, 30 cells):**

```
H \ resample |  1 min |  5 min | 15 min | 30 min | 60 min
----------------------------------------------------------
     all H   |   PASS |   PASS |   PASS |  MARG. |   PASS
```

**Verdict: PASS — 24/30 cells (80%).** bf25 shows a robust 3–8% RMSE improvement over carry across all tested H values and bar sizes from 1 to 60 min. Result is H-insensitive. The 30-min bar is marginal (+0.1%).

### Gate 5: Edge Concentration (`conditional_dynamics/gate1b_sweep.py`)

**Hypothesis:** The Gate 4 improvement concentrates in the ACTIVE regime, confirming a rough-vol mechanism rather than a generic carry improvement.

**Result (90-day sweep, move_pct ∈ {10%, 20%, 30%}):** 0/42 PASS at 10% or 20%. At 30%, 6/42 PASS (5-min bars only). No robust active-vs-quiet separation. **Verdict: WEAK.** The Gate 4 bf25 gain is unconditional — present in both regimes.

### Research infrastructure

| File | Role |
|---|---|
| [demo/python/research/shared/smile_pipeline.py](demo/python/research/shared/smile_pipeline.py) | IV extraction, BS math, `evaluate_forecasts`, GPU bisection |
| [demo/python/research/shared/robustness_sweeps.py](demo/python/research/shared/robustness_sweeps.py) | Cache, `resample_panel`, `pool_by_tenor_bucket`, cell classifiers |
| [demo/python/research/shared/synthetic_smile.py](demo/python/research/shared/synthetic_smile.py) | Known-truth rough-smile generator for pipeline validation |
| [demo/python/research/tests/smoke_test.py](demo/python/research/tests/smoke_test.py) | 14 no-data unit tests + synthetic roundtrip |
| [demo/python/research/tests/test_robustness_sweeps.py](demo/python/research/tests/test_robustness_sweeps.py) | 13 unit tests for sweep helpers |