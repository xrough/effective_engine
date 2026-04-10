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

A sequential gate system testing in `demo/python/research/` whether SPY intraday option data (OPRA, 127 trading days, Aug 2025 – Feb 2026) contains the structural signatures predicted by rough-volatility theory.

The shared pipeline recovers the forward from call-put parity, then extracts the smile features

$$
\mathrm{rr25} = \mathrm{IV}_{25c} - \mathrm{IV}_{25p},
\qquad
\mathrm{bf25} = \frac{\mathrm{IV}_{25c} + \mathrm{IV}_{25p}}{2} - \mathrm{IV}_{\mathrm{ATM}}.
$$

The rough-vol structural coefficients are then defined as

$$
\alpha = \frac{\mathrm{rr25}}{T^{H-\frac12}\sigma_{\mathrm{ATM}}},
\qquad
\gamma = \frac{\mathrm{bf25}}{T^{2H-1}\,\mathrm{ATV}},
\qquad
\mathrm{ATV} = \sigma_{\mathrm{ATM}}^2 T.
$$

These quantities are computed per bar and then used as the rough structural state in the forecast gates.

### Gate 1: Skew Structure 

Implemented in (`skew_scaling/`)
**Hypothesis:** the short-end skew follows a rough-style power law across maturities,

$$
|\mathrm{rr25}(T)| \propto T^H,
\qquad
\log |\mathrm{rr25}(T)| = a + \beta \log T,
\qquad
\beta \approx H = 0.10.
$$

**Result (127 days, ~48 000 timestamps, ≥3 expiries each):**

| Statistic | Value | Target |
|---|---|---|
| Median β | +0.21 | +0.10 |
| Mean R² | 0.86 | > 0.70 |
| β CV | 0.42 | < 0.50 |

**Verdict: WEAK.** A genuine power-law term structure exists ($R^2 = 0.86$, stable), but the slope $\beta \approx +0.21$ is systematically above the rough-Bergomi prior $\beta \approx 0.10$. The scaling shape is confirmed; the exponent does not match the original $H = 0.10$ prior.

### Gate 2: Temporal Forecast 
Implemented in (`roughtemporal_intraday/`)
**Hypothesis:** the rough structural forecast

$$
\widehat{\mathrm{rr25}}_{t+1}^{\,\mathrm{rough}}
\text{and}
\widehat{\mathrm{bf25}}_{t+1}^{\,\mathrm{rough}}
$$

should produce mean-reverting residuals and beat naive carry on 1-step-ahead $\mathrm{rr25}$ / $\mathrm{bf25}$ RMSE.

**Robustness sweep (90 days, H ∈ {0.03–0.20}, resample ∈ {1–60 min}, 30 cells):**

```
H \ resample |  1 min |  5 min | 15 min | 30 min | 60 min
----------------------------------------------------------
     all H   |   FAIL |   FAIL | MARG.  | MARG.  | MARG.
```

**Verdict: 0/30 PASS.** The raw rough forecast underperforms carry at 1–5 min. The gap narrows at coarser bars but never crosses the PASS threshold.

### Gate 3: Regime Dynamics 
Implemented in (`conditional_dynamics/`). **Hypothesis:** any raw rough-vol edge should concentrate in the ACTIVE regime,
$$
|r_t| > q_{1-\texttt{move\_pct}}\bigl(|r|\bigr), r_t = \log \frac{F_t}{F_{t-1}},
$$
and should be absent in QUIET.

**Robustness sweep (90 days, 126 cells: H × resample × move_pct):** All cells FAIL or MARGINAL. No regime-specific advantage detected. **Verdict: FAIL.**

### Gate 4: Incremental Edge 

Implemented in (`roughtemporal_intraday/gate1_sweep.py`). **Hypothesis:** rough does not need to replace carry; it only needs to improve it. Gate 4 tests two hybrids:

$$
\widehat{x}_{t+1}^{\,\mathrm{cond}} = x_t + a + b\bigl(\widehat{x}^{\,\mathrm{rough}}_t - x_t\bigr),
$$

which is the carry-conditioned rough correction, and a recency-weighted rough forecaster built from EWMA estimates of $\alpha_t$ and $\gamma_t$.

The question is whether these hybrids beat carry by more than $2\%$ on $\mathrm{bf25}$ RMSE.

**Robustness sweep (90 days, H ∈ {0.03–0.20}, resample ∈ {1–60 min}, 30 cells):**

```
H \ resample |  1 min |  5 min | 15 min | 30 min | 60 min
----------------------------------------------------------
     all H   |   PASS |   PASS |   PASS |  MARG. |   PASS
```

**Verdict: PASS — 24/30 cells (80%).** $\mathrm{bf25}$ shows a robust $3\%$ to $8\%$ RMSE improvement over carry across all tested $H$ values and bar sizes from 1 to 60 min. The result is largely $H$-insensitive. The 30-min bar is only marginal ($+0.1\%$).

### Gate 5: Edge Concentration 

Implemented in (`conditional_dynamics/gate1b_sweep.py`). **Hypothesis:** the Gate 4 hybrid improvement should satisfy

$$
\Delta_{\mathrm{active}} > 0, \Delta_{\mathrm{quiet}} \le 0,
$$

so that the rough enhancement is genuinely concentrated in stressed regimes rather than being a generic carry improvement.

**Result (90-day sweep, move\_pct in $\{10\%, 20\%, 30\%\}$):** 0/42 PASS at $10\%$ or $20\%$. At $30\%$, 6/42 PASS (5-min bars only). No broad active-vs-quiet separation survives. **Verdict: WEAK.** The Gate 4 $\mathrm{bf25}$ gain is mostly unconditional, with only a narrow regime-specific pocket.

### Research infrastructure

| File | Role |
|---|---|
| [demo/python/research/shared/smile_pipeline.py](demo/python/research/shared/smile_pipeline.py) | IV extraction, BS math, `evaluate_forecasts`, GPU bisection |
| [demo/python/research/shared/robustness_sweeps.py](demo/python/research/shared/robustness_sweeps.py) | Cache, `resample_panel`, `pool_by_tenor_bucket`, cell classifiers |
| [demo/python/research/shared/synthetic_smile.py](demo/python/research/shared/synthetic_smile.py) | Known-truth rough-smile generator for pipeline validation |
| [demo/python/research/tests/smoke_test.py](demo/python/research/tests/smoke_test.py) | 14 no-data unit tests + synthetic roundtrip |
| [demo/python/research/tests/test_robustness_sweeps.py](demo/python/research/tests/test_robustness_sweeps.py) | 13 unit tests for sweep helpers |
