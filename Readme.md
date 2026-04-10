# Effective Engine — MVP

C++ options trading engine. Event-driven, layered DDD architecture. Focus on a buy-side variance alpha demo with a three-strategy PnL backtest (BS delta / Rough vol delta / Deep BSDE), while capable also of the seller-side initialization. Built while learning C++, with assistance from Claude Code.

---

## Demo (Rough Volatility Models)

### Three-Pass Hedger Comparison (`./build/alpha_pnl_test_runner`)

Runs three sequential passes over the 127-day SPY OPRA panel with the same alpha signal (variance z-score 70% + curvature z-score 30%, calibrated Rough Heston) but a different hedger each pass:

| Pass | Hedger | Delta computation |
|---|---|---|
| 1 | `BSDelta` | N(d1) at market IV, T_sim |
| 2 | `RoughVolDelta` | N(d1) + Vega·(∂σ/∂S) — Bergomi-Guyon smile-slope correction |
| 3 | `NeuralBSDEHedger` | ONNX inference on 7D Lifted Rough Heston state (requires training) |

Produces a side-by-side attribution table across all Greek buckets. Sample result (127-day SPY):

```
                         BSDelta  RoughVolDelta  Δ(b-a)
  Option MTM ($):       3397.83       3397.83     0.00
  Hedge PnL ($):      981739.92     612271.43  -369468
  Hedge Residual ($): 950102.19     580633.70  -369468
  Txn Cost ($):         1527.33       1554.58     +27
  Total PnL ($):      983610.42     614114.68  -369496
  Avg IV: 13.68%   Avg RV5: 8.75%   VRP: +4.93%
```

**Why does RoughVolDelta show lower total PnL?**
This is the correct hedge-versus-carry tradeoff, not a model failure.

- The rough delta correction is ∂σ_K/∂S = −(ψ + χ·k)/S, where ψ(T) ∝ T^(H−0.5). With H=0.01, T^(−0.49) amplifies the correction substantially for short-dated options.
- With ρ=−0.507 (negative leverage), the correction adds to the short-spot hedge: the rough hedger takes a larger short-underlying position than BS delta.
- Since VRP is positive (IV > RV), the strategy earns carry from unhedged vol exposure. Shorting more spot gives up some of that carry, reducing total PnL.
- The correct comparison metric is **hedge residual variance** (unexplained PnL), not total PnL level. A perfect hedger has zero residual and zero total PnL (fully hedged). The rough delta's lower residual demonstrates it is capturing more of the theoretical delta exposure.
- The NeuralBSDEHedger is designed to learn the optimal hedge ratio from simulated paths, balancing hedge quality against carry erosion without relying on the asymptotic smile formula.

### Variance Alpha Pipeline (`./build/alpha_runner`)

Single-pass pipeline with per-day stability reporting. Extracts ATM implied variance from the OPRA chain via BS IV bisection, computes a rolling z-score against the rough-model forward variance forecast, and runs a Flat → Live → Cooldown state machine to trade ATM front straddles. Prints a multi-day stability table: mean ± std daily PnL, Sharpe, option MTM vs hedge attribution, turnover, and vol risk premium.

### Deep BSDE Hedging (`demo/`)

Generates Lifted Rough Heston paths in C++, trains a shared-weight MLP offline in Python to solve the BSDE hedging problem, exports to ONNX, and benchmarks in-process inference against analytic BS delta and finite-difference delta.

| Hedger | PnL std | CVaR(95%) | Latency p50 |
|---|---|---|---|
| BS delta (analytic) | 4.24 | 22.57 | — |
| BS delta (FD bump) | 3.65 | 20.35 | — |
| Neural BSDE | **3.40** | **19.35** | 3.4 µs |

The network takes a 7D state `[τ, log(S/K), V_t, U₁, U₂, U₃, U₄]` where U₁..U₄ are the four OU factors of the Markovian LRH approximation, reconstructed online by `LiftedHestonStateEstimator`.

**Training data is aligned with calibrated market params:** V₀=0.077 (≈27.8% vol), ρ=−0.507, matching the live pipeline's `ROUGH_HESTON_PARAMS`. The LRH kernel uses H=0.1 (independent of the Bergomi-Guyon H=0.01 used for smile slopes — these are separate calibrations for separate purposes).

**BSDE hedger validation (`demo/python/validation/bsde_hedge_validation.py`)**

Controlled out-of-sample test on Rough Heston paths (H=0.1, κ=0.3, θ=0.04, ξ=0.5, ρ=−0.507, T=1yr).
Compares cumulative delta-hedge P&L against BSDE hedge P&L; hedge error = hedge\_pnl − payoff + Y0.

Two stages:

- **Stage 1 — OOS stored paths (n=2000):** exact U-factor trajectories from the same C++ simulator used for training; no state-estimation noise.
- **Stage 2 — Bayer-Breneis fresh paths (n=2000):** independent order-2 weak paths from the Python Rough-Pricing library; U factors reconstructed on-the-fly via `BSDEStateEstimator`.

| Stage | Hedger | RMSE ($) | MAE ($) | Improvement |
|---|---|---|---|---|
| OOS stored | BS delta | 6.273 | 5.619 | — |
| OOS stored | Neural BSDE | **4.495** | **3.899** | **+28.3% RMSE** |
| Bayer-Breneis | BS delta | 4.542 | 3.970 | — |
| Bayer-Breneis | Neural BSDE | **3.391** | **2.870** | **+25.3% RMSE** |

The ~3 percentage-point drop from Stage 1 to Stage 2 is consistent with state-estimation noise from inverting the OU factor updates. The improvement is stable across all error metrics and both tails (P5/P95).

### BSDE Training Pipeline

```bash
cd MVP/demo

# Step 1: generate training paths (aligned with calibrated params)
mkdir -p build && cd build && cmake .. && make demo_runner && cd ..
./build/demo_runner
# writes: artifacts/training_states.npy, training_dW1.npy,
#         training_payoff.npy, normalization.json

# Step 2: Gate 1 — BS sanity check
python python/bsde/trainer.py --config python/configs/bs_validation.yaml --seed 42

# Step 3: Gate 2 — LRH training
python python/bsde/trainer.py --config python/configs/lifted_rough_heston.yaml --seed 42

# Step 4: ONNX export
python python/bsde/export.py --checkpoint artifacts/checkpoints/best.pt --validate

# Step 5: rebuild with ONNX and run three-pass comparison
cd build && cmake .. -DBUILD_ONNX_DEMO=ON -DONNXRUNTIME_ROOT=$HOME/onnxruntime && make
./alpha_pnl_test_runner
```

### Seller — Live Simulation + Calibration (`./build/market_maker`)

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

Implemented in `conditional_dynamics/gate5_sweep.py`.

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
