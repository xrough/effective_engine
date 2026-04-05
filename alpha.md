# Variance Alpha Strategy: Implementation Note

## Objective

Implement a standalone strategy layer on top of the existing engine that compares:

- market short-dated ATM implied variance
- rough-model forecast of future realized variance

and converts the discrepancy into target option positions plus hedge instructions.

The strategy layer should **reuse** the existing:

- data adapters
- rough calibration / simulation
- pricing and Greeks engine
- delta hedger

It should only add a thin decision layer.

---

## Minimal Components

### 1. `ImpliedVarianceExtractor`
Responsibility:
- read the short-dated option chain
- select expiry nearest the forecast horizon
- select strike nearest ATM-forward
- build a simple ATM implied variance proxy

Output:
- ATM implied vol
- ATM implied variance
- total implied variance

Suggested output type:

```cpp
struct ImpliedVariancePoint {
    bool valid;
    Date expiry;
    double strike;
    double atm_implied_vol;
    double atm_implied_variance;
    double total_implied_variance;
    double call_mid;
    double put_mid;
};
```

---

### 2. `VarianceAlphaSignal`
Responsibility:
- query the rough forecast for the same horizon
- compare market implied variance against forecast realized variance
- standardize the spread into a tradable signal

Core signal:

\[
s_t = IVar^{mkt}_{t,T} - \widehat{RV}^{rough}_{t,T}
\]

Normalized version:

\[
z_t = rac{s_t - m_t}{\sigma_t}
\]

where \(m_t\) and \(\sigma_t\) are rolling mean and rolling standard deviation.

Suggested output type:

```cpp
struct SignalSnapshot {
    Timestamp ts;
    bool valid;
    Date expiry;

    double atm_implied_variance;
    double rough_forecast_realized_variance;
    double raw_spread;
    double zscore;

    bool event_blocked;
    bool liquidity_ok;
    bool calibration_ok;
};
```

---

### 3. `StrategyController`
Responsibility:
- map signal to target positions
- maintain a simple state machine
- trigger entry, exit, resize, and roll

Suggested states:

```cpp
enum class StrategyState {
    Flat,
    Live,
    Cooldown
};
```

Suggested target output:

```cpp
enum class TradeType {
    None,
    LongFrontVariance,
    ShortFrontVariance
};

struct TargetBook {
    Timestamp ts;
    TradeType trade_type;
    double target_vega;
    double target_delta_band;
    bool enter;
    bool exit;
    bool resize;
};
```

Version 1 should only support:
- long ATM front straddle
- short ATM front straddle

No skew trades, no multi-leg optimization, no multi-horizon logic.

---

## Trading Logic

### Entry
- if `zscore > z_entry`: market front variance is rich -> short front variance
- if `zscore < -z_entry`: market front variance is cheap -> long front variance

### Exit
- if `|zscore| < z_exit`
- or max holding period reached
- or event block triggered
- or stop-loss triggered
- or expiry too close

### Position sizing
Use signal-scaled vega targeting:

```cpp
double scale = std::min(std::abs(zscore) / z_cap, 1.0);
target_vega = base_vega_budget * scale;
```

Then convert target vega into call/put quantities using the selected ATM straddle.

---

## Event-Driven Integration

### Recompute alpha on:
- `OnBarClose`
- `OnOptionQuoteBatchUpdate`
- `OnSurfaceRebuilt`

### Do not recompute alpha on:
- every tick

### Use high-frequency events only for:
- delta hedge checks
- stop checks
- risk limit checks

Recommended event flow:

```text
Option chain update
    -> extract ATM implied variance
    -> query rough forecast realized variance
    -> update signal
    -> update state machine
    -> compute target book
    -> submit orders
    -> pass resulting position to existing delta hedger
```

---

## Risk Gating

Before entry, require:

- acceptable bid-ask / liquidity
- no major scheduled event in blocked window
- acceptable rough calibration quality
- sufficient risk budget

During live position, monitor:

- delta drift
- gamma exposure
- hedge cost
- stop-loss
- proximity to expiry

Version 1 should keep this simple and explicit.

---

## Required Interfaces

```cpp
double forecast_realized_variance(
    const Timestamp& now,
    const Date& expiry,
    const RoughState& state
);

ImpliedVariancePoint extract_implied_variance(
    const MarketSnapshot& market,
    const OptionChain& chain,
    const Date& expiry
);

SignalSnapshot update_signal(
    const Timestamp& ts,
    const ImpliedVariancePoint& iv_point,
    double rough_forecast_realized_variance,
    const StrategyContext& ctx
);

TargetBook compute_target_book(
    const SignalSnapshot& signal,
    const CurrentBook& current_book,
    const RiskLimits& limits
);
```

---

## Minimal PnL Attribution

Keep this separate from existing book-level reporting.

```cpp
struct StrategyPnLBreakdown {
    double total_pnl;
    double option_mtm;
    double delta_hedge_pnl;
    double transaction_cost;
};
```

At minimum, verify that strategy PnL is driven by convergence of the variance spread rather than accidental directional exposure.

---

## Implementation Order

1. Build ATM implied variance extraction.
2. Expose rough forecast realized variance for the same expiry.
3. Implement rolling spread and z-score signal.
4. Implement simple `Flat -> Live -> Cooldown` state machine.
5. Trade only ATM front straddles.
6. Reuse existing delta hedger.
7. Add basic PnL attribution and event gating.
8. Only after this works, extend to calendar spreads or richer normalization.

---

## Practical Rule

Do not start with full rough-Bergomi execution logic inside the strategy loop.

Start with the narrow pipeline:

**ATM option chain -> implied variance proxy -> rough forecast variance -> z-score signal -> target straddle position -> delta hedge -> PnL attribution**

If this pipeline is stable, everything else can be upgraded later.
