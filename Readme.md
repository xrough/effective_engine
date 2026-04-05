# Effective Engine — MVP

C++ options trading engine. Event-driven, layered DDD architecture. Covers sell-side market making and a buy-side variance alpha demo, with a separate deep BSDE neural hedging demo. Built while learning C++, with assistance from Claude Code.

## Quick start

```bash
# Seller pipeline (market making + calibration)
cmake -S . -B build && cmake --build build --parallel
./build/market_maker

# Buyer alpha pipeline demo
cd demo && mkdir -p build && cd build && cmake .. && make alpha_runner
cd .. && ./build/alpha_runner

# Deep BSDE neural hedging demo
cd demo && mkdir -p build && cd build && cmake .. && make demo_runner
cd .. && ./build/demo_runner     # see demo/Readme.md for full pipeline
```

## What it does

**Seller — live simulation + calibration (`./build/market_maker`)**

Quotes bid/ask spreads (Rough Bergomi skew), simulates a probabilistic counterparty (30% fill), runs threshold-based delta hedging, enforces live risk limits (max loss $1M, max delta 10,000), then replays on an isolated bus to calibrate implied volatility via golden-section search and hot-inject the result.

**Buyer — variance alpha demo (`./build/alpha_runner`)**

Extracts ATM implied variance from synthetic option quotes via BS IV bisection, computes a rolling z-score against the rough-model forward variance forecast (`xi0 * T`), and runs a Flat → Live → Cooldown state machine to trade ATM front straddles when the spread is statistically significant. Delta-hedges the resulting position and reports a PnL breakdown (option MTM, delta hedge PnL, transaction cost).

**Deep BSDE hedging demo ([demo/](demo/Readme.md))**

Generates lifted rough Heston paths in C++, trains a shared-weight MLP offline in Python to solve the BSDE hedging problem, exports to ONNX, and benchmarks in-process inference against analytic BS delta and finite-difference delta.

| Hedger | PnL std | CVaR(95%) | Latency p50 |
|---|---|---|---|
| BS delta (analytic) | 4.24 | 22.57 | — |
| BS delta (FD bump) | 3.65 | 20.35 | — |
| Neural BSDE | **3.40** | **19.35** | 3.4 µs |

## Architecture

```
src/
  core/
    events/         EventBus (type-erased pub/sub), Events
    domain/         Instrument, PositionManager, RiskMetrics
    analytics/      IPricingEngine, BS/RoughVol engines, CalibrationEngine,
                    ImpliedVarianceExtractor
    application/    PortfolioService
    infrastructure/ MarketDataAdapter, ParameterStore
    interfaces/     IEntryPolicy, IExecutionPolicy, IHedgeStrategy, IQuoteStrategy

  modules/
    seller/         QuoteEngine, DeltaHedger, SellerRiskApp, SellerModule,
                    BacktestCalibrationApp, ProbabilisticTaker
    buyer/          IAlphaSignal (buyer-local interface)
                    BuyerModule  (wiring template — accepts injected strategy impls)

demo/cpp/           Concrete buyer strategy (VarianceAlphaSignal, StrategyController)
                    + simulation adapters (SyntheticOptionFeed, SimpleExecSim)
                    + PnL attribution (AlphaPnLTracker)
                    + Deep BSDE path generation (LiftedHestonSim)
```

**Boundary rule:**
- `src/core/` — reusable tools and contracts (domain, analytics, infrastructure, interfaces)
- `src/modules/seller/` — the market-making engine
- `src/modules/buyer/` — wiring pattern and buyer-local interface only; no concrete strategy logic
- `demo/cpp/` — one specific buyer strategy built on top of the engine; replace freely without touching `src/`

**Event flow (seller):**
```
MarketDataEvent → QuoteEngine → QuoteGeneratedEvent
                → ProbabilisticTaker → FillEvent
                  → PortfolioService / DeltaHedger → OrderSubmittedEvent
                  → SellerRiskApp → RiskControlEvent / RiskAlertEvent
```

**Event flow (buyer alpha demo):**
```
MarketDataEvent → SyntheticOptionFeed → OptionMidQuoteEvent
               → ImpliedVarianceExtractor (σ²_atm)
               → VarianceAlphaSignal → SignalSnapshotEvent (z-score)
               → StrategyController → OrderSubmittedEvent (straddle entry)
               → SimpleExecSim → FillEvent
               → DeltaHedger / AlphaPnLTracker
```

`src/main.cpp` is the sole composition root for the seller pipeline. `demo/cpp/alpha_main.cpp` is the composition root for the buyer demo.

## Key files

| File | Role |
|---|---|
| [src/core/events/EventBus.hpp](src/core/events/EventBus.hpp) | Type-erased pub/sub dispatcher |
| [src/core/analytics/PricingEngine.hpp](src/core/analytics/PricingEngine.hpp) | `IPricingEngine` + BS / intrinsic implementations |
| [src/core/analytics/RoughVolPricingEngine.hpp](src/core/analytics/RoughVolPricingEngine.hpp) | Rough Bergomi skew with hot parameter injection |
| [src/core/analytics/ImpliedVarianceExtractor.hpp](src/core/analytics/ImpliedVarianceExtractor.hpp) | BS IV bisection → σ², σ²T |
| [src/core/interfaces/IEntryPolicy.hpp](src/core/interfaces/IEntryPolicy.hpp) | Trade entry decision interface |
| [src/modules/buyer/IAlphaSignal.hpp](src/modules/buyer/IAlphaSignal.hpp) | Buyer-local signal interface |
| [src/modules/buyer/BuyerModule.hpp](src/modules/buyer/BuyerModule.hpp) | Buyer wiring template (injection pattern) |
| [src/modules/seller/SellerModule.hpp](src/modules/seller/SellerModule.hpp) | Seller composition entry point |
| [demo/cpp/VarianceAlphaSignal.hpp](demo/cpp/VarianceAlphaSignal.hpp) | Concrete alpha signal (rolling z-score) |
| [demo/cpp/StrategyController.hpp](demo/cpp/StrategyController.hpp) | Flat/Live/Cooldown state machine |
| [demo/cpp/AlphaPnLTracker.hpp](demo/cpp/AlphaPnLTracker.hpp) | Per-instrument PnL attribution |
