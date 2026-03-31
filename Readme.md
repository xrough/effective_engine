# Options Market Maker MVP

A C++ simulation of an options market-making desk using a layered DDD architecture and event-driven design. Built while learning C++, with assistance from Claude Code.

## What it does

- Quotes bid/ask spreads on options (Black-Scholes, intrinsic, or Rough Bergomi skew)
- Simulates a probabilistic counterparty generating trades (30% fill probability)
- Threshold-based delta hedging via an outbound order router
- Real-time risk monitoring with loss and delta limit enforcement
- Backtests on historical data and calibrates implied volatility via golden-section search

## Quick start

```bash
cmake -S . -B build && cmake --build build --parallel
./build/market_maker
```

## Architecture

Four-layer DDD structure in `src/`:

| Layer | Path | Responsibility |
|---|---|---|
| Events | `src/core/events/` | `EventBus` pub/sub dispatcher; all domain event definitions |
| Domain | `src/core/domain/` | `Instrument` hierarchy, `InstrumentFactory`, `PositionManager`, `PortfolioAggregate`, `RiskMetrics` |
| Analytics | `src/core/analytics/` | `IPricingEngine` + implementations, `CalibrationEngine`, `IRiskPolicy` |
| Application | `src/core/application/` + `src/modules/` | `PortfolioService`, `QuoteEngine`, `DeltaHedger`, `SellerRiskApp`, `BacktestCalibrationApp` |
| Infrastructure | `src/core/infrastructure/` | `MarketDataAdapter`, `OrderRouter`, `ParameterStore` |
| Composition | `src/main.cpp` | **Only file where concrete class names appear**; wires all components via constructor injection |

## Event flow

```
Phase 1 — Live Simulation:

  MarketDataEvent (MarketDataAdapter)
    ├→ QuoteEngine       → QuoteGeneratedEvent (bid/ask around BS theo)
    ├→ DeltaHedger       → caches spot price
    └→ PortfolioService  → PortfolioUpdateEvent (mark-to-market)

  QuoteGeneratedEvent
    └→ ProbabilisticTaker → FillEvent (30% probability)

  FillEvent
    ├→ PortfolioService  → PortfolioUpdateEvent (position + PnL update)
    └→ DeltaHedger       → OrderSubmittedEvent (if |Δ| > threshold)

  PortfolioUpdateEvent
    └→ SellerRiskApp     → RiskControlEvent / RiskAlertEvent

  OrderSubmittedEvent
    └→ OrderRouter       → send_to_exchange() [stub]

Phase 2 — Backtest & Calibration (isolated bus):

  MarketDataEvent (CSV replay)
    └→ BacktestCalibrationApp → caches (S, option, market_price)
         → CalibrationEngine::solve()  [golden-section, minimizes MSE]
         → ParamUpdateEvent → ParameterStore (main bus)
```

## Key files

- [src/main.cpp](src/main.cpp) — composition root; the only place concrete types are named
- [src/core/events/EventBus.hpp](src/core/events/EventBus.hpp) — type-erased synchronous pub/sub dispatcher
- [src/core/analytics/PricingEngine.hpp](src/core/analytics/PricingEngine.hpp) — `IPricingEngine` interface + BS and simple implementations
- [src/core/analytics/RoughVolPricingEngine.hpp](src/core/analytics/RoughVolPricingEngine.hpp) — Rough Bergomi skew correction with hot parameter injection
- [src/modules/seller/SellerRiskApp.hpp](src/modules/seller/SellerRiskApp.hpp) — live risk monitoring + limit enforcement
- [src/modules/seller/BacktestCalibrationApp.hpp](src/modules/seller/BacktestCalibrationApp.hpp) — historical replay + vol calibration

## Design patterns

| Pattern | Example |
|---|---|
| Observer | `EventBus` — all inter-component communication |
| Strategy | `IPricingEngine`, `IRiskPolicy` — swapped only in `main.cpp` |
| Factory | `InstrumentFactory::make_call()`, `make_put()`, `make_underlying()` |
| Adapter | `MarketDataAdapter` (CSV → events), `OrderRouter` (events → exchange) |
| Command | `OrderSubmittedEvent` encapsulates order intent; `OrderRouter` is the receiver |
| Aggregate Root | `PortfolioAggregate` owns positions + PnL consistency |

**Extensibility rule:** to add a new pricing model or risk policy, implement `IPricingEngine` / `IRiskPolicy` and swap the concrete type only in `main.cpp`.

## Key parameters (configurable in `src/main.cpp`)

| Parameter | Value |
|---|---|
| Quote half-spread | $0.05 |
| Delta hedge threshold | 0.5 |
| Trade probability | 30% |
| RNG seed | 42 |
| Market vol (ground truth) | 0.25 |
| Model initial vol | 0.15 |
| Calibration range | [0.01, 1.0] |
| Risk limit — max loss | $1M (BlockOrders) |
| Risk limit — max delta | 10,000 (ReduceOnly) |
