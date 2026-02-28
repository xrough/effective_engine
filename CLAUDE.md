# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# First time setup
mkdir -p build && cd build && cmake .. && make

# Rebuild after changes
cd build && make

# Run simulation
./build/market_maker
```

No test suite exists yet. Correctness is verified by inspecting the simulation output (event log + final position summary).

## Architecture — Current MVP

Four-layer DDD structure in `src/`:

| Layer | Path | Responsibility |
|---|---|---|
| Events | `src/events/` | `EventBus` pub/sub dispatcher; domain event definitions |
| Domain | `src/domain/` | Pure logic: `Instrument` hierarchy, `InstrumentFactory`, `PositionManager`, `IPricingEngine` + `SimplePricingEngine` |
| Application | `src/application/` | `QuoteEngine` (fixed-spread quoting), `DeltaHedger` (threshold-based delta hedge) |
| Infrastructure | `src/infrastructure/` | `MarketDataAdapter` (CSV → events), `ProbabilisticTaker` (simulated counterparty, 30% trade probability) |
| Composition | `src/main.cpp` | **Only file where concrete class names appear**; wires all components via constructor injection |

**Event flow:**
```
MarketDataEvent
  → QuoteEngine → QuoteGeneratedEvent
  → ProbabilisticTaker → TradeExecutedEvent
  → PositionManager (update inventory)
  → DeltaHedger → OrderSubmittedEvent (when |Δ| > threshold)
```

**Key configurable parameters in `src/main.cpp`:**
- Quote half-spread: `$0.05` (total spread $0.10)
- Delta hedge threshold: `0.5`
- Probabilistic taker trade probability: `30%`
- RNG seed: `42` (reproducible simulation)

**Design patterns:** Observer (`EventBus`), Strategy (`IPricingEngine`, `IQuoteStrategy`, `IDeltaHedgeStrategy`), Factory (`InstrumentFactory`), Adapter (`MarketDataAdapter`), Command (`OrderSubmittedEvent`).

**Extensibility rule:** To add a new pricing model or hedging strategy, implement the relevant interface (`IPricingEngine`, etc.) and swap the concrete type only in `main.cpp`.

## Roadmap: Two Planned Applications

Defined in `Risk_Calibration.md`. Neither is implemented yet. Target folder: `src/applications/`.

### RealtimeRiskApp

Live portfolio monitoring and limit enforcement.

- **Subscribes to:** `TradeEvent`, `MarketDataEvent`, `OrderEvent`, `ParamUpdateEvent`
- **Publishes:** `RiskControlEvent` (BlockOrder, CancelOrders, ReduceOnly), `RiskAlertEvent`
- **Per-account state:** `PortfolioAggregate` → Positions, RealizedPnL, UnrealizedPnL, Greeks (Δ/Γ/ν/θ), VaR, IntradayDrawdown
- **Key interfaces to implement:** `IRiskPolicy::evaluate(AccountId, RiskMetrics) → vector<RiskControlEvent>`, `IApplication`
- **Design constraint:** Low-latency, deterministic; use snapshots for recovery

### BacktestCalibrationApp

Historical replay, strategy evaluation, and model parameter calibration.

- **Inputs:** Historical event replay (from event store), Strategy + PricingModel config, CalibrationObjective
- **Outputs:** `BacktestReport`, `ParamUpdateEvent` (written to parameter store)
- **Key interfaces:** `IModelParamSource::getParams(ModelId, Timestamp)`, `CalibrationEngine::observe()/solve()`
- **Calibration objective:** minimize `(price_model − price_market)²` or `(IV_model − IV_market)²`
- **Parameter feedback loop:** `CalibrationApp → ParamUpdateEvent → ParameterStore → PricingModel / RealtimeRiskApp`
- **Design constraint:** Deterministic and reproducible; use virtual clock for replay

### Shared rules for both applications (from `Risk_Calibration.md §3-4`)

- Separate state storage keys; no shared mutable state between apps
- Cross-app communication only through events
- Use `shared_ptr<const Event>` (immutable events)
- Version parameters with timestamps; shard portfolios by account for scaling
