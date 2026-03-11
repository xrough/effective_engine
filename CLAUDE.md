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
| Domain | `src/domain/` | Pure logic: `Instrument` hierarchy, `InstrumentFactory`, `PositionManager`, `IPricingEngine` + `SimplePricingEngine` + `BlackScholesPricingEngine`, `CalibrationEngine` (golden-section σ search), `PortfolioAggregate`, `RiskMetrics`, `IRiskPolicy` + `SimpleRiskPolicy` |
| Application | `src/application/` | `QuoteEngine` (fixed-spread quoting), `DeltaHedger` (threshold-based delta hedge), `RealtimeRiskApp` (live risk monitoring + limit enforcement), `BacktestCalibrationApp` (historical replay + BS vol calibration) |
| Infrastructure | `src/infrastructure/` | `MarketDataAdapter` (CSV → events), `ProbabilisticTaker` (simulated counterparty, 30% trade probability), `ParameterStore` (versioned model parameters) |
| Bindings | `src/bindings/` | pybind11 bindings exposing C++ core as `omm_core` Python extension |
| Composition | `src/main.cpp` | **Only file where concrete class names appear**; wires all components via constructor injection |

**Event flow:**
```
Phase 1 – Live Simulation:
  MarketDataEvent
    → QuoteEngine → QuoteGeneratedEvent
    → ProbabilisticTaker → TradeExecutedEvent
      → PositionManager (update inventory)
      → DeltaHedger → OrderSubmittedEvent (when |Δ| > threshold)
      → RealtimeRiskApp → RiskControlEvent / RiskAlertEvent

Phase 2 – Backtest & Calibration (independent bus):
  MarketDataEvent (CSV replay)
    → BacktestCalibrationApp (cache S, option, market_price)
    → CalibrationEngine::solve() (golden-section search)
    → ParamUpdateEvent → ParameterStore (main bus)
```

**Key configurable parameters in `src/main.cpp`:**
- Quote half-spread: `$0.05` (total spread $0.10)
- Delta hedge threshold: `0.5`
- Probabilistic taker trade probability: `30%`
- RNG seed: `42` (reproducible simulation)
- Market BS vol (data): `0.25` | Model initial vol: `0.15` | Calibration range: `[0.01, 1.0]`
- Account ID: `"DESK_A"` | Model ID: `"bs_model"`

**Design patterns:** Observer (`EventBus`), Strategy (`IPricingEngine`, `IRiskPolicy`), Factory (`InstrumentFactory`), Adapter (`MarketDataAdapter`), Command (`OrderSubmittedEvent`).

**Extensibility rule:** To add a new pricing model or hedging strategy, implement the relevant interface (`IPricingEngine`, etc.) and swap the concrete type only in `main.cpp`.

## Implemented Applications

### RealtimeRiskApp (`src/application/RealtimeRiskApp.hpp`)

Live portfolio monitoring and limit enforcement.

- **Subscribes to:** `TradeExecutedEvent`, `MarketDataEvent`
- **Publishes:** `RiskControlEvent` (BlockOrder, ReduceOnly), `RiskAlertEvent`
- **Per-account state:** `PortfolioAggregate` → Positions, RealizedPnL, UnrealizedPnL, Greeks (Δ/Γ/ν/θ)
- **Policy (`SimpleRiskPolicy`):** BlockOrders on loss > $1M; ReduceOnly on |Δ| > 10000

### BacktestCalibrationApp (`src/application/BacktestCalibrationApp.hpp`)

Historical replay, strategy evaluation, and model parameter calibration.

- **Replays** CSV on an isolated bus → caches `(S, option, market_price)` tuples
- **Calibrates** BS vol via `CalibrationEngine::solve()` (golden-section search, minimizes MSE)
- **Publishes** `ParamUpdateEvent` → `ParameterStore` on main bus
- **Result:** Converges to σ = 0.25 exactly (0% error vs. market vol)

### Shared design rules

- Separate state storage keys; no shared mutable state between apps
- Cross-app communication only through events
- Use `shared_ptr<const Event>` (immutable events)
- Version parameters with timestamps

## Python Implementations

Three implementations exist side-by-side:

| Mode | Entry point | Description |
|---|---|---|
| Pure Python | `python/heritage/main.py` | Full DDD mirror archived in `python/heritage/` |
| Hybrid C++/Python | `python/hybrid_main.py` | Python app layer + C++ core (`omm_core.so` via pybind11) |

```bash
# Build Python bindings
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON && cmake --build build --target omm_core

# Run hybrid implementation
PYTHONPATH=build python3.13 python/hybrid_main.py

# Run pure Python
python3 python/heritage/main.py
```

All three implementations converge to vol = 0.2500 (0% calibration error).
