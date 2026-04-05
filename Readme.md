# Effective Engine — MVP

C++ options trading engine. Event-driven, layered DDD architecture. Covers both sell-side market making and buy-side variance alpha, with a deep BSDE neural hedging demo. Built while learning C++, with assistance from Claude Code.

## Quick start

```bash
cmake -S . -B build && cmake --build build --parallel
./build/market_maker
```

## What it does

**Phase 1 — Live simulation**

Quotes bid/ask spreads (Rough Bergomi skew or Black-Scholes), simulates a probabilistic counterparty (30% fill), runs threshold-based delta hedging, and enforces live risk limits (max loss $1M, max delta 10,000).

**Phase 2 — Backtest and calibration**

Replays the same market data on an isolated bus, calibrates implied volatility via golden-section search, and hot-injects the result back into the live pricing engine.

**Variance Alpha pipeline (BuyerModule)**

Extracts ATM implied variance from option mid quotes via BS IV bisection, computes a rolling z-score against the rough-model forward variance forecast (`xi0 * T`), and runs a Flat → Live → Cooldown state machine to trade ATM front straddles when the spread is statistically significant.

**Deep BSDE hedging demo ([demo/](demo/Readme.md))**

Generates lifted rough Heston paths in C++, trains a shared-weight MLP offline in Python to solve the BSDE hedging problem, exports to ONNX, and benchmarks in-process inference against analytic BS delta and finite-difference delta:

| Hedger | PnL std | CVaR(95%) | Latency p50 |
|---|---|---|---|
| BS delta (analytic) | 4.24 | 22.57 | — |
| BS delta (FD bump) | 3.65 | 20.35 | — |
| Neural BSDE | **3.40** | **19.35** | 3.4 µs |

## Architecture

Four-layer DDD in `src/core/` (events → domain → analytics → application/infrastructure), with two isolated module composition entry points in `src/modules/`:

- `src/modules/buyer/` — variance alpha: implied variance extraction, rolling z-score signal, Flat/Live/Cooldown strategy controller. Own `EventBus` and `RoughVolPricingEngine` instance.
- `src/modules/seller/` — market making: quoting, delta hedging, risk control, backtest calibration. Own `EventBus` and `RoughVolPricingEngine` instance.

The two modules share no direct dependencies. Each exposes a static `install()` that wires all internal components and returns a context struct. Cross-cutting concerns (calibration parameter feedback) flow only through the shared `main_bus` via `ParamUpdateEvent → ParameterStore`.

`src/main.cpp` is the sole composition root — the only file where concrete class names appear. All intra-module communication goes through that module's `EventBus` (synchronous pub/sub, type-erased via `std::type_index` + `std::any`).

**Extensibility rule:** to add a new pricing model or risk policy, implement `IPricingEngine` / `IRiskPolicy` and swap the concrete type only in `main.cpp`.

## Key interfaces

- [src/core/events/EventBus.hpp](src/core/events/EventBus.hpp) — type-erased pub/sub dispatcher
- [src/core/analytics/PricingEngine.hpp](src/core/analytics/PricingEngine.hpp) — `IPricingEngine` + BS / intrinsic implementations
- [src/core/analytics/RoughVolPricingEngine.hpp](src/core/analytics/RoughVolPricingEngine.hpp) — Rough Bergomi skew with hot parameter injection
- [src/core/interfaces/IAlphaSignal.hpp](src/core/interfaces/IAlphaSignal.hpp) — buyer-side signal interface
- [src/core/interfaces/IEntryPolicy.hpp](src/core/interfaces/IEntryPolicy.hpp) — order decision interface
- [src/modules/seller/SellerModule.hpp](src/modules/seller/SellerModule.hpp) — seller composition entry point
- [src/modules/buyer/BuyerModule.hpp](src/modules/buyer/BuyerModule.hpp) — buyer composition entry point
