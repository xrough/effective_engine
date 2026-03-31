# Options Market Maker MVP

A C++ simulation MVP of an options market-making desk using a layered DDD architecture and event-driven design. The project is completed with the assitance of Claude Code while I am slowly learning C++.

## Goals

- Quotes bid/ask spreads on options using Black-Scholes or a simple intrinsic model
- Simulates a probabilistic counterparty generating trades
- Hedging
- Monitors risk in real time and enforces loss/delta limits
- Backtests on historical data and calibrates implied vol.

## Key files

- [src/main.cpp](src/main.cpp) — wiring: the only place concrete types are named
- [src/events/EventBus.hpp](src/events/EventBus.hpp) — type-erased pub/sub dispatcher
- [src/application/RealtimeRiskApp.hpp](src/application/RealtimeRiskApp.hpp) — live risk monitoring
- [src/application/BacktestCalibrationApp.hpp](src/application/BacktestCalibrationApp.hpp) — historical replay + vol calibration
## Quick start

```bash
cmake -S . -B build && cmake --build build --parallel
./build/market_maker
```
