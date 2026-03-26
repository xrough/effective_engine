# Options Market Maker MVP

A C++ simulation MVP of an options market-making desk using a layered DDD architecture and event-driven design. A Python lab connects to an external gRPC pricing service for real-data calibration experiments. The project is completed with the assitance of Claude Code while I am slowly learning C++.

## Goals

- Quotes bid/ask spreads on options using Black-Scholes or a simple intrinsic model
- Simulates a probabilistic counterparty (30% hit rate) generating trades
- Delta-hedges the portfolio when exposure exceeds a threshold
- Monitors risk in real time and enforces loss/delta limits
- Backtests on historical data and calibrates implied vol via golden-section search
- Benchmarks BS, GBM-MC, and Heston models against real Yahoo Finance options data (Python lab)

## Key files

- [src/main.cpp](src/main.cpp) — wiring: the only place concrete types are named
- [src/events/EventBus.hpp](src/events/EventBus.hpp) — type-erased pub/sub dispatcher
- [src/application/RealtimeRiskApp.hpp](src/application/RealtimeRiskApp.hpp) — live risk monitoring
- [src/application/BacktestCalibrationApp.hpp](src/application/BacktestCalibrationApp.hpp) — historical replay + vol calibration
- [lab/experiments/real_data_experiment.py](lab/experiments/real_data_experiment.py) — fetch → calibrate → visualize

## Quick start

```bash
# Build and run the C++ simulation
cmake -S . -B build && cmake --build build --parallel
./build/market_maker

# Run the Python lab experiment (requires Rough-Pricing gRPC service)
MODEL_SERVICE_ADDR=localhost:50051 \
  python3 lab/experiments/real_data_experiment.py --ticker AAPL --start 2024-01-01 --end 2024-06-30
```

For full class diagrams, event flows, and build details see [structure.md](structure.md) (local only).
