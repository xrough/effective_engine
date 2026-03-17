# Options Market Maker MVP — C++ Structure Reference

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    C++ PROJECT STRUCTURE — omm namespace                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LAYER 0 │ events/EventBus.hpp
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────────────────────────────────┐
  │  EventBus                                            │
  │  ─────────────────────────────────────────────────  │
  │  - handlers_: map<type_index, vector<function>>      │
  │  + subscribe<T>(handler: function<void(const T&)>)   │
  │  + publish<T>(event: const T&)                       │
  │  + clear()                                           │
  │                                                      │
  │  Pattern: Observer — type-erased via std::any        │
  │  Synchronous, single-threaded                        │
  └──────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LAYER 0 │ events/Events.hpp — Domain Events (pure data structs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MarketDataEvent          QuoteGeneratedEvent      TradeExecutedEvent
  ────────────────         ───────────────────      ──────────────────
  timestamp                instrument_id            instrument_id
  underlying_price         bid_price                side: enum Side
                           ask_price                  {Buy, Sell}
                           timestamp                price
                                                    quantity
                                                    timestamp

  OrderSubmittedEvent      RiskControlEvent         RiskAlertEvent
  ───────────────────      ────────────────         ──────────────
  instrument_id            account_id               account_id
  side: Side               action: RiskAction         metric_name
  quantity                   {BlockOrders,           value
  order_type: OrderType       CancelOrders,          limit
    {Market, Limit}           ReduceOnly}
                           reason

  ParamUpdateEvent
  ────────────────
  model_id
  params: map<string,double>
  updated_at

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LAYER 1 │ domain/ — Pure Business Logic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Instrument  (abstract)               InstrumentFactory  [Factory]
  ──────────────────────               ─────────────────────────────────────
  # id_: string                        + make_underlying(id) → shared_ptr<Underlying>
  + id() const                         + make_call(underlying_id, strike, expiry) → shared_ptr<Option>
  + type_name() = 0                    + make_put(underlying_id, strike, expiry)  → shared_ptr<Option>
       │                               - make_option_id(...)  [generates ID string]
       ├── Underlying  (final)
       │   type_name() = "underlying"
       │   delta = 1.0 (by convention)
       │
       └── Option  (final)
           underlying_id_: string
           strike_: double
           expiry_: time_point
           option_type_: OptionType {Call, Put}

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  IPricingEngine  (abstract)  [Strategy]
  ──────────────────────────────────────
  + price(option, underlying_price) → PriceResult{theo, delta} = 0
       │
       ├── SimplePricingEngine  (final)
       │     Call theo = max(0, S−K),  delta = +0.5
       │     Put  theo = max(0, K−S),  delta = −0.5
       │
       └── BlackScholesPricingEngine  (final)
             - vol_: double
             - r_:   double (default 0.05)
             + set_vol(v) / get_vol()
             Full BS formula with N(x) via std::erf

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  PositionManager                      RiskMetrics  (value object)
  ───────────────                      ─────────────────────────────
  - positions_: map<id, int>           realized_pnl
  + on_trade_executed(event)           unrealized_pnl
  + get_position(id) → int             delta, gamma, vega, theta
  + compute_portfolio_delta(deltas)    var_1d
  + print_positions()                  intraday_drawdown

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  PortfolioAggregate  [Aggregate Root]
  ─────────────────────────────────────────────────────────────────────
  - account_id_: string
  - positions_: map<id, int>
  - avg_cost_:  map<id, double>
  - options_:   vector<shared_ptr<Option>>
  - realized_pnl_, unrealized_pnl_, total_pnl_high_
  + applyTrade(TradeExecutedEvent)
  + markToMarket(IPricingEngine&, S)
  + computeMetrics(IPricingEngine&, S) → RiskMetrics
  + get_position(id) → int

  IRiskPolicy  (abstract)  [Strategy]      CalibrationEngine
  ────────────────────────────────────     ─────────────────────────────────
  + evaluate(account_id, RiskMetrics)      - observations_: vector<Observation>
      → vector<RiskControlEvent> = 0       + observe(market_price, model_price)
       │                                   + solve(lo, hi, loss_fn, tol) → double
       └── SimpleRiskPolicy  (final)       + mse() → double
             loss_limit_     = $1M         + observation_count() → int
             delta_limit_    = 10,000
             drawdown_limit_ = $500K       Observation {market_price, model_price}
             → BlockOrders if PnL < -limit
             → ReduceOnly  if |Δ| > limit
             → (alert only for drawdown)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LAYER 2 │ application/ — Orchestration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  IQuoteStrategy (abstract)              QuoteEngine  (final)  [Strategy]
  ─────────────────────────              ─────────────────────────────────────
  + on_market_data(event) = 0            Injects: bus, pricing_engine, options[]
                                         half_spread_ = $0.05
                                         + register_handlers()
                                         + on_market_data(MarketDataEvent)
                                           → calls pricing_engine.price(opt, S)
                                           → publishes QuoteGeneratedEvent

  DeltaHedger
  ─────────────────────────────────────────────────────────────────────────
  Injects: bus, position_manager, pricing_engine, options[], underlying_id
  delta_threshold_ = 0.5,  last_price_: double
  + register_handlers()
  - on_trade_executed(TradeExecutedEvent)
      1. position_manager.on_trade_executed()
      2. compute_delta_map(last_price) → map<id, Δ>
      3. position_manager.compute_portfolio_delta(deltas)
      4. if |Δ_portfolio| > threshold → publish OrderSubmittedEvent
  - update_market_price(price)   [called from MarketDataEvent handler]

  RealtimeRiskApp
  ─────────────────────────────────────────────────────────────────────────
  Injects: bus, pricing_engine, options[], underlying_id, account_id, risk_policy
  - portfolio_:   PortfolioAggregate  (private, per-account state)
  - last_price_:  double
  + register_handlers()
  - on_trade(TradeExecutedEvent)
      1. portfolio.applyTrade()
      2. portfolio.markToMarket(pricing_engine, last_price_)
      3. metrics = portfolio.computeMetrics(...)
      4. actions = risk_policy.evaluate(account_id, metrics)
      5. publish_risk_actions(actions)
  - on_market(MarketDataEvent)
      1. last_price_ = event.underlying_price
      2–5. same as on_trade

  BacktestCalibrationApp
  ─────────────────────────────────────────────────────────────────────────
  Injects: backtest_bus, main_bus, market_engine, model_engine,
           options[], calibrator, model_id
  - raw_observations_: vector<RawObs{S, option, market_price}>
  + register_handlers()   [on backtest_bus]
  - on_market(MarketDataEvent)
      → market_price = market_engine.price(opt, S).theo
      → raw_observations_.push_back(...)
  + finalize() → double
      → loss_fn(σ): sets model_engine.set_vol(σ), recomputes all prices, returns MSE
      → calibrator.solve(0.01, 1.0, loss_fn) → best_vol
      → publish ParamUpdateEvent on main_bus

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LAYER 3 │ infrastructure/ — I/O & External Adapters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MarketDataAdapter  [Adapter]         ProbabilisticTaker
  ────────────────────────────         ──────────────────────────────────────
  Injects: bus, csv_path               Injects: bus, trade_probability, seed
  - load_from_csv() / hardcoded_ticks  - rng_: mt19937, dist_: uniform[0,1)
  + run()                              + register_handlers()
    RawTick{ts_str, price}             - on_quote_generated(QuoteGeneratedEvent)
    → parse → MarketDataEvent            → 30% chance → publish TradeExecutedEvent
    → bus.publish<MarketDataEvent>         (random Buy or Sell)

  ParameterStore
  ─────────────────────────────────────────────────────────────────────────
  Injects: bus
  - history_: map<model_id, vector<VersionedParams{params, updated_at}>>
  + subscribe_handlers()
  - on_param_update(ParamUpdateEvent)  [appends new version]
  + get_params(model_id) → map<string, double>
  + print_all()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EVENT FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1 — LIVE SIMULATION  (single main_bus)

  MarketDataAdapter.run()
        │
        │ publish<MarketDataEvent>
        ▼
  ┌─────────────────────────────────────────────────┐
  │                  main_bus                       │
  └──┬──────────────────────────┬───────────────────┘
     │                          │
     ▼                          ▼
  QuoteEngine               DeltaHedger              RealtimeRiskApp
  .on_market_data()         .update_market_price()   .on_market()
     │                                                  │
     │ publish<QuoteGeneratedEvent>                     │ publish<RiskControlEvent>
     ▼                                                  │ publish<RiskAlertEvent>
  ProbabilisticTaker                                    │
  .on_quote_generated()  (30% hit rate)
     │
     │ publish<TradeExecutedEvent>
     ▼
  ┌─────────────────────────────────────────────────┐
  │                  main_bus                       │
  └──┬──────────────┬────────────────────────────────┘
     │              │                  │
     ▼              ▼                  ▼
  PositionMgr   DeltaHedger        RealtimeRiskApp
  .on_trade()   .on_trade()        .on_trade()
                   │                  │
                   │ (if |Δ|>0.5)     │ publish<RiskControlEvent>
                   ▼                  │ publish<RiskAlertEvent>
            publish<OrderSubmittedEvent>

PHASE 2 — BACKTEST & CALIBRATION  (isolated backtest_bus + main_bus)

  MarketDataAdapter.run()  [on backtest_bus]
        │
        │ publish<MarketDataEvent>
        ▼
  BacktestCalibrationApp.on_market()
  [caches RawObs{S, option, market_price} for each tick]
        │
        │ .finalize() called after replay
        ▼
  CalibrationEngine.solve(lo=0.01, hi=1.0, loss_fn)
  [golden-section search, ~50 iterations, converges σ → 0.25]
        │
        │ publish<ParamUpdateEvent>  [on main_bus]
        ▼
  ParameterStore.on_param_update()
  [stores versioned params: {"vol": 0.25}]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DESIGN PATTERNS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Observer   │ EventBus  (type-erased publish/subscribe)
  Strategy   │ IPricingEngine → Simple / BlackScholes
             │ IRiskPolicy   → SimpleRiskPolicy
             │ IQuoteStrategy → QuoteEngine
  Factory    │ InstrumentFactory (static, stateless)
  Adapter    │ MarketDataAdapter (CSV/hardcoded → MarketDataEvent)
  Command    │ OrderSubmittedEvent (intent encapsulated as data)
  Aggregate  │ PortfolioAggregate (consistency boundary per account)
```

---

## Project Layout

```
MVP/
├── src/                         # C++ platform (sole entry point)
│   ├── main.cpp
│   ├── events/                  EventBus · domain events
│   ├── domain/                  Instrument · PricingEngine · CalibrationEngine · RiskPolicy · PortfolioAggregate
│   ├── application/             QuoteEngine · DeltaHedger · RealtimeRiskApp · BacktestCalibrationApp
│   └── infrastructure/          MarketDataAdapter · ProbabilisticTaker · ParameterStore
│                                ModelServiceClient (gRPC, compiled only with BUILD_GRPC_CLIENT=ON)
│
├── lab/                         # Python research experiments (no platform logic)
│   ├── experiments/
│   │   └── real_data_experiment.py   fetch → calibrate → visualize
│   ├── grpc_client/
│   │   └── rough_pricing_client.py   gRPC client for RoughPricingService
│   └── data/
│       └── yfinance_fetcher.py        Yahoo Finance adapter
│
├── proto/
│   └── rough_pricing.proto      gRPC service contract (single source of truth)
├── generated/python/            Python stubs (grpcio-tools output, re-generate with `make proto-python`)
├── CMakeLists.txt
└── Makefile                     proto-cpp · proto-python · build · build-grpc
```

## Build & Run

### Platform (C++)

```bash
# Configure and build
cmake -S . -B build && cmake --build build --parallel

# Run the simulation (converges to vol = 0.2500)
./build/market_maker
```

### Lab Experiments (Python)

The experiment scripts call the Rough-Pricing gRPC service
for model calibration (BS, GBM-MC, Heston, and future rough-vol models).

```bash
# Terminal 1 — start the model service
cd ~/rough_pricing_env/Rough-Pricing
python3 -m roughvol.service.server

# Terminal 2 — run the experiment
cd MVP/lab
pip install yfinance scipy pandas matplotlib grpcio
MODEL_SERVICE_ADDR=localhost:50051 \
  python3 experiments/real_data_experiment.py --ticker AAPL --start 2024-01-01 --end 2024-06-30
```

Pass `--skip-heston` to skip the slow 5D optimisation (quick smoke test).

### What the experiment does

Fetches real spot history and live options chain from Yahoo Finance, then benchmarks three calibration models against real option prices:

| Phase | Description |
|-------|-------------|
| **Phase 0** | Download spot history + options chain (yfinance → CSV) |
| **Phase 1** | Calibrate BS (closed-form, ~0 ms), GBM-MC (1D, ~10–30 s), Heston (5D, ~60–300 s) via gRPC |
| **Phase 2** | Save four diagnostic plots to `data/plots/` |

### Output plots

| File | Content |
|------|---------|
| `data/plots/spot_<TICKER>.png` | Daily closing price over the fetched date range |
| `data/plots/vol_smile_<TICKER>.png` | BS implied vol vs moneyness (K/S), one line per expiry |
| `data/plots/calib_mse_<TICKER>.png` | Horizontal bar chart comparing MSE: BS / GBM-MC / Heston |
| `data/plots/price_fit_<TICKER>.png` | Scatter of model price vs market price; y = x line = perfect fit |

Calibration summary printed to stdout:

```
══════════════ Calibration Results ══════════════
  Ticker : AAPL
  Spot   : $182.34
  Options: 12 contracts (2 expiries)

Model    │ Parameters                              │ MSE        │ Time (s)
─────────┼─────────────────────────────────────────┼────────────┼─────────
BS       │ σ_ATM=0.2347                            │ 2.31e-04   │ 0.001
GBM MC   │ σ=0.2341                                │ 2.89e-04   │ 18.4
Heston   │ κ=2.10  θ=0.055  ξ=0.31  ρ=-0.68  v0=0.049 │ 8.12e-05 │ 124.7
═════════════════════════════════════════════════
```
