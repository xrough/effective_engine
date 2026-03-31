# Order of Instanization in the main.cpp

The order follows a strict dependency-first rule — nothing is created before the things it depends on. The key constraint driving this order: a subscriber must be registered before the event it handles is published. adapter.run() at step 7 is the trigger — everything from steps 1–6 is setup that must complete first, or events will fire with no one listening.

## Sell--side 
1. Instruments (options, underlying)
      ↓ no dependencies — pure data
2. main_bus
      ↓ no dependencies — empty message bus
3. ParameterStore (main_bus)
      ↓ needs main_bus to subscribe ParamUpdateEvent
      ↓ must exist BEFORE Phase 2 publishes calibration results
4. live_bus
      ↓ Phase 1's isolated bus, separate from main_bus
5. RoughVolPricingEngine
      ↓ no bus dependency — pure math
      ↓ must exist BEFORE SellerModule (which injects it into all components)
6. SellerModule::install(live_bus, cfg, options, rough_engine)
      ↓ internally creates in order:
        6a. PortfolioService   (needs bus, engine, options)
        6b. QuoteEngine        (needs bus, engine, options)
        6c. PositionManager + DeltaHedger  (needs bus, engine, options)
        6d. SellerRiskApp      (needs bus)
        6e. RiskControlEvent / RiskAlertEvent log subscribers
        6f. OrderRouter        (needs bus)
        6g. ProbabilisticTaker (needs bus)
      ↓ order within install() matters: subscribers must exist
        before the events they consume are published
7. MarketDataAdapter.run()
      ↓ starts publishing MarketDataEvents — everything must be wired first
      ↓ triggers the entire event cascade

── Phase 2 ──────────────────────────────────────────

8. backtest_bus  (isolated — no shared state with live_bus)
9. market_engine + model_engine + calibrator
10. BacktestCalibrationApp (backtest_bus, main_bus, ...)
      ↓ publishes to main_bus → ParameterStore (created at step 3)
11. backtest adapter.run()
12. backtest_app.finalize()  → ParamUpdateEvent → ParameterStore
13. rough_engine->update_params()  ← hot-inject calibrated params
