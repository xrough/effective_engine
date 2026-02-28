# Risk Applications README

**Scope:** This document describes only the two applications inside an event-driven options platform:

1. **Realtime Risk Application** – live realized risk monitoring and control
2. **Backtest & Calibration Application** – historical replay, strategy evaluation, and parameter calibration

The goal is a design that translates directly into modern **OOP C++**.

---

# 1. Realtime Risk Application

## 1.1 Purpose

Provide **live monitoring and enforcement** of realized and exposure risk.

Responsibilities:

* Track portfolio state per account
* Update realized/unrealized PnL
* Compute Greeks exposure
* Enforce limits
* Trigger risk actions

This app must be low-latency and deterministic.

---

## 1.2 Inputs

Subscribed Events:

```text
TradeEvent
MarketDataEvent
OrderEvent
ParamUpdateEvent
```

---

## 1.3 Outputs

```text
RiskControlEvent
RiskAlertEvent
```

Examples:

* BlockOrder
* CancelOrders
* ReduceOnly
* Alert

---

## 1.4 Domain State

Per account:

```text
PortfolioAggregate
    ├── Positions
    ├── RealizedPnL
    ├── UnrealizedPnL
    ├── Greeks Exposure
    └── Risk Metrics Cache
```

---

## 1.5 Workflow

1. Receive TradeEvent → update positions
2. Receive MarketDataEvent → update mark price
3. Compute risk metrics
4. Evaluate risk policy
5. Emit risk control if needed

---

## 1.6 Pseudocode

```cpp
class RealtimeRiskApp : public IApplication
{
private:
    EventBus& bus;
    RiskEngine& riskEngine;
    RiskPolicy& policy;

    std::unordered_map<AccountId, PortfolioAggregate> portfolios;

public:
    void onTrade(const TradeEvent& e)
    {
        auto& pf = portfolios[e.account];
        pf.applyTrade(e);

        RiskMetrics m = computeMetrics(e.account);

        auto actions = policy.evaluate(e.account, m);

        for(auto& a : actions)
            bus.publish(a);
    }

    void onMarket(const MarketDataEvent& e)
    {
        for(auto acct : accountsHolding(e.instrument))
        {
            RiskMetrics m = computeMetrics(acct);
            auto actions = policy.evaluate(acct, m);

            for(auto& a : actions)
                bus.publish(a);
        }
    }
};
```

---

## 1.7 Risk Metrics Example

```text
RealizedPnL
UnrealizedPnL
Delta
Gamma
Vega
Theta
VaR
IntradayDrawdown
Exposure by Asset
```

---

## 1.8 Risk Policy Example

```cpp
class SimpleRiskPolicy : public IRiskPolicy
{
public:
    vector<RiskControlEvent> evaluate(AccountId acct, RiskMetrics m)
    {
        vector<RiskControlEvent> out;

        if(m.realizedPnL < -1e6)
            out.push_back(BlockOrders(acct));

        if(abs(m.delta) > 10000)
            out.push_back(ReduceOnly(acct));

        return out;
    }
};
```

---

# 2. Backtest & Calibration Application

## 2.1 Purpose

Run historical simulations to:

* Evaluate trading strategies
* Evaluate pricing models
* Calibrate model parameters
* Publish updated parameters

This app must be deterministic and reproducible.

---

## 2.2 Inputs

From Event Store Replay:

```text
Historical MarketDataEvent
Historical TradeEvent (optional)
```

Configuration:

```text
Strategy
Pricing Model
Initial Parameters
Execution Model
Calibration Objective
```

---

## 2.3 Outputs

```text
BacktestReport
ParamUpdateEvent
```

Parameters are written to parameter store and consumed by realtime pricing/risk.

---

## 2.4 Workflow

1. Replay historical events
2. Strategy generates signals
3. Simulated execution creates trades
4. Portfolio updates
5. Compare model vs market
6. Run calibration optimizer
7. Publish ParamUpdateEvent

---

## 2.5 Pseudocode

```cpp
class BacktestCalibrationApp : public IApplication
{
private:
    Strategy& strategy;
    PricingModel& model;
    CalibrationEngine& calibrator;
    PortfolioAggregate portfolio;

public:
    void onMarket(const MarketDataEvent& e)
    {
        auto signal = strategy.onTick(e, portfolio);

        auto trade = simulateExecution(signal);

        if(trade)
            portfolio.applyTrade(trade);

        auto pred = model.price(e.instrument);
        calibrator.observe(e, pred);
    }

    void finalize()
    {
        auto params = calibrator.solve();

        bus.publish( ParamUpdateEvent(model.id(), params) );
    }
};
```

---

## 2.6 Calibration Engine

### Goal

Minimize error between model and market.

Example loss:

```text
(price_model − price_market)^2
(IV_model − IV_market)^2
```

---

### Pseudocode

```cpp
class CalibrationEngine
{
private:
    vector<Observation> obs;

public:
    void observe(const Observation& o, const Prediction& p)
    {
        obs.push_back({o,p});
    }

    ParamMap solve()
    {
        return optimizer.optimize(lossFunction, initialParams);
    }
};
```

---

## 2.7 Parameter Feedback

Flow:

```text
CalibrationApp
    ↓
ParamUpdateEvent
    ↓
Parameter Store
    ↓
Pricing Model / Realtime Risk
```

Realtime app reloads parameters via:

```cpp
class IModelParamSource
{
public:
    virtual ParamMap getParams(ModelId id, Timestamp asOf) = 0;
};
```

---

# 3. Separation Between Applications

Even though both apps share infrastructure, they must be logically isolated.

Rules:

* Separate state storage keys
* Separate in-memory domain objects
* No shared mutable state
* Communication only through events

This allows independent deployment or scaling.

---

# 4. Key C++ Implementation Notes

* Use immutable events (`shared_ptr<const Event>`)
* Use snapshots for realtime recovery
* Use virtual clock for replay
* Version parameters with timestamps
* Shard portfolios by account for scaling

---

# 5. Minimal Folder Layout

```text
/applications
    realtime_risk_app.hpp
    realtime_risk_app.cpp
    backtest_calibration_app.hpp
    backtest_calibration_app.cpp
```

---

# 6. Summary

Realtime Risk Application ensures live safety of the trading system through incremental portfolio monitoring and limit enforcement.

Backtest & Calibration Application ensures long-term model quality through deterministic replay, strategy testing, and parameter optimization.

Together they provide a complete closed-loop risk and model management framework for an event-driven options trading platform.
