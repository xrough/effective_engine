#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"

// ============================================================
// File: SimpleExecSim.hpp  (demo/cpp/)
// Role: simulation adapter — converts OrderSubmittedEvent into
//       FillEvent at the current market price. 
//       SimpleExecSim is a demo infrastructure shim — it simulates the role of a real Order Management System (OMS) or broker.
// StrategyController仅仅表达交易的意愿，但由于现实中OMS的存在，订单并不一定会以理想价格成交，甚至可能无法成交。SimpleExecSim负责处理非理想的情况。The "market reality" it applies is just bid/ask spread — buys fill at ask, sells fill at bid, instead of the ideal mid.
//
// Fill price logic:
//   - Underlying orders (e.g. delta hedge): fill at last spot price.
//   - Option orders (e.g. "ATM_CALL", "ATM_PUT"): fill at the option's
//     ask price for buys, bid price for sells (from OptionMidQuoteEvent).
//     Falls back to mid price if bid/ask are not populated.
//
// This is demo infrastructure only. DeltaHedger and AlphaPnLTracker
// have no knowledge of this class — they only consume the FillEvents
// it emits, which carry standard FillEvent semantics.
//
// Replacing this with a real OMS requires no changes to any engine
// component.
// ============================================================

namespace omm::demo {

class SimpleExecSim {
public:
    explicit SimpleExecSim(
        std::shared_ptr<events::EventBus> bus,
        double initial_price = 150.0
    ) : bus_(std::move(bus)), last_price_(initial_price) {} // initialization list after :

    void register_handlers() { // recall that register_handlers() is the subscription registration step — it tells the EventBus which events this component wants to listen to, and what to do when each arrives.
        // 跟踪最新标的价格
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { // lambda captures 'this' pointer to access member variables
                last_price_ = e.underlying_price;
            }
        );
        // 记录每个期权品种的最新行情（买/卖/中间价）
        // Buying at ask and selling at bid is the standard execution cost assumption.
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) {
                last_mid_[e.instrument_id] = e.mid_price;
                last_bid_[e.instrument_id] = e.bid_price;
                last_ask_[e.instrument_id] = e.ask_price;
            }
        );
        // 订单 → 立即成交
        bus_->subscribe<events::OrderSubmittedEvent>(
            [this](const events::OrderSubmittedEvent& o) { on_order(o); }
        );
    }

private:
    void on_order(const events::OrderSubmittedEvent& o) {//an order has already been submitted, what price should it fill at?
        events::FillEvent fill;
        fill.instrument_id = o.instrument_id;
        fill.side          = o.side;
        fill.fill_qty      = o.quantity;
        fill.producer      = "alpha_exec";
        fill.timestamp     = std::chrono::system_clock::now();

        // 期权成交：按买入/卖出方向使用买卖价，降低模拟过于乐观的问题
        auto it_mid = last_mid_.find(o.instrument_id);
        if (it_mid != last_mid_.end()) { // if the key was found in the map.
            // 这是期权订单：买入时用卖出价（ask），卖出时用买入价（bid）
            if (o.side == events::Side::Buy) {
                auto it = last_ask_.find(o.instrument_id);
                fill.fill_price = (it != last_ask_.end() && it->second > 0.0)
                                  ? it->second : it_mid->second;
            } else {
                auto it = last_bid_.find(o.instrument_id);
                fill.fill_price = (it != last_bid_.end() && it->second > 0.0)
                                  ? it->second : it_mid->second;
            }
        } else {
            // 标的资产（delta 对冲）：使用最新现货价格
            fill.fill_price = last_price_;
        }

        bus_->publish(fill);
    }

    std::shared_ptr<events::EventBus>       bus_;
    double                                   last_price_;
    std::unordered_map<std::string, double>  last_mid_;
    std::unordered_map<std::string, double>  last_bid_;
    std::unordered_map<std::string, double>  last_ask_;
};

} // namespace omm::demo
