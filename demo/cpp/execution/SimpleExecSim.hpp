#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
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

struct SimpleExecSimConfig {
    // <= 0 means no cap; otherwise large orders are split across fills.
    int    max_fill_qty = 0;
    // Optional extra friction on top of bid/ask for option orders.
    double option_slippage_bps = 0.0;
    // Used for underlying orders where no bid/ask book is available.
    double underlying_half_spread_bps = 0.5;
    // Number of market/quote events before an accepted order becomes fillable.
    int    latency_events = 0;
    bool   verbose = false;
};

class SimpleExecSim {
public:
    explicit SimpleExecSim(
        std::shared_ptr<events::EventBus> bus,
        double initial_price = 150.0,
        SimpleExecSimConfig config = {}
    ) : bus_(std::move(bus))
      , last_price_(initial_price)
      , config_(config) {} // initialization list after :

    void register_handlers() { // recall that register_handlers() is the subscription registration step — it tells the EventBus which events this component wants to listen to, and what to do when each arrives.
        // 跟踪最新标的价格
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { // lambda captures 'this' pointer to access member variables
                ++event_seq_;
                last_price_ = e.underlying_price;
                flush_ready_orders();
            }
        );
        // 记录每个期权品种的最新行情（买/卖/中间价）
        // Buying at ask and selling at bid is the standard execution cost assumption.
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) {
                ++event_seq_;
                last_mid_[e.instrument_id] = e.mid_price;
                last_bid_[e.instrument_id] = e.bid_price;
                last_ask_[e.instrument_id] = e.ask_price;
                flush_ready_orders();
            }
        );
        // 订单 → execution queue → fill(s)
        bus_->subscribe<events::OrderSubmittedEvent>(
            [this](const events::OrderSubmittedEvent& o) { on_order(o); }
        );
    }

    // Useful at the end of a deterministic replay to fill marketable residuals.
    void flush_all() {
        flush_ready_orders(/*force_ready=*/true);
    }

private:
    struct PendingOrder {
        events::OrderSubmittedEvent order;
        int remaining_qty = 0;
        std::size_t ready_after_seq = 0;
    };

    void on_order(events::OrderSubmittedEvent o) {//an order has already been submitted, what price should it fill at?
        if (o.quantity <= 0) {
            return;
        }
        if (o.order_id.empty()) {
            o.order_id = "SIM-" + std::to_string(++order_seq_);
        }
        if (o.producer.empty()) {
            o.producer = "alpha_exec";
        }
        if (o.timestamp == events::Timestamp{}) {
            o.timestamp = std::chrono::system_clock::now();
        }

        PendingOrder pending;
        pending.order = std::move(o);
        pending.remaining_qty = pending.order.quantity;
        pending.ready_after_seq = event_seq_ + static_cast<std::size_t>(
            std::max(config_.latency_events, 0)
        );
        pending_.push_back(std::move(pending));

        if (config_.verbose) {
            const auto& order = pending_.back().order;
            std::cout << "[SimpleExecSim] accepted order " << order.order_id
                      << " producer=" << order.producer
                      << " instrument=" << order.instrument_id
                      << " qty=" << order.quantity << "\n";
        }

        flush_ready_orders();
    }

    void flush_ready_orders(bool force_ready = false) {
        if (publishing_fills_) {
            return;
        }

        std::vector<events::FillEvent> fills;
        std::deque<PendingOrder> still_pending;

        while (!pending_.empty()) {
            PendingOrder pending = std::move(pending_.front());
            pending_.pop_front();

            if (!force_ready && event_seq_ < pending.ready_after_seq) {
                still_pending.push_back(std::move(pending));
                continue;
            }

            double fill_price = 0.0;
            if (!compute_fill_price(pending.order, fill_price)) {
                still_pending.push_back(std::move(pending));
                continue;
            }

            if (!is_marketable(pending.order, fill_price)) {
                still_pending.push_back(std::move(pending));
                continue;
            }

            int fill_qty = pending.remaining_qty;
            if (config_.max_fill_qty > 0) {
                fill_qty = std::min(fill_qty, config_.max_fill_qty);
            }
            if (fill_qty <= 0) {
                continue;
            }

            pending.remaining_qty -= fill_qty;

            events::FillEvent fill{
                pending.order.instrument_id,
                pending.order.side,
                fill_price,
                fill_qty,
                pending.order.producer,
                std::chrono::system_clock::now()
            };
            fill.order_id = pending.order.order_id;
            fill.requested_qty = pending.order.quantity;
            fill.remaining_qty = pending.remaining_qty;
            fill.is_partial = pending.remaining_qty > 0;
            fill.reference_price = pending.order.reference_price;
            fills.push_back(fill);

            if (pending.remaining_qty > 0) {
                still_pending.push_back(std::move(pending));
            }
        }

        pending_ = std::move(still_pending);

        publishing_fills_ = true;
        for (const auto& fill : fills) {
            if (config_.verbose) {
                std::cout << "[SimpleExecSim] fill order=" << fill.order_id
                          << " instrument=" << fill.instrument_id
                          << " qty=" << fill.fill_qty
                          << " price=" << std::fixed << std::setprecision(4)
                          << fill.fill_price
                          << (fill.is_partial ? " partial" : "") << "\n";
            }
            bus_->publish(fill);
        }
        publishing_fills_ = false;
    }

    bool compute_fill_price(
        const events::OrderSubmittedEvent& o,
        double& out_price
    ) const {
        auto it_mid = last_mid_.find(o.instrument_id);
        if (it_mid != last_mid_.end()) { // if the key was found in the map.
            // 这是期权订单：买入时用卖出价（ask），卖出时用买入价（bid）
            if (o.side == events::Side::Buy) {
                auto it = last_ask_.find(o.instrument_id);
                out_price = (it != last_ask_.end() && it->second > 0.0)
                            ? it->second : it_mid->second;
                out_price += out_price * config_.option_slippage_bps / 10000.0;
            } else {
                auto it = last_bid_.find(o.instrument_id);
                out_price = (it != last_bid_.end() && it->second > 0.0)
                            ? it->second : it_mid->second;
                out_price -= out_price * config_.option_slippage_bps / 10000.0;
            }
            return out_price > 0.0;
        }

        // 标的资产（delta 对冲）：使用订单携带的参考价，否则使用最新现货价格
        out_price = (o.reference_price > 0.0) ? o.reference_price : last_price_;
        double half_spread = out_price * config_.underlying_half_spread_bps / 10000.0;
        out_price += (o.side == events::Side::Buy) ? half_spread : -half_spread;
        return out_price > 0.0;
    }

    static bool is_marketable(
        const events::OrderSubmittedEvent& o,
        double fill_price
    ) {
        if (o.order_type == events::OrderType::Market || o.limit_price <= 0.0) {
            return true;
        }
        if (o.side == events::Side::Buy) {
            return fill_price <= o.limit_price;
        }
        return fill_price >= o.limit_price;
    }

    std::shared_ptr<events::EventBus>       bus_;
    double                                   last_price_;
    SimpleExecSimConfig                      config_;
    std::deque<PendingOrder>                 pending_;
    std::unordered_map<std::string, double>  last_mid_;
    std::unordered_map<std::string, double>  last_bid_;
    std::unordered_map<std::string, double>  last_ask_;
    std::size_t                              event_seq_ = 0;
    std::size_t                              order_seq_ = 0;
    bool                                     publishing_fills_ = false;
};

} // namespace omm::demo
