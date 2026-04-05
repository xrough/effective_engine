#pragma once
#include <memory>
#include <string>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"

// ============================================================
// File: SimpleExecSim.hpp  (demo/cpp/)
// Role: simulation adapter — converts OrderSubmittedEvent into
//       FillEvent at the current market price.
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
    ) : bus_(std::move(bus)), last_price_(initial_price) {}

    void register_handlers() {
        // 跟踪最新标的价格
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) {
                last_price_ = e.underlying_price;
            }
        );
        // 订单 → 立即成交
        bus_->subscribe<events::OrderSubmittedEvent>(
            [this](const events::OrderSubmittedEvent& o) { on_order(o); }
        );
    }

private:
    void on_order(const events::OrderSubmittedEvent& o) {
        events::FillEvent fill;
        fill.instrument_id = o.instrument_id;
        fill.side          = o.side;
        fill.fill_price    = last_price_;
        fill.fill_qty      = o.quantity;
        fill.producer      = "alpha_exec";
        fill.timestamp     = std::chrono::system_clock::now();
        bus_->publish(fill);
    }

    std::shared_ptr<events::EventBus> bus_;
    double                            last_price_;
};

} // namespace omm::demo
