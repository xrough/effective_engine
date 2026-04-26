#include "OrderRouter.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================
// OrderRouter.cpp — execution boundary
//
// In this MVP, OrderRouter runs in deterministic simulation mode. It accepts
// OrderSubmittedEvent commands, applies simple execution realism, and emits
// FillEvent facts. Replacing send_to_exchange() with FIX/WebSocket I/O later
// does not require callers to self-fill.
// ============================================================

namespace omm::infrastructure {

OrderRouter::OrderRouter(
    std::shared_ptr<events::EventBus> bus,
    OrderRouterConfig config)
    : bus_(std::move(bus))
    , config_(config) {}

void OrderRouter::register_handlers() {
    bus_->subscribe<events::MarketDataEvent>(
        [this](const events::MarketDataEvent& evt) {
            this->on_market_data(evt);
        }
    );
    bus_->subscribe<events::OrderSubmittedEvent>(
        [this](events::OrderSubmittedEvent evt) {
            this->on_order(evt);
        }
    );
}

void OrderRouter::flush_all() {
    flush_ready_orders(/*force_ready=*/true);
}

void OrderRouter::on_market_data(const events::MarketDataEvent& evt) {
    ++event_seq_;
    last_price_ = evt.underlying_price;
    flush_ready_orders();
}

void OrderRouter::on_order(events::OrderSubmittedEvent evt) {
    if (evt.quantity <= 0) {
        return;
    }
    if (evt.order_id.empty()) {
        evt.order_id = "ROUTER-" + std::to_string(++order_seq_);
    }
    if (evt.producer.empty()) {
        evt.producer = "broker";
    }
    if (evt.timestamp == events::Timestamp{}) {
        evt.timestamp = std::chrono::system_clock::now();
    }

    if (config_.verbose) {
        std::cout << "[OrderRouter] accepted order " << evt.order_id
                  << " producer=" << evt.producer
                  << "  " << (evt.side == events::Side::Buy ? "BUY" : "SELL")
                  << " qty=" << evt.quantity
                  << " instrument=" << evt.instrument_id
                  << " type=" << (evt.order_type == events::OrderType::Market
                                  ? "Market" : "Limit")
                  << "\n";
    }

    send_to_exchange(evt);

    PendingOrder pending;
    pending.order = std::move(evt);
    pending.remaining_qty = pending.order.quantity;
    pending.ready_after_seq = event_seq_ + static_cast<std::size_t>(
        std::max(config_.latency_events, 0)
    );
    pending_.push_back(std::move(pending));

    flush_ready_orders();
}

void OrderRouter::send_to_exchange(const events::OrderSubmittedEvent& /*evt*/) {
    // External I/O boundary. Simulation fills are generated locally by
    // flush_ready_orders(); production code would send the protocol message
    // here and publish FillEvent when an execution report is received.
}

void OrderRouter::flush_ready_orders(bool force_ready) {
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
        if (!compute_fill_price(pending.order, fill_price) ||
            !is_marketable(pending.order, fill_price)) {
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
            std::cout << "[OrderRouter] fill order=" << fill.order_id
                      << " producer=" << fill.producer
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

bool OrderRouter::compute_fill_price(
    const events::OrderSubmittedEvent& evt,
    double& out_price
) const {
    out_price = (evt.reference_price > 0.0) ? evt.reference_price : last_price_;
    const double half_spread = out_price * config_.half_spread_bps / 10000.0;
    out_price += (evt.side == events::Side::Buy) ? half_spread : -half_spread;
    return out_price > 0.0;
}

bool OrderRouter::is_marketable(
    const events::OrderSubmittedEvent& evt,
    double fill_price
) {
    if (evt.order_type == events::OrderType::Market || evt.limit_price <= 0.0) {
        return true;
    }
    if (evt.side == events::Side::Buy) {
        return fill_price <= evt.limit_price;
    }
    return fill_price >= evt.limit_price;
}

} // namespace omm::infrastructure
