#include "OrderRouter.hpp"
#include <iostream>
#include <iomanip>

// ============================================================
// OrderRouter.cpp — skeleton implementation
//
// Current behavior: logs the order and calls send_to_exchange()
// stub. No FillEvent is published yet — DeltaHedger still
// self-fills directly (see DeltaHedger.cpp).
//
// Migration path to full implementation:
//   Step 1 (here): seam established, lambda in main.cpp replaced
//   Step 2: send_to_exchange() sends real protocol message
//   Step 3: async fill callback publishes FillEvent onto bus_
//   Step 4: DeltaHedger stops self-filling; relies on this event
// ============================================================

namespace omm::infrastructure {

OrderRouter::OrderRouter(std::shared_ptr<events::EventBus> bus)
    : bus_(std::move(bus)) {}

void OrderRouter::register_handlers() {
    bus_->subscribe<events::OrderSubmittedEvent>(
        [this](const events::OrderSubmittedEvent& evt) {
            this->on_order(evt);
        }
    );
}

void OrderRouter::on_order(const events::OrderSubmittedEvent& evt) {
    std::cout << "[OrderRouter] received order:"
              << "  " << (evt.side == events::Side::Buy ? "BUY" : "SELL")
              << "  qty=" << evt.quantity
              << "  instrument=" << evt.instrument_id
              << "  type=" << (evt.order_type == events::OrderType::Market
                               ? "Market" : "Limit")
              << "\n";

    // Hand off to execution stub
    // In a real system: serialize to FIX/WebSocket, send to venue
    send_to_exchange(evt);
}

void OrderRouter::send_to_exchange(const events::OrderSubmittedEvent& /*evt*/) {
    // ── STUB ─────────────────────────────────────────────────
    // TODO: translate evt → external protocol message and send
    //
    // On receiving ExecutionReport (fill confirmation):
    //   events::FillEvent fill{
    //       evt.instrument_id,
    //       evt.side,
    //       fill_price,   // from exchange report
    //       filled_qty,   // may be < evt.quantity (partial fill)
    //       "broker",
    //       now()
    //   };
    //   bus_->publish(fill);
    // ─────────────────────────────────────────────────────────
}

} // namespace omm::infrastructure
