#pragma once
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"

// ============================================================
// File: OrderRouter.hpp
// Role: Infrastructure boundary between internal order commands
//       and the external execution world (exchange / broker).
//
// Responsibility:
//   - Subscribes to OrderSubmittedEvent (internal command)
//   - Translates to external protocol (FIX, WebSocket, REST) [stub]
//   - Receives exchange acknowledgement / fill confirmation
//   - Publishes FillEvent back onto the bus (our perspective)
//
// Current state:
//   The production protocol boundary is still a stub, but simulation mode
//   now models an order lifecycle: accept order, apply spread/slippage,
//   optionally split large orders, and publish FillEvent with provenance.
//
// When fully implemented:
//   OrderSubmittedEvent → [FIX/WebSocket] → Exchange
//                                               ↓ (async, can fail)
//   FillEvent ← [protocol adapter] ← Exchange fill report
//
// Anti-corruption layer pattern:
//   Inbound:  MarketDataAdapter  (feed protocol  → MarketDataEvent)
//   Outbound: OrderRouter        (OrderSubmittedEvent → exchange protocol)
// ============================================================

namespace omm::infrastructure {

struct OrderRouterConfig {
    int    max_fill_qty = 0;          // <= 0 means no cap
    double half_spread_bps = 0.5;     // used for underlying market orders
    int    latency_events = 0;        // market-data events before fillable
    bool   verbose = true;
};

class OrderRouter {
public:
    explicit OrderRouter(
        std::shared_ptr<events::EventBus> bus,
        OrderRouterConfig config = {}
    );

    // register_handlers() — subscribe to OrderSubmittedEvent
    void register_handlers();

    void flush_all();

private:
    struct PendingOrder {
        events::OrderSubmittedEvent order;
        int remaining_qty = 0;
        std::size_t ready_after_seq = 0;
    };

    // on_order() — receives internal order command
    void on_order(events::OrderSubmittedEvent evt);

    void on_market_data(const events::MarketDataEvent& evt);
    void flush_ready_orders(bool force_ready = false);
    bool compute_fill_price(
        const events::OrderSubmittedEvent& evt,
        double& out_price
    ) const;
    static bool is_marketable(
        const events::OrderSubmittedEvent& evt,
        double fill_price
    );

    // send_to_exchange() — STUB: external I/O boundary
    //   Replace with real FIX/WebSocket/REST call.
    //   On fill confirmation: publish FillEvent onto bus_.
    //
    // Example future implementation:
    //   auto fix_msg = to_fix_new_order(evt);
    //   venue_session_->send(fix_msg);
    //   // FillEvent published asynchronously when ExecutionReport arrives
    void send_to_exchange(const events::OrderSubmittedEvent& evt);

    std::shared_ptr<events::EventBus> bus_;
    OrderRouterConfig                 config_;
    std::deque<PendingOrder>          pending_;
    double                            last_price_ = 150.0;
    std::size_t                       event_seq_ = 0;
    std::size_t                       order_seq_ = 0;
    bool                              publishing_fills_ = false;
};

} // namespace omm::infrastructure
