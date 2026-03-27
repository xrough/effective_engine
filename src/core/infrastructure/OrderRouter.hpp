#pragma once
#include <memory>
#include <string>
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
// Current state: SKELETON
//   send_to_exchange() is a no-op stub. The seam is in place but
//   no real network I/O is performed. DeltaHedger still self-fills
//   directly until this is wired to a real execution venue.
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

class OrderRouter {
public:
    explicit OrderRouter(std::shared_ptr<events::EventBus> bus);

    // register_handlers() — subscribe to OrderSubmittedEvent
    void register_handlers();

private:
    // on_order() — receives internal order command
    //   Current: logs and calls send_to_exchange() stub
    //   Future:  also tracks order state (pending → acked → filled)
    void on_order(const events::OrderSubmittedEvent& evt);

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
};

} // namespace omm::infrastructure
