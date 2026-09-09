#include <iostream>
#include <memory>
#include <vector>

#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "execution/SimpleExecSim.hpp"

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[execution_smoke_test] " << message << "\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    using namespace omm;

    auto bus = std::make_shared<events::EventBus>();

    std::vector<events::ExecutionReportEvent> reports;
    std::vector<events::FillEvent> fills;

    bus->subscribe<events::ExecutionReportEvent>(
        [&](const events::ExecutionReportEvent& report) {
            reports.push_back(report);
        }
    );
    bus->subscribe<events::FillEvent>(
        [&](const events::FillEvent& fill) {
            fills.push_back(fill);
        }
    );

    demo::SimpleExecSimConfig cfg;
    cfg.max_fill_qty = 3;
    cfg.underlying_half_spread_bps = 0.0;

    demo::SimpleExecSim exec(bus, 100.0, cfg);
    exec.register_handlers();

    const auto ts = std::chrono::system_clock::now();
    bus->publish(events::MarketDataEvent{ts, 100.0});

    events::OrderSubmittedEvent order{
        "AAPL",
        events::Side::Buy,
        5,
        events::OrderType::Market
    };
    order.order_id = "SMOKE-1";
    order.producer = "hedge_order";
    order.reference_price = 100.0;
    bus->publish(order);

    bus->publish(events::MarketDataEvent{ts, 101.0});

    events::OrderSubmittedEvent bad_order{
        "AAPL",
        events::Side::Sell,
        0,
        events::OrderType::Market
    };
    bad_order.order_id = "SMOKE-BAD";
    bus->publish(bad_order);

    bool ok = true;
    ok &= expect(reports.size() == 4, "expected 4 execution reports");
    ok &= expect(fills.size() == 2, "expected 2 fills from split order");

    if (reports.size() >= 4) {
        ok &= expect(reports[0].status == events::OrderStatus::Accepted,
                     "first report should be Accepted");
        ok &= expect(reports[1].status == events::OrderStatus::PartiallyFilled,
                     "second report should be PartiallyFilled");
        ok &= expect(reports[2].status == events::OrderStatus::Filled,
                     "third report should be Filled");
        ok &= expect(reports[3].status == events::OrderStatus::Rejected,
                     "fourth report should be Rejected");
    }
    if (fills.size() >= 2) {
        ok &= expect(fills[0].fill_qty == 3 && fills[0].remaining_qty == 2,
                     "first fill should be qty=3 remaining=2");
        ok &= expect(fills[1].fill_qty == 2 && fills[1].remaining_qty == 0,
                     "second fill should be qty=2 remaining=0");
    }

    if (!ok) {
        return 1;
    }

    std::cout << "[execution_smoke_test] passed\n";
    return 0;
}
