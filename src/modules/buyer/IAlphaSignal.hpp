#pragma once
#include "../../core/events/Events.hpp"

// ============================================================
// File: IAlphaSignal.hpp
// Role: buyer-local interface for alpha signal generators.
//
// Buyer-side only — lives in modules/buyer/, not core/interfaces/,
// because no seller component implements this contract.
// ============================================================

namespace omm::buyer {

class IAlphaSignal {
public:
    virtual ~IAlphaSignal() = default;

    // Called when a new market data tick arrives (high-frequency).
    // Implementations use this for delta/stop checks, not signal recomputation.
    virtual void on_market_data(const events::MarketDataEvent& event) = 0;

    // Register EventBus subscriptions. Must be called before any events fire.
    virtual void register_handlers() = 0;
};

} // namespace omm::buyer
