#pragma once
#include "../events/Events.hpp"

// ============================================================
// File: IAlphaSignal.hpp
// Role: Alpha signal interface for buy-side trading strategies
//
// Buyer-side only.
// ============================================================

namespace omm::core {

class IAlphaSignal {
public:
    virtual ~IAlphaSignal() = default;

    // Response to market and generate trading signals.
    virtual void on_market_data(const events::MarketDataEvent& event) = 0; 
};

} // namespace omm::core
