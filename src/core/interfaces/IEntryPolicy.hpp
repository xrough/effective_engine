#pragma once
#include <optional>
#include <string>
#include "../events/Events.hpp"
#include "../domain/RiskMetrics.hpp"

// ============================================================
// Files：IEntryPolicy.hpp
// Role：decision gate, Alpha -> OrderRequest given risk metrics.
//

// Principles:
//   - Pure behavioral contract: evaluate() is imperative — given a signal + risk metrics, return whether to enter and with what parameters
//   - No EventBus; caller (OrderEngine) is responsible for submitting the OrderRequest to the IExecutionPolicy
// ============================================================

namespace omm::core {

// OrderRequest — represents a decision to enter a trade, returned by IEntryPolicy::evaluate() when conditions are met. 
struct OrderRequest {
    std::string instrument_id; 
    events::Side side;         // 方向（Buy/Sell，从我方视角）
    double quantity;           
    double limit_price;        
    std::string strategy_id;   // strategy identifier for tracking/logging
};

class IEntryPolicy {
public:
    virtual ~IEntryPolicy() = default;

    // evaluate() — given the latest market data and risk metrics, decide whether to enter a trade. Returns:
    //   - std::nullopt: no action
    //   - OrderRequest: parameters for the new trade to enter (e.g., instrument, side, quantity)
    virtual std::optional<OrderRequest> evaluate(
        const events::MarketDataEvent& market,
        const domain::RiskMetrics&     metrics
    ) = 0;
};

} // namespace omm::core
