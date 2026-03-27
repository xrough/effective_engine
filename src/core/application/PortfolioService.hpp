#pragma once
#include <memory>
#include <string>
#include <vector>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"
#include "../domain/Instrument.hpp"
#include "../domain/PortfolioAggregate.hpp"
#include "../analytics/PricingEngine.hpp"

// ============================================================
// File: PortfolioService.hpp
//
// Design: Isolate "position tracking" from risk management applications, allowing SellerRiskApp / BuyerRiskApp
//       to focus solely on strategy evaluation logic without managing position states.
//
// Event Subscription:
//   FillEvent       → Update positions (applyFill), trigger revaluation
//   MarketDataEvent → Update underlying prices, trigger revaluation    
//
// Event Publishing:
//   PortfolioUpdateEvent → Contains latest RiskMetrics snapshot for risk management applications to consume
//
// Lifecycle: Created and connected by SellerModule / BuyerModule's install() method.
// ============================================================

namespace omm::application {

class PortfolioService {
public:
    PortfolioService(
        std::shared_ptr<events::EventBus>            bus,
        std::shared_ptr<domain::IPricingEngine>      pricing_engine,
        std::vector<std::shared_ptr<domain::Option>> options,
        std::string                                  underlying_id,
        std::string                                  account_id
    );

    void register_handlers();

private:
    // deal is filled, update position and revalue
    void on_fill(const events::FillEvent& event);

    // market data updated: refresh price → revalue → publish snapshot (if holding positions)
    void on_market(const events::MarketDataEvent& event);

    // value portfolio and publish PortfolioUpdateEvent
    void publish_snapshot();

    std::shared_ptr<events::EventBus>       bus_;
    std::shared_ptr<domain::IPricingEngine> pricing_engine_;
    std::string                             underlying_id_;
    std::string                             account_id_;

    domain::PortfolioAggregate portfolio_;
    double                     last_price_; // 最近一次标的价格
};

} // namespace omm::application
