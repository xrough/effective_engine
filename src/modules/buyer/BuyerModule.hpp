#pragma once
#include <memory>
#include <iostream>
#include "../../core/events/EventBus.hpp"
#include "../../core/analytics/RoughVolPricingEngine.hpp"
#include "ImpliedVarianceExtractor.hpp"
#include "VarianceAlphaSignal.hpp"
#include "StrategyController.hpp"

// ============================================================
// File：BuyerModule.hpp
// Role: entrypoint for buyer-side logic, assembling all components and wiring them to the EventBus.
// ============================================================

namespace omm::buyer {

struct BuyerConfig {
    AlphaSignalConfig        signal;      // signal parameters (window, threshold, vega budget)
    StrategyControllerConfig controller;  // strategy parameters (timeout, cooldown, stop-loss)
    double interest_rate = 0.05;       
};

// BuyerContext — install() returns this struct containing shared pointers to all buyer components, allowing external access if needed (e.g., for testing or monitoring).
struct BuyerContext {
    std::shared_ptr<ImpliedVarianceExtractor> extractor;
    std::shared_ptr<VarianceAlphaSignal>      signal;
    std::shared_ptr<StrategyController>       controller;
};

class BuyerModule {
public:
    // install() — construct and register all buyer components.
    //
    // parameters:
    //   bus      
    //   rough_engine — tuned rough vola model for signal generation
    //   cfg          
    static BuyerContext install(
        std::shared_ptr<events::EventBus>              bus,
        std::shared_ptr<domain::RoughVolPricingEngine> rough_engine,
        const BuyerConfig&                             cfg = {}
    ) {
        // ── implied variance extractor ──────────────────────────────────────
        auto extractor = std::make_shared<ImpliedVarianceExtractor>(
            bus, cfg.interest_rate
        );
        extractor->register_handlers();
        std::cout << "[BuyerModule] ImpliedVarianceExtractor is installed"
                  << " (r=" << cfg.interest_rate << ")\n";

        // ── variance alpha ────────────────────────────────────
        auto signal = std::make_shared<VarianceAlphaSignal>(
            bus, extractor, rough_engine, cfg.signal
        );
        signal->register_handlers();
        std::cout << "[BuyerModule] VarianceAlphaSignal is installed"
                  << " (window=" << cfg.signal.window
                  << ", z_entry=" << cfg.signal.z_entry << ")\n";

        // ── strategy controller ─────────────────────────────────────────
        auto controller = std::make_shared<StrategyController>(
            bus, cfg.signal, cfg.controller
        );
        controller->register_handlers();
        std::cout << "[BuyerModule] StrategyController is installed"
                  << " (max_holding=" << cfg.controller.max_holding_bars
                  << ", cooldown=" << cfg.controller.cooldown_bars << ")\n";

        // ── signal snapshot logging ──────────────────────────
        bus->subscribe<events::SignalSnapshotEvent>(
            [](const events::SignalSnapshotEvent& s) {
                if (s.valid && std::abs(s.zscore) > 0.5) {
                    std::cout << "[Alpha Signal] zscore=" << s.zscore
                              << "  spread=" << s.raw_spread
                              << "  IV=" << s.atm_implied_variance
                              << "  RoughFV=" << s.rough_forecast_variance << "\n";
                }
            }
        );

        return BuyerContext{extractor, signal, controller};
    }
};

} // namespace omm::buyer
