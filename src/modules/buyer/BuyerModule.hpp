#pragma once
#include <memory>
#include <iostream>
#include "../../core/events/EventBus.hpp"
#include "../../core/events/Events.hpp"
#include "../../core/analytics/ImpliedVarianceExtractor.hpp"
#include "../../core/interfaces/IEntryPolicy.hpp"
#include "IAlphaSignal.hpp"

// ============================================================
// File: BuyerModule.hpp
// Role: wiring template for the buyer-side pipeline.
//
// Accepts injected implementations of IAlphaSignal and IEntryPolicy
// so the demo (or any caller) supplies the concrete strategy logic.
// BuyerModule only handles registration and logging — no concrete
// strategy instantiation happens here.
// ============================================================

namespace omm::buyer {

struct BuyerConfig {
    double interest_rate = 0.05;
};

// BuyerContext — returned by install(); holds shared_ptrs for external
// access (monitoring, testing).
struct BuyerContext {
    std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor;
    std::shared_ptr<IAlphaSignal>                        signal;
    std::shared_ptr<core::IEntryPolicy>                  controller;
};

class BuyerModule {
public:
    // install() — wire all buyer components onto the bus.
    //
    // Parameters:
    //   bus        — shared event bus for the buyer pipeline
    //   extractor  — IV extractor (from core/analytics, shared tool)
    //   signal     — concrete IAlphaSignal impl (injected by demo)
    //   controller — concrete IEntryPolicy impl (injected by demo)
    //   cfg        — optional config (interest rate etc.)
    static BuyerContext install(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<IAlphaSignal>                        signal,
        std::shared_ptr<core::IEntryPolicy>                  controller,
        const BuyerConfig&                                   cfg = {}
    ) {
        (void)cfg;

        // ── 注册 extractor 处理器（owned by BuyerModule） ────
        extractor->register_handlers();
        std::cout << "[BuyerModule] ImpliedVarianceExtractor 已注册\n";

        // ── signal 和 controller 由调用方在 install() 前注册 ──
        // 调用方（demo）负责在传入前调用各自的 register_handlers()。
        std::cout << "[BuyerModule] IAlphaSignal + IEntryPolicy 已由调用方注册\n";

        // ── 信号快照日志订阅 ──────────────────────────────────
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
