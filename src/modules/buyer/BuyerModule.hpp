#pragma once
#include <memory>
#include <iostream>
#include "../../core/events/EventBus.hpp"
#include "../../core/analytics/RoughVolPricingEngine.hpp"
#include "ImpliedVarianceExtractor.hpp"
#include "VarianceAlphaSignal.hpp"
#include "StrategyController.hpp"

// ============================================================
// 文件：BuyerModule.hpp
// 职责：买方（Alpha 策略）模块的组合入口。
//
// 模式：配置驱动的静态组合（Config-Driven Static Composition）
//   install() 构造所有买方组件，完成与 EventBus 的连线。
//   调用方（main.cpp）只需提供共享基础设施和配置参数。
//
// 信号流：
//   OptionMidQuoteEvent
//     → ImpliedVarianceExtractor（BS 隐含波动率反推）
//     → VarianceAlphaSignal（滚动 z-score → SignalSnapshotEvent）
//     → StrategyController（状态机 → OrderSubmittedEvent）
// ============================================================

namespace omm::buyer {

struct BuyerConfig {
    AlphaSignalConfig        signal;      // 信号参数（窗口、阈值、vega 预算）
    StrategyControllerConfig controller;  // 策略参数（超时、冷却、止损）
    double interest_rate = 0.05;         // 无风险利率（供 BS 隐含波动率计算）
};

// BuyerContext — install() 的返回值
// 持有所有买方组件的 shared_ptr，确保生命周期延续至仿真结束
struct BuyerContext {
    std::shared_ptr<ImpliedVarianceExtractor> extractor;
    std::shared_ptr<VarianceAlphaSignal>      signal;
    std::shared_ptr<StrategyController>       controller;
};

class BuyerModule {
public:
    // install() — 构造并连线所有买方组件，返回 BuyerContext
    //
    // 参数：
    //   bus          — 事件总线（生命周期由调用方管理）
    //   rough_engine — 已校准的粗糙波动率引擎（提供 xi0 参数）
    //   cfg          — 买方运行时配置
    static BuyerContext install(
        std::shared_ptr<events::EventBus>              bus,
        std::shared_ptr<domain::RoughVolPricingEngine> rough_engine,
        const BuyerConfig&                             cfg = {}
    ) {
        // ── 隐含方差提取器 ─────────────────────────────────────
        auto extractor = std::make_shared<ImpliedVarianceExtractor>(
            bus, cfg.interest_rate
        );
        extractor->register_handlers();
        std::cout << "[BuyerModule] ImpliedVarianceExtractor 已注册"
                  << " (r=" << cfg.interest_rate << ")\n";

        // ── 方差 Alpha 信号 ────────────────────────────────────
        auto signal = std::make_shared<VarianceAlphaSignal>(
            bus, extractor, rough_engine, cfg.signal
        );
        signal->register_handlers();
        std::cout << "[BuyerModule] VarianceAlphaSignal 已注册"
                  << " (window=" << cfg.signal.window
                  << ", z_entry=" << cfg.signal.z_entry << ")\n";

        // ── 策略控制器 ─────────────────────────────────────────
        auto controller = std::make_shared<StrategyController>(
            bus, cfg.signal, cfg.controller
        );
        controller->register_handlers();
        std::cout << "[BuyerModule] StrategyController 已注册"
                  << " (max_holding=" << cfg.controller.max_holding_bars
                  << ", cooldown=" << cfg.controller.cooldown_bars << ")\n";

        // ── 信号快照日志订阅（调试用）─────────────────────────
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
