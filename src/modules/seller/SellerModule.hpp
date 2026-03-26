#pragma once
#include <memory>
#include <vector>
#include <string>
#include "../../core/events/EventBus.hpp"
#include "../../core/domain/Instrument.hpp"
#include "../../core/analytics/PricingEngine.hpp"
#include "../../core/analytics/RiskPolicy.hpp"
#include "../../core/application/PortfolioService.hpp"
#include "QuoteEngine.hpp"
#include "DeltaHedger.hpp"
#include "SellerRiskApp.hpp"
#include "ProbabilisticTaker.hpp"

// ============================================================
// 文件：SellerModule.hpp
// 职责：卖方（做市商）模块的组合入口。
//
// 模式：配置驱动的静态组合（Config-Driven Static Composition）
//   install() 构造所有卖方组件，完成与 EventBus 的连线。
//   调用方（main.cpp）只需提供共享基础设施和配置参数。
//
// Phase 2：引入 PortfolioService，将持仓追踪从 SellerRiskApp 中解耦。
//   PortfolioService 订阅 FillEvent + MarketDataEvent → 发布 PortfolioUpdateEvent
//   SellerRiskApp 只订阅 PortfolioUpdateEvent → 评估策略 → 发布风控事件
// ============================================================

namespace omm::seller {

struct SellerConfig {
    double      half_spread       = 0.05;     // 单边价差（美元）
    double      delta_threshold   = 0.5;      // Delta 对冲触发阈值
    double      trade_probability = 0.30;     // 概率成交模拟器的成交概率
    unsigned    rng_seed          = 42;       // 随机数种子（确保仿真可重现）
    double      loss_limit        = 1e6;      // BlockOrders 亏损上限（美元）
    double      delta_limit       = 10000.0;  // ReduceOnly Delta 绝对值上限
    double      drawdown_limit    = 5e5;      // RiskAlertEvent 日内回撤阈值
    std::string account_id        = "DESK_A";
    std::string underlying_id     = "AAPL";
};

class SellerModule {
public:
    // install() — 构造并连线所有卖方组件
    //
    // 参数：
    //   bus      — 主事件总线（shared_ptr，生命周期由调用方管理）
    //   cfg      — 卖方运行时配置
    //   options  — 待报价的期权合约列表
    //   pricing  — 定价引擎（策略模式，可替换）
    static void install(
        std::shared_ptr<events::EventBus>                    bus,
        const SellerConfig&                                  cfg,
        const std::vector<std::shared_ptr<domain::Option>>& options,
        std::shared_ptr<domain::IPricingEngine>              pricing
    ) {
        // ── 持仓服务（模式无关，两端共享）───────────────────
        // PortfolioService 订阅 FillEvent + MarketDataEvent
        // 发布 PortfolioUpdateEvent 供 SellerRiskApp 消费
        auto portfolio_svc = std::make_shared<application::PortfolioService>(
            bus, pricing, options, cfg.underlying_id, cfg.account_id
        );
        portfolio_svc->register_handlers();

        // ── 报价引擎 ─────────────────────────────────────────
        auto quote_eng = std::make_shared<application::QuoteEngine>(
            bus, pricing, options, cfg.half_spread
        );
        quote_eng->register_handlers();

        // ── Delta 对冲器 ─────────────────────────────────────
        // DeltaHedger 维护独立的 PositionManager（用于对冲 Delta 计算）
        auto position_mgr = std::make_shared<domain::PositionManager>();
        auto hedger = std::make_shared<application::DeltaHedger>(
            bus, position_mgr, pricing, options,
            cfg.underlying_id, cfg.delta_threshold
        );
        hedger->register_handlers();
        bus->subscribe<events::MarketDataEvent>(
            [hedger](const events::MarketDataEvent& e) {
                hedger->update_market_price(e.underlying_price);
            }
        );

        // ── 实时风控（SellerRiskApp）─────────────────────────
        // Phase 2：只需 bus + account_id + risk_policy，不再管理持仓
        auto risk_policy = std::make_shared<domain::SimpleRiskPolicy>(
            cfg.loss_limit, cfg.delta_limit, cfg.drawdown_limit
        );
        auto risk_app = std::make_shared<application::SellerRiskApp>(
            bus, cfg.account_id, risk_policy
        );
        risk_app->register_handlers();

        // ── 概率成交模拟器 ───────────────────────────────────
        auto taker = std::make_shared<infrastructure::ProbabilisticTaker>(
            bus, cfg.trade_probability, cfg.rng_seed
        );
        taker->register_handlers();
    }
};

} // namespace omm::seller
