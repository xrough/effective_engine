#pragma once
#include <memory>
#include <string>
#include <vector>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"
#include "../domain/Instrument.hpp"
#include "../domain/PricingEngine.hpp"
#include "../domain/PortfolioAggregate.hpp"
#include "../domain/RiskPolicy.hpp"

// ============================================================
// 文件：RealtimeRiskApp.hpp
// 职责：实时风险监控应用 — 订阅成交与行情事件，实时评估账户风险，
//       并在触发限额时通过 EventBus 发出风控指令。
//
// 对应 Risk_Calibration.md §1（Realtime Risk Application）
//
// 事件订阅：
//   TradeExecutedEvent  → on_trade()   — 更新持仓，触发风险检查
//   MarketDataEvent     → on_market()  — 重新估值，触发风险检查
//
// 事件发布：
//   RiskControlEvent  — 风控指令（BlockOrders / ReduceOnly）
//   RiskAlertEvent    — 风险预警（仅通知，不强制动作）
//
// 设计：
//   - 与 DeltaHedger 共享同一个 EventBus，但管理独立的 PortfolioAggregate
//   - PortfolioAggregate 是 RealtimeRiskApp 的私有状态，外部不可访问
//   - 风险策略（IRiskPolicy）通过构造函数注入，支持运行时替换（策略模式）
// ============================================================

namespace omm::application {

class RealtimeRiskApp {
public:
    RealtimeRiskApp(
        std::shared_ptr<events::EventBus>            bus,
        std::shared_ptr<domain::IPricingEngine>      pricing_engine,
        std::vector<std::shared_ptr<domain::Option>> options,
        std::string                                  underlying_id,
        std::string                                  account_id,
        std::shared_ptr<domain::IRiskPolicy>         risk_policy
    );

    // register_handlers() — 向 EventBus 注册事件处理器
    void register_handlers();

private:
    // on_trade() — 成交事件处理器
    //   1. 更新持仓（applyTrade）
    //   2. 重新估值（markToMarket）
    //   3. 计算风险指标（computeMetrics）
    //   4. 评估风险策略（evaluate）
    //   5. 发布风控事件
    void on_trade(const events::TradeExecutedEvent& event);

    // on_market() — 行情事件处理器
    //   1. 更新标的价格
    //   2. 重新估值（markToMarket）
    //   3. 计算风险指标（computeMetrics）
    //   4. 评估风险策略
    //   5. 发布风控事件（如有）
    void on_market(const events::MarketDataEvent& event);

    // publish_risk_actions() — 将策略评估结果广播到 EventBus
    void publish_risk_actions(
        const std::vector<events::RiskControlEvent>& actions
    );

    std::shared_ptr<events::EventBus>       bus_;
    std::shared_ptr<domain::IPricingEngine> pricing_engine_;
    std::string                             underlying_id_;
    std::string                             account_id_;
    std::shared_ptr<domain::IRiskPolicy>    risk_policy_;

    domain::PortfolioAggregate portfolio_; // 账户持仓聚合（私有状态）
    double                     last_price_; // 最近一次标的价格（用于跨事件估值）
};

} // namespace omm::application
