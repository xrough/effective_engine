#pragma once
#include <memory>
#include <string>
#include <vector>
#include "../../core/events/EventBus.hpp"
#include "../../core/events/Events.hpp"
#include "../../core/analytics/RiskPolicy.hpp"

// ============================================================
// 文件：SellerRiskApp.hpp
// 职责：卖方风险策略应用 — 订阅持仓快照，评估风险限额，
//       在触发时发出风控指令。
//
// Phase 2 简化（PortfolioService 解耦后）：
//   SellerRiskApp 不再维护持仓状态，只订阅 PortfolioUpdateEvent，
//   应用 IRiskPolicy，发布 RiskControlEvent / RiskAlertEvent。
//
// 事件订阅：
//   PortfolioUpdateEvent → on_portfolio_update() → 评估策略 → 发布风控
//
// 事件发布：
//   RiskControlEvent  — 风控指令（BlockOrders / ReduceOnly）
//   RiskAlertEvent    — 风险预警（仅通知，不强制动作）
//
// 设计：
//   - 持仓追踪由 PortfolioService 负责（解耦）
//   - 风险策略（IRiskPolicy）通过构造函数注入（策略模式）
// ============================================================

namespace omm::application {

class SellerRiskApp {
public:
    SellerRiskApp(
        std::shared_ptr<events::EventBus>    bus,
        std::string                          account_id,
        std::shared_ptr<domain::IRiskPolicy> risk_policy
    );

    // register_handlers() — 向 EventBus 注册 PortfolioUpdateEvent 处理器
    void register_handlers();

private:
    // on_portfolio_update() — 收到持仓快照后评估风险策略
    void on_portfolio_update(const events::PortfolioUpdateEvent& event);

    // publish_risk_actions() — 将策略评估结果广播到 EventBus
    void publish_risk_actions(
        const std::vector<events::RiskControlEvent>& actions
    );

    std::shared_ptr<events::EventBus>    bus_;
    std::string                          account_id_;
    std::shared_ptr<domain::IRiskPolicy> risk_policy_;
};

} // namespace omm::application
