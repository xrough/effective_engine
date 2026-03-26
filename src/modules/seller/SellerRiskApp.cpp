#include "SellerRiskApp.hpp"
#include <iostream>
#include <iomanip>

// ============================================================
// SellerRiskApp 实现（Phase 2 简化版）
//
// 工作流：
//   收到 PortfolioUpdateEvent（由 PortfolioService 生成）
//   → 评估 IRiskPolicy
//   → 发布 RiskControlEvent / RiskAlertEvent
// ============================================================

namespace omm::application {

SellerRiskApp::SellerRiskApp(
    std::shared_ptr<events::EventBus>    bus,
    std::string                          account_id,
    std::shared_ptr<domain::IRiskPolicy> risk_policy)
    : bus_(std::move(bus))
    , account_id_(std::move(account_id))
    , risk_policy_(std::move(risk_policy)) {}

void SellerRiskApp::register_handlers() {
    // 订阅持仓快照事件（由 PortfolioService 在每次成交/行情更新后发布）
    bus_->subscribe<events::PortfolioUpdateEvent>(
        [this](const events::PortfolioUpdateEvent& evt) {
            this->on_portfolio_update(evt);
        }
    );
}

void SellerRiskApp::on_portfolio_update(
    const events::PortfolioUpdateEvent& event) {

    // 仅处理本账户的快照（多账户场景下过滤）
    if (event.account_id != account_id_) return;

    const auto& metrics = event.metrics;

    std::cout << "[实时风控|" << account_id_ << "] 持仓快照:"
              << "  已实现盈亏=$" << std::fixed << std::setprecision(2)
              << metrics.realized_pnl
              << "  未实现盈亏=$" << metrics.unrealized_pnl
              << "  组合Δ=" << std::setprecision(3) << metrics.delta << "\n";

    // ── 评估风险策略（策略模式委托）───────────────────────
    auto actions = risk_policy_->evaluate(account_id_, metrics);
    publish_risk_actions(actions);

    // ── 发布日内回撤预警（独立于强制动作）────────────────
    if (metrics.intraday_drawdown > 0.0) {
        events::RiskAlertEvent alert{
            account_id_,
            "intraday_drawdown",
            metrics.intraday_drawdown,
            5e5  // 日内回撤预警阈值（$500,000）
        };
        bus_->publish(alert);
    }
}

void SellerRiskApp::publish_risk_actions(
    const std::vector<events::RiskControlEvent>& actions) {

    for (const auto& action : actions) {
        bus_->publish(action);

        std::cout << "[实时风控|" << account_id_ << "] 发出风控指令: ";
        switch (action.action) {
            case events::RiskAction::BlockOrders:
                std::cout << "冻结下单  原因: " << action.reason << "\n";
                break;
            case events::RiskAction::CancelOrders:
                std::cout << "批量撤单  原因: " << action.reason << "\n";
                break;
            case events::RiskAction::ReduceOnly:
                std::cout << "限制为减仓  原因: " << action.reason << "\n";
                break;
        }
    }
}

} // namespace omm::application
