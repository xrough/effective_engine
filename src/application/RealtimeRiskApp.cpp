#include "RealtimeRiskApp.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

// ============================================================
// RealtimeRiskApp 实现
//
// 工作流（与 Risk_Calibration.md §1.5 一致）：
//   1. 收到 TradeExecutedEvent → 更新持仓 → 重新估值 → 计算 Greeks → 评估策略 → 发布风控
//   2. 收到 MarketDataEvent   → 更新标的价 → 重新估值 → 计算 Greeks → 评估策略 → 发布风控
// ============================================================

namespace omm::application {

RealtimeRiskApp::RealtimeRiskApp(
    std::shared_ptr<events::EventBus>            bus,
    std::shared_ptr<domain::IPricingEngine>      pricing_engine,
    std::vector<std::shared_ptr<domain::Option>> options,
    std::string                                  underlying_id,
    std::string                                  account_id,
    std::shared_ptr<domain::IRiskPolicy>         risk_policy)
    : bus_(std::move(bus))
    , pricing_engine_(std::move(pricing_engine))
    , underlying_id_(std::move(underlying_id))
    , account_id_(std::move(account_id))
    , risk_policy_(std::move(risk_policy))
    , portfolio_(account_id_, options)   // 初始化持仓聚合（账户 ID + 合约列表）
    , last_price_(150.0) {}              // 默认标的价格（开盘估计值）

void RealtimeRiskApp::register_handlers() {
    // 订阅成交事件 — 每笔客户成交后触发风险检查
    bus_->subscribe<events::TradeExecutedEvent>(
        [this](const events::TradeExecutedEvent& evt) {
            this->on_trade(evt);
        }
    );

    // 订阅行情事件 — 每次价格更新后重新估值并检查风险
    bus_->subscribe<events::MarketDataEvent>(
        [this](const events::MarketDataEvent& evt) {
            this->on_market(evt);
        }
    );
}

void RealtimeRiskApp::on_trade(const events::TradeExecutedEvent& event) {
    // ── 步骤 1：更新持仓 ──────────────────────────────────
    portfolio_.applyTrade(event);

    // ── 步骤 2：按最新市价重新估值 ───────────────────────
    portfolio_.markToMarket(*pricing_engine_, last_price_);

    // ── 步骤 3：计算当前风险指标快照 ─────────────────────
    auto metrics = portfolio_.computeMetrics(*pricing_engine_, last_price_);

    std::cout << "[实时风控|" << account_id_ << "] 成交后风险快照:"
              << "  已实现盈亏=$" << std::fixed << std::setprecision(2) << metrics.realized_pnl
              << "  未实现盈亏=$" << metrics.unrealized_pnl
              << "  组合Δ=" << std::setprecision(3) << metrics.delta << "\n";

    // ── 步骤 4：评估风险策略 ──────────────────────────────
    auto actions = risk_policy_->evaluate(account_id_, metrics);

    // ── 步骤 5：发布风控事件（如有触发）──────────────────
    publish_risk_actions(actions);

    // ── 发布日内回撤预警（独立于强制动作）────────────────
    if (metrics.intraday_drawdown > 0.0) {
        events::RiskAlertEvent alert{
            account_id_,
            "intraday_drawdown",
            metrics.intraday_drawdown,
            5e5 // 日内回撤预警阈值（$500,000）
        };
        bus_->publish(alert);
    }
}

void RealtimeRiskApp::on_market(const events::MarketDataEvent& event) {
    // ── 步骤 1：更新标的价格 ──────────────────────────────
    last_price_ = event.underlying_price;

    // ── 步骤 2：按新价格重新估值 ─────────────────────────
    portfolio_.markToMarket(*pricing_engine_, last_price_);

    // ── 步骤 3：计算风险指标快照 ─────────────────────────
    auto metrics = portfolio_.computeMetrics(*pricing_engine_, last_price_);

    // ── 步骤 4-5：评估并发布（仅在有持仓时输出，避免噪音）
    if (portfolio_.get_position(underlying_id_) != 0 || metrics.delta != 0.0) {
        auto actions = risk_policy_->evaluate(account_id_, metrics);
        publish_risk_actions(actions);
    }
}

void RealtimeRiskApp::publish_risk_actions(
    const std::vector<events::RiskControlEvent>& actions) {

    for (const auto& action : actions) {
        // 将风控指令广播到 EventBus（可被 OrderRouter 等组件订阅）
        bus_->publish(action);

        std::cout << "[实时风控|" << account_id_ << "] 🚨 发出风控指令: ";
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
