#include "RiskPolicy.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================
// SimpleRiskPolicy 实现
//
// 每次 evaluate() 调用：
//   1. 检查已实现亏损是否超出止损阈值 → 发出 BlockOrders
//   2. 检查组合 Delta 是否超出方向性风险限额 → 发出 ReduceOnly
//   3. 检查日内回撤是否超出预警阈值 → 记录日志（此处通过 RiskAlertEvent）
// ============================================================

namespace omm::domain {

SimpleRiskPolicy::SimpleRiskPolicy(
    double loss_limit,
    double delta_limit,
    double drawdown_limit)
    : loss_limit_(loss_limit)
    , delta_limit_(delta_limit)
    , drawdown_limit_(drawdown_limit) {}

std::vector<events::RiskControlEvent> SimpleRiskPolicy::evaluate(
    const std::string& account_id,
    const RiskMetrics& metrics) const {

    std::vector<events::RiskControlEvent> actions;

    // ── 规则 1：止损触发 ──────────────────────────────────
    // 当已实现亏损超出阈值时，冻结账户下单权限
    if (metrics.realized_pnl < -loss_limit_) {
        actions.push_back(events::RiskControlEvent{
            account_id,
            events::RiskAction::BlockOrders,
            "已实现亏损 $" + std::to_string(static_cast<int>(metrics.realized_pnl))
            + " 超出止损限额 $" + std::to_string(static_cast<int>(loss_limit_))
        });
        std::cout << "[风控策略|" << account_id << "] ⛔ 触发止损！"
                  << " 已实现盈亏: $" << std::fixed << std::setprecision(2)
                  << metrics.realized_pnl
                  << "  限额: -$" << loss_limit_ << "  → BlockOrders\n";
    }

    // ── 规则 2：Delta 超限 ────────────────────────────────
    // 当组合方向性风险超出限额时，限制为仅允许减仓方向的订单
    if (std::abs(metrics.delta) > delta_limit_) {
        actions.push_back(events::RiskControlEvent{
            account_id,
            events::RiskAction::ReduceOnly,
            "组合 Delta " + std::to_string(metrics.delta)
            + " 超出限额 ±" + std::to_string(static_cast<int>(delta_limit_))
        });
        std::cout << "[风控策略|" << account_id << "] ⚠ Delta 超限！"
                  << " Delta: " << std::fixed << std::setprecision(1)
                  << metrics.delta
                  << "  限额: ±" << delta_limit_ << "  → ReduceOnly\n";
    }

    // ── 规则 3：日内回撤预警 ──────────────────────────────
    // 当日内回撤超出预警阈值时，记录预警（不强制动作，仅通知）
    if (metrics.intraday_drawdown > drawdown_limit_) {
        std::cout << "[风控预警|" << account_id << "] 📉 日内回撤预警！"
                  << " 回撤: $" << std::fixed << std::setprecision(2)
                  << metrics.intraday_drawdown
                  << "  阈值: $" << drawdown_limit_ << "\n";
        // 注意：此处仅打印日志，RiskAlertEvent 由 RealtimeRiskApp 独立发布
    }

    return actions;
}

} // namespace omm::domain
