#pragma once
#include <vector>
#include "../domain/RiskMetrics.hpp"
#include "../events/Events.hpp"

// ============================================================
// 文件：RiskPolicy.hpp
// 职责：定义风险策略接口（IRiskPolicy）及默认实现（SimpleRiskPolicy）。
//
// 模式：策略模式（Strategy Pattern）
//   - IRiskPolicy 抽象"如何根据风险指标做出决策"
//   - SimpleRiskPolicy 实现基本限额管控规则
//   - 未来可扩展：VaRPolicy、RegulatorPolicy 等，只需实现接口
//
// 与 Risk_Calibration.md §1.8 对应。
// ============================================================

namespace omm::domain {

// ============================================================
// IRiskPolicy — 风险策略接口
//   输入：当前账户 ID + RiskMetrics 快照
//   输出：需要执行的风控事件列表（可为空）
// ============================================================
class IRiskPolicy {
public:
    virtual ~IRiskPolicy() = default;

    // evaluate() — 评估风险指标，返回需要触发的风控事件
    virtual std::vector<events::RiskControlEvent> evaluate(
        const std::string&   account_id,
        const RiskMetrics&   metrics
    ) const = 0;
};

// ============================================================
// SimpleRiskPolicy — 简单限额风险策略（MVP 实现）
//
// 规则（来自 Risk_Calibration.md §1.8）：
//   1. realized_pnl < -loss_limit_         → BlockOrders（止损冻结）
//   2. |delta| > delta_limit_              → ReduceOnly（Delta 超限，限制增仓）
//   3. intraday_drawdown > drawdown_limit_ → 发出 RiskAlertEvent（预警，不强制动作）
//
// 参数（构造函数可配置）：
//   loss_limit_     — 最大允许亏损（默认 $1,000,000）
//   delta_limit_    — 最大允许绝对 Delta（默认 10,000）
//   drawdown_limit_ — 日内回撤预警阈值（默认 $500,000）
// ============================================================
class SimpleRiskPolicy final : public IRiskPolicy {
public:
    explicit SimpleRiskPolicy(
        double loss_limit     = 1e6,
        double delta_limit    = 10000.0,
        double drawdown_limit = 5e5
    );

    std::vector<events::RiskControlEvent> evaluate(
        const std::string& account_id,
        const RiskMetrics& metrics
    ) const override;

private:
    double loss_limit_;     // 触发 BlockOrders 的亏损阈值（绝对值）
    double delta_limit_;    // 触发 ReduceOnly 的 Delta 绝对值上限
    double drawdown_limit_; // 触发风险预警的日内回撤上限
};

} // namespace omm::domain
