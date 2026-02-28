#pragma once

// ============================================================
// 文件：RiskMetrics.hpp
// 职责：定义单账户的风险指标快照（纯数据值对象）。
//
// RiskMetrics 由 PortfolioAggregate::computeMetrics() 生成，
// 传递给 IRiskPolicy::evaluate() 进行风险策略评估。
//
// 设计原则：
//   - 纯值对象（Value Object），无行为、无 I/O。
//   - 一次性计算后不可变，保证风控评估的确定性。
// ============================================================

namespace omm::domain {

struct RiskMetrics {
    // ── 盈亏指标 ──────────────────────────────────────────
    double realized_pnl      = 0.0; // 已实现盈亏（美元），平仓后计入
    double unrealized_pnl    = 0.0; // 未实现盈亏（美元），按当前市价估值

    // ── 希腊字母（Greeks）────────────────────────────────
    // 衡量期权组合对市场因子的一阶敏感度
    double delta             = 0.0; // Δ — 组合价值对标的价格的一阶导数（dV/dS）
    double gamma             = 0.0; // Γ — Delta 对标的价格的变化率（d²V/dS²）
    double vega              = 0.0; // ν — 组合价值对隐含波动率的敏感度（dV/dσ）
    double theta             = 0.0; // Θ — 时间衰减：每日组合价值减少量（dV/dt）

    // ── 下行风险指标 ─────────────────────────────────────
    double var_1d            = 0.0; // 1日 VaR（在险价值），置信度 95%
    double intraday_drawdown = 0.0; // 日内最大回撤（从日内高水位计算）
};

} // namespace omm::domain
