#pragma once
#include <optional>
#include <string>
#include "../events/Events.hpp"
#include "../domain/RiskMetrics.hpp"

// ============================================================
// 文件：IEntryPolicy.hpp
// 职责：入场决策纯接口 — 将 Alpha 信号转化为订单请求。
//
// 设计原则：
//   - 纯行为契约，evaluate() 是命令式的：给定信号 + 风险指标，
//     返回是否入场以及入场参数
//   - 不含 EventBus；调用方（OrderEngine）负责将 OrderRequest 提交给
//     IExecutionPolicy
// ============================================================

namespace omm::core {

// OrderRequest — 入场决策的输出：描述一笔待提交订单的参数
struct OrderRequest {
    std::string instrument_id; // 目标合约
    events::Side side;         // 方向（Buy/Sell，从我方视角）
    double quantity;           // 数量（手）
    double limit_price;        // 限价（0.0 = 市价单）
    std::string strategy_id;   // 策略标识（供审计/归因使用）
};

class IEntryPolicy {
public:
    virtual ~IEntryPolicy() = default;

    // evaluate() — 根据信号与当前风险指标决定是否入场
    //   返回 std::nullopt 表示跳过此次信号（不入场）
    virtual std::optional<OrderRequest> evaluate(
        const events::MarketDataEvent& market,
        const domain::RiskMetrics&     metrics
    ) = 0;
};

} // namespace omm::core
