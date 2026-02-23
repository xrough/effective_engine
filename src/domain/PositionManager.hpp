#pragma once
#include <string>
#include <unordered_map>
#include "../events/Events.hpp"

// ============================================================
// 文件：PositionManager.hpp
// 职责：跟踪做市商当前在所有金融工具上的净持仓（Inventory）。
//
// 持仓语义：
//   正数 → 多头（Long）：我们持有该资产，价格上涨则获利
//   负数 → 空头（Short）：我们已卖出该资产，价格下跌则获利
//
// 做市商视角（与客户方向相反）：
//   客户 Buy（买入期权）→ 我们卖出 → 持仓减少（空头）
//   客户 Sell（卖出期权）→ 我们买入 → 持仓增加（多头）
//
// 订阅事件：TradeExecutedEvent（成交事件）
// ============================================================

namespace omm::domain {

class PositionManager {
public:
    PositionManager() = default;

    // ----------------------------------------------------------
    // on_trade_executed() — 成交事件处理器
    //
    // 根据成交方向更新对应合约的持仓。
    // 此方法既可直接注册到 EventBus，也可由其他组件（如 DeltaHedger）
    // 直接调用（用于对冲成交后的直接持仓更新，避免重入事件链）。
    // ----------------------------------------------------------
    void on_trade_executed(const events::TradeExecutedEvent& event);

    // ----------------------------------------------------------
    // get_position() — 查询单个合约的当前净持仓
    //
    // 若合约从未出现过任何成交，返回 0。
    // ----------------------------------------------------------
    int get_position(const std::string& instrument_id) const;

    // ----------------------------------------------------------
    // compute_portfolio_delta() — 计算组合的总 Delta 敞口
    //
    // 公式：Δ_portfolio = Σ (Δ_instrument_i × position_i)
    //
    // 参数：
    //   deltas — 各合约 ID 到其 Delta 值的映射（由 DeltaHedger 提供）
    //
    // 返回：有符号浮点数，表示总 Delta 敞口
    //   正值 → 净多头 Delta（标的价格上涨则组合获益）
    //   负值 → 净空头 Delta（标的价格下跌则组合获益）
    // ----------------------------------------------------------
    double compute_portfolio_delta(
        const std::unordered_map<std::string, double>& deltas
    ) const;

    // ----------------------------------------------------------
    // print_positions() — 将当前持仓打印到标准输出（用于仿真结束时汇报）
    // ----------------------------------------------------------
    void print_positions() const;

private:
    // 核心数据：合约 ID → 有符号净持仓
    // 使用 unordered_map 保证 O(1) 查找和更新
    std::unordered_map<std::string, int> positions_;
};

} // namespace omm::domain
