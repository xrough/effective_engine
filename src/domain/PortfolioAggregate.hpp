#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "RiskMetrics.hpp"
#include "PricingEngine.hpp"
#include "../events/Events.hpp"

// ============================================================
// 文件：PortfolioAggregate.hpp
// 职责：维护单账户的完整投资组合状态，并计算风险指标快照。
//
// 领域聚合根（Domain Aggregate Root）：
//   PortfolioAggregate 封装了账户内所有持仓、盈亏状态，
//   保证内部数据一致性（持仓变更必须通过 applyTrade() 进行）。
//
// 主要职责：
//   1. applyTrade()    — 应用成交，更新净持仓与已实现盈亏
//   2. markToMarket()  — 按当前市价更新未实现盈亏与高水位
//   3. computeMetrics()— 聚合希腊字母与风险指标，生成 RiskMetrics 快照
// ============================================================

namespace omm::domain {

class PortfolioAggregate {
public:
    // 构造函数：传入账户 ID 与期权合约列表（用于 Greeks 计算）
    PortfolioAggregate(
        std::string account_id,
        std::vector<std::shared_ptr<Option>> options
    );

    // applyTrade() — 应用一笔成交事件
    //   - 更新对应合约的净持仓（net position）
    //   - 对于减仓操作，计算并累计已实现盈亏（FIFO 成本基础）
    void applyTrade(const events::TradeExecutedEvent& event);

    // markToMarket() — 按当前标的价格重新估值
    //   - 更新每个合约的未实现盈亏（按持仓量 × (当前价 - 平均成本)）
    //   - 更新日内高水位，计算当前回撤
    void markToMarket(const IPricingEngine& engine, double underlying_price);

    // computeMetrics() — 生成当前时刻的风险指标快照
    //   - 汇总 realized_pnl、unrealized_pnl
    //   - 计算组合 Delta（Σ delta_i × position_i）
    //   - 计算日内最大回撤
    RiskMetrics computeMetrics(const IPricingEngine& engine,
                                double underlying_price) const;

    // 查询指定合约净持仓
    int get_position(const std::string& instrument_id) const;

private:
    std::string account_id_;   // 账户标识符（如 "DESK_A"）

    // 净持仓：instrument_id → 净头寸（正 = 多头，负 = 空头）
    std::unordered_map<std::string, int> positions_;

    // 平均成本：instrument_id → 平均建仓成本（美元/手）
    std::unordered_map<std::string, double> avg_cost_;

    // 期权合约列表（用于 Greeks 计算时映射合约属性）
    std::vector<std::shared_ptr<Option>> options_;

    double realized_pnl_      = 0.0; // 累计已实现盈亏
    double unrealized_pnl_    = 0.0; // 当前未实现盈亏（随每次 markToMarket 更新）
    double total_pnl_high_    = 0.0; // 日内总盈亏高水位（用于计算回撤）
};

} // namespace omm::domain
