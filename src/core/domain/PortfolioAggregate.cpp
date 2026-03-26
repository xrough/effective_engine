#include "PortfolioAggregate.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// ============================================================
// PortfolioAggregate 实现
//
// 设计说明：
//   持仓更新采用"净持仓 + 平均成本"模型（Average Cost Basis）：
//   - 增仓时：加权平均更新平均成本
//   - 减仓时：按平均成本计算已实现盈亏，不改变剩余仓位的成本
// ============================================================

namespace omm::domain {

PortfolioAggregate::PortfolioAggregate(
    std::string account_id,
    std::vector<std::shared_ptr<Option>> options)
    : account_id_(std::move(account_id))
    , options_(std::move(options)) {}

void PortfolioAggregate::applyFill(const events::FillEvent& event) {
    const std::string& id = event.instrument_id;

    // 我方视角（Phase 2 统一语义）：
    //   Buy  → 我们买入 → 持仓增加
    //   Sell → 我们卖出 → 持仓减少
    int pos_change = (event.side == events::Side::Buy)
                     ? +event.fill_qty   // 我们买入：多头增加
                     : -event.fill_qty;  // 我们卖出：空头增加

    int old_pos  = positions_[id];
    int new_pos  = old_pos + pos_change;
    double price = event.fill_price;

    if (old_pos == 0 || (old_pos > 0 && pos_change > 0) || (old_pos < 0 && pos_change < 0)) {
        // 同向加仓：加权平均更新成本
        double old_notional = avg_cost_[id] * std::abs(old_pos);
        double new_notional = price * std::abs(pos_change);
        int    total_qty    = std::abs(old_pos) + std::abs(pos_change);
        avg_cost_[id] = (total_qty > 0)
                        ? (old_notional + new_notional) / total_qty
                        : price;
    } else {
        // 反向减仓：计算已实现盈亏
        int close_qty = std::min(std::abs(old_pos), std::abs(pos_change));
        // 做市商持有多头时：平仓盈亏 = (平仓价 - 成本价) × 数量
        // 做市商持有空头时：平仓盈亏 = (成本价 - 平仓价) × 数量
        double pnl = (old_pos > 0)
                     ? (price - avg_cost_[id]) * close_qty
                     : (avg_cost_[id] - price) * close_qty;
        realized_pnl_ += pnl;

        if (std::abs(pos_change) > std::abs(old_pos)) {
            // 反手：先平仓再反向建仓，更新成本为新价格
            avg_cost_[id] = price;
        }
        // 若完全平仓，保留 avg_cost_（下次建仓时会覆盖）
    }

    positions_[id] = new_pos;

    std::cout << "[持仓聚合|" << account_id_ << "] "
              << id << " 持仓更新: "
              << std::showpos << old_pos << " → " << new_pos
              << std::noshowpos
              << "  已实现盈亏累计: $"
              << std::fixed << std::setprecision(2) << realized_pnl_ << "\n";
}

void PortfolioAggregate::markToMarket(
    const IPricingEngine& engine, double underlying_price) {

    double unrealized = 0.0;

    for (const auto& opt : options_) {
        int pos = get_position(opt->id());
        if (pos == 0) continue;

        // 按 Black-Scholes / 简化模型获取当前理论价格
        PriceResult result = engine.price(*opt, underlying_price);
        double cost        = avg_cost_[opt->id()];

        // 未实现盈亏 = (当前价 - 成本价) × 净持仓
        // 注意：做市商多头 pos>0 时希望价格上涨；空头 pos<0 时希望价格下跌
        unrealized += (result.theo - cost) * static_cast<double>(pos);
    }

    unrealized_pnl_ = unrealized;

    // 更新日内总盈亏高水位（用于回撤计算）
    double total_pnl = realized_pnl_ + unrealized_pnl_;
    total_pnl_high_  = std::max(total_pnl_high_, total_pnl);
}

RiskMetrics PortfolioAggregate::computeMetrics(
    const IPricingEngine& engine, double underlying_price) const {

    RiskMetrics m;
    m.realized_pnl   = realized_pnl_;
    m.unrealized_pnl = unrealized_pnl_;

    // 计算组合 Delta：Σ(delta_i × position_i)
    for (const auto& opt : options_) {
        int pos = get_position(opt->id());
        if (pos == 0) continue;
        PriceResult result = engine.price(*opt, underlying_price);
        m.delta += result.delta * static_cast<double>(pos);
    }

    // 日内最大回撤 = 高水位 - 当前总盈亏
    double total_pnl      = realized_pnl_ + unrealized_pnl_;
    m.intraday_drawdown   = total_pnl_high_ - total_pnl;

    // Gamma、Vega、Theta 此处简化为 0（SimplePricingEngine 不提供这些值）
    // 接入完整 BS 引擎后可在此扩展
    m.gamma = 0.0;
    m.vega  = 0.0;
    m.theta = 0.0;

    // VaR 简化估算：1日 VaR ≈ |Delta| × S × σ × √(1/252) （95% 置信度）
    // 此处使用固定波动率 20% 作为占位符
    const double vol_proxy = 0.20;
    m.var_1d = std::abs(m.delta) * underlying_price * vol_proxy
               / std::sqrt(252.0);

    return m;
}

int PortfolioAggregate::get_position(const std::string& instrument_id) const {
    auto it = positions_.find(instrument_id);
    return (it != positions_.end()) ? it->second : 0;
}

} // namespace omm::domain
