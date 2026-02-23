#include "PositionManager.hpp"
#include <iostream>
#include <iomanip>

namespace omm::domain {

void PositionManager::on_trade_executed(
    const events::TradeExecutedEvent& event) {

    // 确定持仓变化方向：
    //   客户 Buy  → 我们做空该合约 → 持仓 -= quantity
    //   客户 Sell → 我们做多该合约 → 持仓 += quantity
    int delta_qty = (event.side == events::Side::Buy)
                    ? -event.quantity   // 客户买入：我们被迫卖出，空头增加
                    : +event.quantity;  // 客户卖出：我们被迫买入，多头增加

    positions_[event.instrument_id] += delta_qty;

    // 打印持仓更新日志，便于仿真过程观察
    std::cout << "[PositionManager] 持仓更新: "
              << event.instrument_id
              << " → " << std::showpos << positions_[event.instrument_id]
              << std::noshowpos << "\n";
}

int PositionManager::get_position(const std::string& instrument_id) const {
    auto it = positions_.find(instrument_id);
    // 若该合约从未出现过成交，默认持仓为 0
    return (it != positions_.end()) ? it->second : 0;
}

double PositionManager::compute_portfolio_delta(
    const std::unordered_map<std::string, double>& deltas) const {

    double total_delta = 0.0;

    // 遍历所有持仓合约，累加各合约的 Delta 贡献
    // Delta 贡献 = 单位 Delta × 持仓数量
    for (const auto& [id, pos] : positions_) {
        auto it = deltas.find(id);
        if (it != deltas.end()) {
            total_delta += it->second * static_cast<double>(pos);
        }
        // 若 deltas 中没有该合约（如已到期），贡献为 0，跳过
    }

    return total_delta;
}

void PositionManager::print_positions() const {
    if (positions_.empty()) {
        std::cout << "[PositionManager] 当前无持仓\n";
        return;
    }
    std::cout << "[PositionManager] 最终持仓汇总:\n";
    for (const auto& [id, pos] : positions_) {
        // 使用 showpos 显示符号（+/-），使多空方向一目了然
        std::cout << "  " << std::left << std::setw(30) << id
                  << std::showpos << pos << std::noshowpos << "\n";
    }
}

} // namespace omm::domain
