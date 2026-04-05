#pragma once
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"

// ============================================================
// File: AlphaPnLTracker.hpp  (demo/cpp/)
// Role: PnL attribution for the variance alpha strategy.
//
// Depends only on stable event semantics (FillEvent,
// OptionMidQuoteEvent) — no reference to synthetic feed or
// how fills are generated. Safe to use with a real market feed.
//
// StrategyPnLBreakdown:
//   option_mtm      — unrealized: mid_price × held_qty per tick
//   delta_hedge_pnl — realized: accumulated from FillEvents
//                     where producer == "hedge_order"
//   transaction_cost — half-spread × |qty| per alpha fill
//   total_pnl       — sum of the above
// ============================================================

namespace omm::demo {

struct StrategyPnLBreakdown {
    double option_mtm       = 0.0;
    double delta_hedge_pnl  = 0.0;
    double transaction_cost = 0.0;
    double total_pnl        = 0.0;
};

class AlphaPnLTracker {
public:
    explicit AlphaPnLTracker(
        std::shared_ptr<events::EventBus> bus,
        double half_spread = 0.05  // 固定半价差估算交易成本
    ) : bus_(std::move(bus)), half_spread_(half_spread) {}

    void register_handlers() {
        bus_->subscribe<events::FillEvent>(
            [this](const events::FillEvent& e) { on_fill(e); }
        );
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_quote(e); }
        );
    }

    StrategyPnLBreakdown breakdown() const {
        StrategyPnLBreakdown b;
        b.option_mtm       = option_mtm_;
        b.delta_hedge_pnl  = delta_hedge_pnl_;
        b.transaction_cost = transaction_cost_;
        b.total_pnl        = option_mtm_ + delta_hedge_pnl_ - transaction_cost_;
        return b;
    }

    void print_summary() const {
        auto b = breakdown();
        std::cout << "\n┌─────────────────────────────────────────────────���────────┐\n"
                  << "│  Strategy PnL Attribution                                │\n"
                  << "└──────────────────────────────────────────────────────────┘\n"
                  << std::fixed << std::setprecision(2)
                  << "  Option MTM (unrealized):  $" << b.option_mtm      << "\n"
                  << "  Delta hedge PnL (realized):$" << b.delta_hedge_pnl << "\n"
                  << "  Transaction cost:         -$" << b.transaction_cost << "\n"
                  << "  ──────────────────���──────────────────────\n"
                  << "  Total PnL:                 $" << b.total_pnl       << "\n\n";
    }

private:
    void on_fill(const events::FillEvent& e) {
        int signed_qty = (e.side == events::Side::Buy) ? e.fill_qty : -e.fill_qty;

        if (e.producer == "hedge_order") {
            // 对冲成交：累计已实现 delta hedge PnL
            // 简化：对冲成交本身的成本由 transaction_cost 捕获
            delta_hedge_pnl_ -= signed_qty * e.fill_price;
            transaction_cost_ += half_spread_ * std::abs(e.fill_qty);
        } else if (e.producer == "alpha_exec") {
            // Alpha 入场/离场成交：更新持仓，累计交易成本
            held_qty_  += signed_qty;
            avg_cost_  += signed_qty * e.fill_price;
            transaction_cost_ += half_spread_ * std::abs(e.fill_qty);
        }
    }

    void on_quote(const events::OptionMidQuoteEvent& e) {
        // 用最新中间价重算 MTM（持仓 × 当前中间价 - 持仓成本）
        if (held_qty_ != 0)
            option_mtm_ = held_qty_ * e.mid_price - avg_cost_;
    }

    std::shared_ptr<events::EventBus> bus_;
    double half_spread_;

    int    held_qty_         = 0;
    double avg_cost_         = 0.0;
    double option_mtm_       = 0.0;
    double delta_hedge_pnl_  = 0.0;
    double transaction_cost_ = 0.0;
};

} // namespace omm::demo
