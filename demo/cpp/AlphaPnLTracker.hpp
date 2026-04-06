#pragma once
#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <chrono>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "core/domain/Instrument.hpp"
#include "core/analytics/PricingEngine.hpp"
#include "core/analytics/ImpliedVarianceExtractor.hpp"

// ============================================================
// File: AlphaPnLTracker.hpp  (demo/cpp/)
// Role: Full Greeks PnL attribution for the variance alpha strategy.
//
// Depends only on stable event semantics — safe to use with a real feed.
//
// StrategyPnLBreakdown (6 buckets):
//   option_mtm      — unrealized MTM from repricing at each tick
//   delta_pnl       — Δ·ΔS  (spot-move attribution)
//   gamma_pnl       — ½Γ·ΔS²  (convexity)
//   vega_pnl        — ν·Δσ_impl  (vol-move, using consistent σ from extractor)
//   theta_pnl       — θ·Δt  (time decay)
//   hedge_resid     — actual_hedge_pnl − delta_pnl (hedger over/under-capture)
//   transaction_cost — half_spread × |qty| per fill
// ============================================================

namespace omm::demo {

struct StrategyPnLBreakdown {
    double option_mtm       = 0.0;
    double delta_pnl        = 0.0;
    double gamma_pnl        = 0.0;
    double vega_pnl         = 0.0;
    double theta_pnl        = 0.0;
    double hedge_resid      = 0.0;
    double delta_hedge_pnl  = 0.0;   // 保留：对冲成交的已实现 PnL
    double transaction_cost = 0.0;
    double total_pnl        = 0.0;
};

class AlphaPnLTracker {
public:
    // options_map: vector of {fill_id, option_ptr} pairs
    // fill_id is the instrument_id used in FillEvent (e.g. "ATM_CALL")
    // option_ptr is used for repricing Greeks — its generated ID may differ
    using OptionEntry = std::pair<std::string, std::shared_ptr<domain::Option>>;

    AlphaPnLTracker(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<domain::IPricingEngine>              engine,
        std::vector<OptionEntry>                             options_map,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        double half_spread = 0.05
    )
        : bus_(std::move(bus))
        , engine_(std::move(engine))
        , options_map_(std::move(options_map))
        , extractor_(std::move(extractor))
        , half_spread_(half_spread)
    {}

    void register_handlers() {
        bus_->subscribe<events::FillEvent>(
            [this](const events::FillEvent& e) { on_fill(e); }
        );
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_quote(e); }
        );
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { on_market_data(e); }
        );
    }

    StrategyPnLBreakdown breakdown() const {
        StrategyPnLBreakdown b;
        b.option_mtm      = option_mtm_;
        b.delta_pnl       = delta_pnl_;
        b.gamma_pnl       = gamma_pnl_;
        b.vega_pnl        = vega_pnl_;
        b.theta_pnl       = theta_pnl_;
        b.delta_hedge_pnl = delta_hedge_pnl_;
        b.hedge_resid     = delta_hedge_pnl_ - delta_pnl_;
        b.transaction_cost = transaction_cost_;
        b.total_pnl = option_mtm_ + delta_hedge_pnl_ - transaction_cost_;
        return b;
    }

    void print_summary() const {
        auto b = breakdown();
        std::cout << "\n┌──────────────────────────────────────────────────────────┐\n"
                  << "│  Strategy PnL Attribution (Greeks Decomposition)         │\n"
                  << "└──────────────────────────────────────────────────────────┘\n"
                  << std::fixed << std::setprecision(2)
                  << "  Option MTM (unrealized):    $" << b.option_mtm      << "\n"
                  << "  Δ PnL (spot move):          $" << b.delta_pnl       << "\n"
                  << "  Γ PnL (convexity):          $" << b.gamma_pnl       << "\n"
                  << "  ν PnL (vol move):           $" << b.vega_pnl        << "\n"
                  << "  θ PnL (time decay):         $" << b.theta_pnl       << "\n"
                  << "  Delta hedge PnL (realized): $" << b.delta_hedge_pnl << "\n"
                  << "  Hedge residual (Δ-capture): $" << b.hedge_resid     << "\n"
                  << "  Transaction cost:          -$" << b.transaction_cost << "\n"
                  << "  ──────────────────────────────────────────────────────\n"
                  << "  Total PnL:                  $" << b.total_pnl       << "\n\n";
    }

private:
    struct InstrumentPosition {
        int    qty      = 0;
        double cost     = 0.0;   // 累计成本（qty × fill_price 之和）
        double last_mid = 0.0;   // 最近一次中间价
        double last_theo = 0.0;  // 最近一次模型理论价（用于 MTM 基准）
    };

    void on_fill(const events::FillEvent& e) {
        int signed_qty = (e.side == events::Side::Buy) ? e.fill_qty : -e.fill_qty;

        if (e.producer == "hedge_order") {
            delta_hedge_pnl_ -= signed_qty * e.fill_price;
            transaction_cost_ += half_spread_ * std::abs(e.fill_qty);
        } else if (e.producer == "alpha_exec") {
            auto& pos = positions_[e.instrument_id];
            pos.qty  += signed_qty;
            pos.cost += signed_qty * e.fill_price;
            transaction_cost_ += half_spread_ * std::abs(e.fill_qty);
        }
    }

    void on_quote(const events::OptionMidQuoteEvent& e) {
        auto it = positions_.find(e.instrument_id);
        if (it != positions_.end()) {
            it->second.last_mid = e.mid_price;
        }
        // 重算所有持仓的 MTM
        option_mtm_ = 0.0;
        for (const auto& [id, pos] : positions_) {
            if (pos.qty != 0 && pos.last_mid > 0.0)
                option_mtm_ += pos.qty * pos.last_mid - pos.cost;
        }
    }

    void on_market_data(const events::MarketDataEvent& e) {
        double new_spot = e.underlying_price;
        double dS       = new_spot - last_spot_;

        // 当前 σ_impl（来自 ImpliedVarianceExtractor，保持一致性）
        auto iv = extractor_->last_point();
        double sigma_impl_now = iv.valid ? iv.atm_implied_vol : sigma_impl_prev_;

        // dt 估算：tick-to-tick（使用固定年化近似，生产环境替换为真实时间差）
        constexpr double DT_ANNUAL = 1.0 / (252.0 * 390.0);  // ~1分钟级tick近似

        for (auto& [fill_id, opt_ptr] : options_map_) {
            auto& pos = positions_[fill_id];   // fill ID matches what StrategyController uses
            if (pos.qty == 0) continue;

            auto pr = engine_->price(*opt_ptr, new_spot);

            // Δ PnL
            delta_pnl_ += pos.qty * pr.delta * dS;

            // Γ PnL
            gamma_pnl_ += pos.qty * 0.5 * pr.gamma * dS * dS;

            // ν PnL：使用一致的 Δσ_impl（不混入 Γ/θ 效应）
            double d_sigma = sigma_impl_now - sigma_impl_prev_;
            vega_pnl_ += pos.qty * pr.vega * d_sigma;

            // θ PnL
            theta_pnl_ += pos.qty * pr.theta * DT_ANNUAL;
        }

        last_spot_       = new_spot;
        sigma_impl_prev_ = sigma_impl_now;
    }

    std::shared_ptr<events::EventBus>                    bus_;
    std::shared_ptr<domain::IPricingEngine>              engine_;
    std::vector<OptionEntry>                             options_map_;
    std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor_;
    double                                               half_spread_;

    std::unordered_map<std::string, InstrumentPosition>  positions_;

    double option_mtm_        = 0.0;
    double delta_pnl_         = 0.0;
    double gamma_pnl_         = 0.0;
    double vega_pnl_          = 0.0;
    double theta_pnl_         = 0.0;
    double delta_hedge_pnl_   = 0.0;
    double transaction_cost_  = 0.0;

    double last_spot_         = 150.0;
    double sigma_impl_prev_   = 0.25;  // BS vol 初始值（冷启动）
};

} // namespace omm::demo
