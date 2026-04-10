#pragma once
#include <cmath>
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
        , default_half_spread_(half_spread)
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

    // ── Per-day session boundary hook ────────────────────────────
    // Call this when the trading date changes (from the on_row callback
    // in alpha_main.cpp). Snapshots the current-day incremental PnL and
    // resets the day-start baseline.
    void on_session_end(const std::string& date) {
        double day_opt   = option_mtm_       - day_start_option_mtm_;
        double day_hedge = delta_hedge_pnl_  - day_start_hedge_pnl_;
        double day_cost  = transaction_cost_ - day_start_txn_cost_;
        double day_total = day_opt + day_hedge - day_cost;

        day_records_.push_back({date, day_opt, day_hedge, day_cost,
                                 n_fills_today_, day_total});
        n_fills_today_        = 0;
        day_start_option_mtm_ = option_mtm_;
        day_start_hedge_pnl_  = delta_hedge_pnl_;
        day_start_txn_cost_   = transaction_cost_;
    }

    void print_summary() const {
        auto b = breakdown();

        // 波动率风险溢价：平均 IV - 平均 RV5
        double avg_iv  = (iv_obs_count_ > 0)  ? iv_obs_sum_  / iv_obs_count_  : 0.0;
        double avg_rv5 = (rv5_count_ > 0)      ? rv5_sum_     / rv5_count_     : 0.0;
        double vrp     = avg_iv - avg_rv5;

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
                  << "  Total PnL:                  $" << b.total_pnl       << "\n\n"
                  << std::setprecision(4)
                  << "  Vol Risk Premium (Databento OPRA)\n"
                  << "  ──────────────────────────────────────────────────────\n"
                  << "  平均隐含波动率 (IV):         " << avg_iv  * 100.0 << "%\n"
                  << "  平均已实现波动率 5-bar (RV): " << avg_rv5 * 100.0 << "%\n"
                  << "  波动率风险溢价 (IV - RV):    " << vrp     * 100.0 << "%\n\n";

        if (!day_records_.empty())
            print_stability_report();
    }

    void print_stability_report() const {
        if (day_records_.empty()) return;

        int N = static_cast<int>(day_records_.size());

        // Compute daily PnL stats
        double sum = 0.0, sum_sq = 0.0;
        double min_pnl = day_records_[0].total_pnl;
        double max_pnl = day_records_[0].total_pnl;
        double total_gross_pnl  = 0.0;
        double total_txn        = 0.0;
        double total_hedge_pnl  = 0.0;
        double total_option_mtm = 0.0;
        double total_trades     = 0.0;

        for (const auto& d : day_records_) {
            sum     += d.total_pnl;
            sum_sq  += d.total_pnl * d.total_pnl;
            min_pnl  = std::min(min_pnl, d.total_pnl);
            max_pnl  = std::max(max_pnl, d.total_pnl);
            total_gross_pnl  += std::abs(d.option_mtm + d.delta_hedge_pnl);
            total_txn        += d.txn_cost;
            total_hedge_pnl  += d.delta_hedge_pnl;
            total_option_mtm += d.option_mtm;
            total_trades     += d.n_fills;
        }
        double mean_pnl = sum / N;
        double var_pnl  = sum_sq / N - mean_pnl * mean_pnl;
        double std_pnl  = (var_pnl > 0.0) ? std::sqrt(var_pnl) : 0.0;
        double sharpe   = (std_pnl > 0.0) ? mean_pnl / std_pnl : 0.0;

        // Attribution shares (of gross absolute PnL)
        double opt_share   = (total_gross_pnl > 0.0) ? total_option_mtm / total_gross_pnl : 0.0;
        double hedge_share = (total_gross_pnl > 0.0) ? total_hedge_pnl  / total_gross_pnl : 0.0;
        double cost_drag   = (total_gross_pnl > 0.0) ? total_txn        / total_gross_pnl : 0.0;
        double turnover    = total_trades / N;

        // Hedge residual distribution (per-day errors stored as option_mtm - delta_pnl proxy)
        // We store total_pnl per day; use that for the distribution
        std::vector<double> pnls;
        pnls.reserve(N);
        for (const auto& d : day_records_) pnls.push_back(d.total_pnl);
        std::sort(pnls.begin(), pnls.end());
        double p5  = pnls[static_cast<int>(0.05 * N)];
        double p50 = pnls[N / 2];
        double p95 = pnls[static_cast<int>(0.95 * N)];

        std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n"
                  << "║  Multi-Day Stability Report — " << N << " days"
                  << std::string(std::max(0, 32 - static_cast<int>(std::to_string(N).size())), ' ') << "║\n"
                  << "╠══════════════════════════════════════════════════════════════╣\n"
                  << std::fixed << std::setprecision(2)
                  << "║  Daily PnL        mean  $" << std::setw(8) << mean_pnl
                  << "  ±  std  $" << std::setw(8) << std_pnl << "          ║\n"
                  << "║               [min $"   << std::setw(8) << min_pnl
                  << "  max $"   << std::setw(8) << max_pnl << "]          ║\n"
                  << "║  Sharpe (daily)   " << std::setw(8) << std::setprecision(3) << sharpe
                  << std::string(37, ' ') << "║\n"
                  << "╠══════════════════════════════════════════════════════════════╣\n"
                  << std::setprecision(1)
                  << "║  Option MTM share     " << std::setw(6) << opt_share   * 100.0 << "%"
                  << std::string(35, ' ') << "║\n"
                  << "║  Delta hedge share    " << std::setw(6) << hedge_share * 100.0 << "%"
                  << std::string(35, ' ') << "║\n"
                  << "║  Txn cost drag        " << std::setw(6) << cost_drag   * 100.0 << "%"
                  << std::string(35, ' ') << "║\n"
                  << "║  Turnover             " << std::setw(6) << std::setprecision(1) << turnover
                  << " fills/day" << std::string(26, ' ') << "║\n"
                  << "╠══════════════════════════════════════════════════════════════╣\n"
                  << std::setprecision(2)
                  << "║  Daily PnL dist   P5 $" << std::setw(8) << p5
                  << "  P50 $" << std::setw(8) << p50
                  << "  P95 $" << std::setw(7) << p95 << " ║\n"
                  << "╚══════════════════════════════════════════════════════════════╝\n\n";
    }

private:
    struct DayRecord {
        std::string date;
        double option_mtm;
        double delta_hedge_pnl;
        double txn_cost;
        int    n_fills;
        double total_pnl;
    };

    struct InstrumentPosition {
        int    qty      = 0;
        double cost     = 0.0;   // 累计成本（qty × fill_price 之和）
        double last_mid = 0.0;   // 最近一次中间价
        double last_theo = 0.0;  // 最近一次模型理论价（用于 MTM 基准）
    };

    void on_fill(const events::FillEvent& e) {
        int signed_qty = (e.side == events::Side::Buy) ? e.fill_qty : -e.fill_qty;
        ++n_fills_today_;

        if (e.producer == "hedge_order") {
            delta_hedge_pnl_ -= signed_qty * e.fill_price;
            transaction_cost_ += default_half_spread_ * std::abs(e.fill_qty);
        } else if (e.producer == "alpha_exec") {
            auto& pos = positions_[e.instrument_id];
            pos.qty  += signed_qty;
            pos.cost += signed_qty * e.fill_price;
            // 使用实际买卖价差（来自 Databento 行情），若无则用默认值
            auto it = half_spread_per_instrument_.find(e.instrument_id);
            double hs = (it != half_spread_per_instrument_.end())
                        ? it->second : default_half_spread_;
            transaction_cost_ += hs * std::abs(e.fill_qty);
        }
    }

    void on_quote(const events::OptionMidQuoteEvent& e) {
        // 更新实际买卖价差（半价差 = (ask - bid) / 2）
        if (e.ask_price > e.bid_price && e.bid_price > 0.0) {
            half_spread_per_instrument_[e.instrument_id] = (e.ask_price - e.bid_price) / 2.0;
        }

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

        // 累计 IV，用于最终报告波动率风险溢价
        if (iv.valid) {
            iv_obs_sum_   += sigma_impl_now;
            iv_obs_count_ += 1;
        }

        // 滚动已实现波动率（5-bar，年化），用于波动率风险溢价报告
        if (last_spot_ > 0.0) {
            double log_ret = std::log(new_spot / last_spot_);
            rv_ring_[rv_idx_ % 5] = log_ret;
            ++rv_idx_;
            if (rv_idx_ >= 5) {
                double sum_sq = 0.0;
                for (double r : rv_ring_) sum_sq += r * r;
                double rv5 = std::sqrt(sum_sq / 5.0 * 252.0 * 390.0);
                rv5_sum_   += rv5;
                rv5_count_ += 1;
            }
        }

        // dt 估算：tick-to-tick（使用固定年化近似，生产环境替换为真实时间差）
        constexpr double DT_ANNUAL = 1.0 / (252.0 * 390.0);  // ~1分钟级tick近似

        for (auto& [fill_id, opt_ptr] : options_map_) {
            auto& pos = positions_[fill_id];   // fill ID matches what StrategyController uses
            if (pos.qty == 0) continue;

            // Use price_at_iv() with T_sim from extractor when available to avoid
            // system_clock::now() T calculation — options may have expired in wall-clock
            // time during historical replay, which drives T→0 and theta to blow up.
            domain::PriceResult pr;
            if (iv.valid && iv.time_to_expiry > 0.0) {
                const bool is_call =
                    (opt_ptr->option_type() == domain::OptionType::Call);
                pr = engine_->price_at_iv(new_spot, opt_ptr->strike(),
                                          iv.time_to_expiry, sigma_impl_now, is_call);
            } else {
                pr = engine_->price(*opt_ptr, new_spot);
            }

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
    double                                               default_half_spread_;

    std::unordered_map<std::string, InstrumentPosition>  positions_;
    std::unordered_map<std::string, double>              half_spread_per_instrument_;

    double option_mtm_        = 0.0;
    double delta_pnl_         = 0.0;
    double gamma_pnl_         = 0.0;
    double vega_pnl_          = 0.0;
    double theta_pnl_         = 0.0;
    double delta_hedge_pnl_   = 0.0;
    double transaction_cost_  = 0.0;

    double last_spot_         = 150.0;
    double sigma_impl_prev_   = 0.25;  // BS vol 初始值（冷启动）

    // 波动率风险溢价跟踪（Databento OPRA 数据支持）
    double iv_obs_sum_        = 0.0;
    int    iv_obs_count_      = 0;
    double rv5_sum_           = 0.0;
    int    rv5_count_         = 0;
    double rv_ring_[5]        = {};    // 环形缓冲区：最近 5 bar 对数收益
    int    rv_idx_            = 0;

    // Per-day stability tracking
    std::vector<DayRecord> day_records_;
    int    n_fills_today_         = 0;
    double day_start_option_mtm_  = 0.0;
    double day_start_hedge_pnl_   = 0.0;
    double day_start_txn_cost_    = 0.0;
};

} // namespace omm::demo
