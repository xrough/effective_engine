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
// File: ExtendedPnLTracker.hpp  (demo/cpp/)
// Role: Full Greeks PnL attribution with Vanna and Volga terms,
//       designed for the alpha PnL test dual-hedger comparison.
//
// Extends the standard 7-Greek breakdown with:
//   vanna_pnl — ∂²V/(∂S∂σ) · ΔS · Δσ   (skew attribution)
//   volga_pnl — ½ · ∂²V/∂σ² · (Δσ)²   (curvature attribution)
//
// Vanna and Volga computed via analytical Black-Scholes formulas:
//   Vanna_BS = −N′(d1) · d2 / σ
//   Volga_BS =  Vega  · d1 · d2 / σ  = S·√T·N′(d1)·d1·d2/σ
//
// Static print_comparison() shows side-by-side table for two passes.
// ============================================================

namespace omm::demo {

// ── Black-Scholes helpers for second-order Greeks ────────────
namespace bs_detail {
inline double d1(double S, double K, double T, double sigma, double r) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}
inline double norm_pdf(double x) {
    static constexpr double INV_SQRT2PI = 0.3989422804014327;
    return INV_SQRT2PI * std::exp(-0.5 * x * x);
}
// Vanna: same sign for call and put
inline double vanna(double S, double K, double T, double sigma, double r) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1v = d1(S, K, T, sigma, r);
    double d2v = d1v - sigma * std::sqrt(T);
    return -norm_pdf(d1v) * d2v / sigma;
}
// Volga: same sign for call and put
inline double volga(double S, double K, double T, double sigma, double r) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double d1v = d1(S, K, T, sigma, r);
    double d2v = d1v - sigma * std::sqrt(T);
    double vega = S * std::sqrt(T) * norm_pdf(d1v);
    if (std::abs(sigma) < 1e-8) return 0.0;
    return vega * d1v * d2v / sigma;
}
} // namespace bs_detail

// ── Extended breakdown ────────────────────────────────────────
struct ExtendedPnLBreakdown {
    // Standard Greeks
    double option_mtm       = 0.0;
    double delta_pnl        = 0.0;
    double gamma_pnl        = 0.0;
    double vega_pnl         = 0.0;
    double theta_pnl        = 0.0;
    double delta_hedge_pnl  = 0.0;
    double hedge_resid      = 0.0;
    double transaction_cost = 0.0;
    double total_pnl        = 0.0;
    // Extended second-order Greeks
    double vanna_pnl        = 0.0;   // ∂²V/(∂S∂σ) · ΔS · Δσ
    double volga_pnl        = 0.0;   // ½ · ∂²V/∂σ² · (Δσ)²
    // Vol risk premium
    double avg_iv           = 0.0;
    double avg_rv5          = 0.0;
    std::string hedger_label;
};

// ── Tracker class ─────────────────────────────────────────────
class ExtendedPnLTracker {
public:
    using OptionEntry = std::pair<std::string, std::shared_ptr<domain::Option>>;

    ExtendedPnLTracker(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<domain::IPricingEngine>              engine,
        std::vector<OptionEntry>                             options_map,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::string                                          hedger_label = "DeltaHedger",
        double                                               rate         = 0.05,
        double                                               half_spread  = 0.05
    )
        : bus_(std::move(bus))
        , engine_(std::move(engine))
        , options_map_(std::move(options_map))
        , extractor_(std::move(extractor))
        , hedger_label_(std::move(hedger_label))
        , rate_(rate)
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

    ExtendedPnLBreakdown breakdown() const {
        ExtendedPnLBreakdown b;
        b.option_mtm       = option_mtm_;
        b.delta_pnl        = delta_pnl_;
        b.gamma_pnl        = gamma_pnl_;
        b.vega_pnl         = vega_pnl_;
        b.theta_pnl        = theta_pnl_;
        b.vanna_pnl        = vanna_pnl_;
        b.volga_pnl        = volga_pnl_;
        b.delta_hedge_pnl  = delta_hedge_pnl_;
        b.hedge_resid      = delta_hedge_pnl_ - delta_pnl_;
        b.transaction_cost = transaction_cost_;
        b.total_pnl        = option_mtm_ + delta_hedge_pnl_ - transaction_cost_;
        b.avg_iv  = (iv_count_  > 0) ? iv_sum_  / iv_count_  : 0.0;
        b.avg_rv5 = (rv5_count_ > 0) ? rv5_sum_ / rv5_count_ : 0.0;
        b.hedger_label = hedger_label_;
        return b;
    }

    // ── Side-by-side comparison table ────────────────────────
    // c == nullptr: 2-column table (backward compatible)
    // c != nullptr: 5-column table (a | b | Δ(b-a) | c | Δ(c-a))
    // Key metric: Hedge Residual — should decrease  a > b > c  if hedgers improve.
    static void print_comparison(const ExtendedPnLBreakdown& a,
                                 const ExtendedPnLBreakdown& b,
                                 const ExtendedPnLBreakdown* c = nullptr) {
        const bool three_col = (c != nullptr);

        auto row = [&](const char* label, double va, double vb, double vc = 0.0) {
            std::cout << "  " << std::left  << std::setw(22) << label
                      << std::right << std::fixed << std::setprecision(2)
                      << std::setw(12) << va
                      << std::setw(14) << vb
                      << std::setw(13) << (vb - va);
            if (three_col)
                std::cout << std::setw(14) << vc
                          << std::setw(13) << (vc - va);
            std::cout << "\n";
        };

        const std::string sep = three_col
            ? "  ─────────────────────────────────────────────────────────────────────────\n"
            : "  ─────────────────────────────────────────────────────────────\n";

        if (three_col) {
            std::cout << "\n┌─────────────────────────────────────────────────────────────────────────┐\n"
                      << "│  Alpha PnL — Three-Strategy Hedge Attribution (BS / Rough / BSDE)       │\n"
                      << "├─────────────────────────────────────────────────────────────────────────┤\n";
        } else {
            std::cout << "\n┌─────────────────────────────────────────────────────────────────┐\n"
                      << "│   Alpha PnL Comparison — DeltaHedger vs NeuralBSDEHedger         │\n"
                      << "├─────────────────────────────────────────────────────────────────┤\n";
        }

        std::cout << "  " << std::left << std::setw(22) << ""
                  << std::right << std::setw(12) << a.hedger_label
                  << std::setw(14) << b.hedger_label
                  << std::setw(13) << "Δ(b-a)";
        if (three_col)
            std::cout << std::setw(14) << c->hedger_label
                      << std::setw(13) << "Δ(c-a)";
        std::cout << "\n" << sep;

        row("Option MTM ($):",     a.option_mtm,      b.option_mtm,      three_col ? c->option_mtm      : 0.0);
        row("Δ PnL ($):",          a.delta_pnl,       b.delta_pnl,       three_col ? c->delta_pnl       : 0.0);
        row("Γ PnL ($):",          a.gamma_pnl,       b.gamma_pnl,       three_col ? c->gamma_pnl       : 0.0);
        row("ν PnL ($):",          a.vega_pnl,        b.vega_pnl,        three_col ? c->vega_pnl        : 0.0);
        row("Vanna PnL ($):",      a.vanna_pnl,       b.vanna_pnl,       three_col ? c->vanna_pnl       : 0.0);
        row("Volga PnL ($):",      a.volga_pnl,       b.volga_pnl,       three_col ? c->volga_pnl       : 0.0);
        row("θ PnL ($):",          a.theta_pnl,       b.theta_pnl,       three_col ? c->theta_pnl       : 0.0);
        row("Hedge PnL ($):",      a.delta_hedge_pnl, b.delta_hedge_pnl, three_col ? c->delta_hedge_pnl : 0.0);
        row("Hedge Residual ($):", a.hedge_resid,     b.hedge_resid,     three_col ? c->hedge_resid     : 0.0);
        row("Txn Cost ($):",       a.transaction_cost,b.transaction_cost,three_col ? c->transaction_cost: 0.0);
        std::cout << sep;
        row("Total PnL ($):",      a.total_pnl,       b.total_pnl,       three_col ? c->total_pnl       : 0.0);

        std::cout << "\n"
                  << std::setprecision(4)
                  << "  Vol Risk Premium\n" << sep
                  << "  Avg IV:      " << a.avg_iv  * 100.0 << "%  (all passes share market data)\n"
                  << "  Avg RV5:     " << a.avg_rv5 * 100.0 << "%\n"
                  << "  VRP (IV-RV): " << (a.avg_iv - a.avg_rv5) * 100.0 << "%\n";
        if (three_col)
            std::cout << "└─────────────────────────────────────────────────────────────────────────┘\n\n";
        else
            std::cout << "└─────────────────────────────────────────────────────────────────┘\n\n";
    }

private:
    struct InstrumentPosition {
        int    qty       = 0;
        double cost      = 0.0;
        double last_mid  = 0.0;
    };

    void on_fill(const events::FillEvent& e) {
        int signed_qty = (e.side == events::Side::Buy) ? e.fill_qty : -e.fill_qty;
        if (e.producer == "hedge_order") {
            delta_hedge_pnl_ -= signed_qty * e.fill_price;
            transaction_cost_ += default_half_spread_ * std::abs(e.fill_qty);
        } else if (e.producer == "alpha_exec") {
            auto& pos = positions_[e.instrument_id];
            pos.qty  += signed_qty;
            pos.cost += signed_qty * e.fill_price;
            auto it = half_spread_.find(e.instrument_id);
            double hs = (it != half_spread_.end()) ? it->second : default_half_spread_;
            transaction_cost_ += hs * std::abs(e.fill_qty);
        }
    }

    void on_quote(const events::OptionMidQuoteEvent& e) {
        if (e.ask_price > e.bid_price && e.bid_price > 0.0)
            half_spread_[e.instrument_id] = (e.ask_price - e.bid_price) / 2.0;
        auto it = positions_.find(e.instrument_id);
        if (it != positions_.end())
            it->second.last_mid = e.mid_price;
        option_mtm_ = 0.0;
        for (const auto& [id, pos] : positions_)
            if (pos.qty != 0 && pos.last_mid > 0.0)
                option_mtm_ += pos.qty * pos.last_mid - pos.cost;
    }

    void on_market_data(const events::MarketDataEvent& e) {
        double new_spot = e.underlying_price;
        double dS       = new_spot - last_spot_;

        auto iv = extractor_->last_point();
        double sigma_now = iv.valid ? iv.atm_implied_vol : sigma_prev_;
        double d_sigma   = sigma_now - sigma_prev_;
        // Use simulated T from the extractor (CSV time_to_expiry) for
        // Vanna/Volga — avoids T→0 blowup when replaying historical data
        // where option expiry has already passed system clock.
        double T_sim = iv.valid ? std::max(iv.time_to_expiry, 1e-4) : last_T_sim_;
        last_T_sim_ = T_sim;

        if (iv.valid) { iv_sum_ += sigma_now; ++iv_count_; }

        // Rolling 5-bar realized vol for VRP report
        if (last_spot_ > 0.0) {
            double lr = std::log(new_spot / last_spot_);
            rv_ring_[rv_idx_ % 5] = lr;
            ++rv_idx_;
            if (rv_idx_ >= 5) {
                double ss = 0.0;
                for (double r : rv_ring_) ss += r * r;
                rv5_sum_ += std::sqrt(ss / 5.0 * 252.0 * 390.0);
                ++rv5_count_;
            }
        }

        constexpr double DT_ANNUAL = 1.0 / (252.0 * 390.0);

        for (auto& [fill_id, opt_ptr] : options_map_) {
            auto& pos = positions_[fill_id];
            if (pos.qty == 0) continue;

            // Use price_at_iv() with T_sim from extractor to avoid system_clock::now()
            // driving T→0 for historically-expired options (theta blowup).
            domain::PriceResult pr;
            if (iv.valid && T_sim > 0.0) {
                const bool is_call =
                    (opt_ptr->option_type() == domain::OptionType::Call);
                pr = engine_->price_at_iv(new_spot, opt_ptr->strike(),
                                          T_sim, sigma_now, is_call);
            } else {
                pr = engine_->price(*opt_ptr, new_spot);
            }
            delta_pnl_ += pos.qty * pr.delta * dS;
            gamma_pnl_ += pos.qty * 0.5 * pr.gamma * dS * dS;
            vega_pnl_  += pos.qty * pr.vega * d_sigma;
            theta_pnl_ += pos.qty * pr.theta * DT_ANNUAL;

            // Vanna and Volga via analytical BS
            // Use simulated T (from IV extractor's time_to_expiry) to avoid
            // T→0 blowup when replaying historical data that has already expired.
            double K  = opt_ptr->strike();
            double va = bs_detail::vanna(new_spot, K, T_sim, sigma_now, rate_);
            double vo = bs_detail::volga(new_spot, K, T_sim, sigma_now, rate_);
            vanna_pnl_ += pos.qty * va * dS * d_sigma;
            volga_pnl_ += pos.qty * 0.5 * vo * d_sigma * d_sigma;
        }

        last_spot_  = new_spot;
        sigma_prev_ = sigma_now;
    }

    // ── Members ───────────────────────────────────────────────
    std::shared_ptr<events::EventBus>                    bus_;
    std::shared_ptr<domain::IPricingEngine>              engine_;
    std::vector<OptionEntry>                             options_map_;
    std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor_;
    std::string                                          hedger_label_;
    double                                               rate_;
    double                                               default_half_spread_;

    std::unordered_map<std::string, InstrumentPosition>  positions_;
    std::unordered_map<std::string, double>              half_spread_;

    double option_mtm_       = 0.0;
    double delta_pnl_        = 0.0;
    double gamma_pnl_        = 0.0;
    double vega_pnl_         = 0.0;
    double theta_pnl_        = 0.0;
    double vanna_pnl_        = 0.0;
    double volga_pnl_        = 0.0;
    double delta_hedge_pnl_  = 0.0;
    double transaction_cost_ = 0.0;

    double last_spot_    = 150.0;
    double sigma_prev_   = 0.25;
    double last_T_sim_   = 0.0822;  // ~30D fallback (updated from extractor)

    double iv_sum_   = 0.0;  int iv_count_  = 0;
    double rv5_sum_  = 0.0;  int rv5_count_ = 0;
    double rv_ring_[5] = {};
    int    rv_idx_     = 0;
};

} // namespace omm::demo
