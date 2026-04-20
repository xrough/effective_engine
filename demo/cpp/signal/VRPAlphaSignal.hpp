#pragma once
#include <deque>
#include <cmath>
#include <memory>
#include <numeric>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "modules/buyer/IAlphaSignal.hpp"
#include "core/analytics/RoughVolPricingEngine.hpp"
#include "core/analytics/ImpliedVarianceExtractor.hpp"
#include "analytics/SmileVarianceExtractor.hpp"

// ============================================================
// File: VRPAlphaSignal.hpp  (demo/cpp/signal/)
// Role: Volatility Risk Premium signal — rolling z-score of IV²−RV².
//
// Signal logic:
//   rv5       = sqrt( sum(r²_last5) / 5 * 252 * 390 )   (5-bar annualised RV)
//   vrp_spread = IV² − rv5²
//   zscore     = (vrp_spread − rolling_mean) / rolling_std
//
// Positive zscore → IV is rich relative to recent realised vol →
// vol is expensive; a long-vol strategy should wait.  In a regime
// where the strategy SELLS vol (or buys straddles to collect premium),
// positive VRP (IV > RV) confirms the structural edge.
//
// Why this matters: the existing pipeline shows VRP = +4.93% over the
// 127-day SPY panel but the variance/curvature signal ignores VRP
// completely.  This signal gates entries on whether VRP is elevated
// above its own rolling mean — not just whether IV is rich vs the
// rough-vol forward variance forecast.
//
// Data: uses atm_implied_vol from ImpliedVarianceExtractor (set each
// bar by HistoricalChainAdapter) and log-returns from MarketDataEvent.
// ============================================================

namespace omm::buyer {

struct VRPSignalConfig {
    AlphaSignalConfig base;   // window=50, z_entry=1.5, z_exit=0.5
    int rv_window = 5;        // bars for RV estimation (5-bar, matches AlphaPnLTracker)
};

class VRPAlphaSignal : public IAlphaSignal {
public:
    VRPAlphaSignal(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       rough_engine,
        VRPSignalConfig cfg = {}
    ) : bus_(std::move(bus))
      , extractor_(std::move(extractor))
      , cfg_(cfg) {}

    void register_handlers() override {
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { on_market_data(e); }
        );
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_option_quote(e); }
        );
    }

    void set_smile_extractor(std::shared_ptr<omm::demo::SmileVarianceExtractor> se) {
        smile_ex_ = std::move(se);
    }

    // MarketDataEvent: update RV ring buffer with each spot log-return.
    void on_market_data(const events::MarketDataEvent& e) override {
        double spot = e.underlying_price;
        if (last_spot_ > 0.0) {
            double lr = std::log(spot / last_spot_);
            rv_ring_[rv_idx_ % 22] = lr;
            ++rv_idx_;
        }
        last_spot_ = spot;
    }

    void on_option_quote(const events::OptionMidQuoteEvent& e) {
        analytics::ImpliedVariancePoint iv = extractor_->last_point();
        if (!iv.valid || iv.atm_implied_vol <= 0.0) return;

        // Compute 5-bar realised variance (annualised)
        double rv5 = 0.0;
        if (rv_idx_ >= cfg_.rv_window) {
            double ss = 0.0;
            for (int i = 0; i < cfg_.rv_window; ++i)
                ss += rv_ring_[(rv_idx_ - 1 - i) % 22]
                    * rv_ring_[(rv_idx_ - 1 - i) % 22];
            rv5 = std::sqrt(ss / cfg_.rv_window * 252.0 * 390.0);
        }

        // VIX-style model-free variance (default); fall back to ATM BS IV²
        double implied_var = (smile_ex_ && smile_ex_->has_vix())
                             ? smile_ex_->vix_variance()
                             : iv.atm_implied_vol * iv.atm_implied_vol;
        double rv5_sq = rv5 * rv5;
        double vrp    = implied_var - rv5_sq;

        vrp_history_.push_back(vrp);
        if ((int)vrp_history_.size() > cfg_.base.window)
            vrp_history_.pop_front();

        bool   window_full = ((int)vrp_history_.size() == cfg_.base.window);
        double zscore      = 0.0;

        if (window_full) {
            auto [mean, std_dev] = rolling_stats(vrp_history_);
            if (std_dev > 1e-12)
                zscore = (vrp - mean) / std_dev;
        }

        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full && rv_idx_ >= cfg_.rv_window;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = implied_var;  // VIX or IV² (for diagnostics)
        snap.raw_spread              = vrp;
        snap.zscore                  = zscore;
        snap.calibration_ok          = true;
        snap.atm_vega                = 0.0;
        snap.atm_spot                = e.underlying;

        bus_->publish(snap);
    }

private:
    static std::pair<double, double> rolling_stats(const std::deque<double>& h) {
        double mean = std::accumulate(h.begin(), h.end(), 0.0) / h.size();
        double var  = 0.0;
        for (double v : h) var += (v - mean) * (v - mean);
        return {mean, std::sqrt(var / h.size())};
    }

    std::shared_ptr<events::EventBus>                    bus_;
    std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor_;
    std::shared_ptr<omm::demo::SmileVarianceExtractor>   smile_ex_;
    VRPSignalConfig                                      cfg_;

    double rv_ring_[22] = {};   // ring buffer: enough for 22-bar HAR window
    int    rv_idx_      = 0;
    double last_spot_   = 0.0;

    std::deque<double> vrp_history_;
};

} // namespace omm::buyer
