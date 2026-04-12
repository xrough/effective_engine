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

// ============================================================
// File: HARRealizedVolSignal.hpp  (demo/cpp/signal/)
// Role: HAR-RV forecast signal — IV vs Heterogeneous Autoregressive
//       realized vol forecast, z-scored.
//
// HAR model (Corsi 2009):
//   RV_hat(t+1) = α + β₁·RV₁(t) + β₅·RV₅(t) + β₂₂·RV₂₂(t)
//
//   RV₁  = last-bar annualised realised vol    (1-bar horizon)
//   RV₅  = 5-bar average annualised RV          (daily horizon)
//   RV₂₂ = 22-bar average annualised RV         (monthly horizon)
//
// Signal:
//   har_spread = atm_iv − sqrt(RV_hat)    (IV premium over HAR forecast)
//   zscore     = (har_spread − rolling_mean) / rolling_std
//
// Why: Gate 2 rejected the *model-based* rough vol forecast (0/30 cells
// pass), but never tested a statistical RV model.  HAR-RV is the
// literature benchmark for RV forecasting (Corsi 2009, Andersen et al.
// 2003).  It adjusts for intraday / daily / weekly vol cycles, giving
// a time-varying RV baseline that is systematically more accurate than
// a flat rolling mean or a rough-model forward variance proxy.
//
// Default coefficients are Corsi (2009) canonical values:
//   β₁ = 0.40, β₅ = 0.20, β₂₂ = 0.30
// These can be overridden in HARSignalConfig if fitted on the SPY panel.
//
// Shares the 22-element rv_ring_ with VRPAlphaSignal when composed via
// CompositeAlphaSignal (each signal maintains its own ring independently).
// ============================================================

namespace omm::buyer {

struct HARSignalConfig {
    AlphaSignalConfig base;   // window=50, z_entry=1.5, z_exit=0.5
    // Corsi (2009) canonical HAR coefficients — no in-sample fitting required
    double alpha  = 0.0;
    double beta1  = 0.40;   // 1-bar (last tick) lag
    double beta5  = 0.20;   // 5-bar (daily) lag
    double beta22 = 0.30;   // 22-bar (monthly) lag
};

class HARRealizedVolSignal : public IAlphaSignal {
public:
    HARRealizedVolSignal(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       rough_engine,
        HARSignalConfig cfg = {}
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
        if (rv_idx_ < 22) return;   // need 22 bars to compute all HAR horizons

        constexpr double ANNUAL = 252.0 * 390.0;

        // RV₁: single-bar annualised vol
        double r1   = rv_ring_[(rv_idx_ - 1) % 22];
        double rv1  = std::sqrt(r1 * r1 * ANNUAL);

        // RV₅: 5-bar average annualised RV
        double ss5 = 0.0;
        for (int i = 0; i < 5; ++i)
            ss5 += rv_ring_[(rv_idx_ - 1 - i) % 22]
                 * rv_ring_[(rv_idx_ - 1 - i) % 22];
        double rv5 = std::sqrt(ss5 / 5.0 * ANNUAL);

        // RV₂₂: 22-bar average annualised RV
        double ss22 = 0.0;
        for (int i = 0; i < 22; ++i)
            ss22 += rv_ring_[(rv_idx_ - 1 - i) % 22]
                  * rv_ring_[(rv_idx_ - 1 - i) % 22];
        double rv22 = std::sqrt(ss22 / 22.0 * ANNUAL);

        // HAR forecast (in variance space, then sqrt back to vol)
        double rv_hat_var = cfg_.alpha
                          + cfg_.beta1  * rv1  * rv1
                          + cfg_.beta5  * rv5  * rv5
                          + cfg_.beta22 * rv22 * rv22;
        double rv_hat = std::sqrt(std::max(rv_hat_var, 1e-8));

        double spread = iv.atm_implied_vol - rv_hat;

        har_history_.push_back(spread);
        if ((int)har_history_.size() > cfg_.base.window)
            har_history_.pop_front();

        bool   window_full = ((int)har_history_.size() == cfg_.base.window);
        double zscore      = 0.0;

        if (window_full) {
            auto [mean, std_dev] = rolling_stats(har_history_);
            if (std_dev > 1e-12)
                zscore = (spread - mean) / std_dev;
        }

        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = rv_hat * rv_hat;   // HAR forecast variance
        snap.raw_spread              = spread;
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
    HARSignalConfig                                      cfg_;

    double rv_ring_[22] = {};
    int    rv_idx_      = 0;
    double last_spot_   = 0.0;

    std::deque<double> har_history_;
};

} // namespace omm::buyer
