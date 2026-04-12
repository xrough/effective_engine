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
// File: RealizedSkewnessSignal.hpp  (demo/cpp/signal/)
// Role: Third standardized moment of recent returns as a fear
//       premium / tail-risk indicator, z-scored.
//
// Signal:
//   RSK = [mean(r³)] / [mean(r²)]^1.5     (third standardized moment)
//   zscore = (RSK − rolling_mean) / rolling_std
//
// Interpretation:
//   RSK < 0 → left-tail return events dominated the recent window
//           → the market should price in more fear premium
//           → IV is likely to be rich
//   A very negative RSK z-score is a confirmatory signal that the
//   tail-driven component of VRP is elevated.
//
// Literature:
//   Conrad, Dittmar & Ghysels (2013) — higher moment return prediction
//   Stilger, Kostakis & Poon (2017) — realized higher moments
//   Feunou, Fontaine & Taamouti (2022) — risk-neutral vs physical moments
//   Multiple 2020+ papers extending realized higher moments to option pricing
//
// Guard: if realized variance is near zero (flat market), RSK is
//   ill-defined and the signal returns early without publishing.
//
// Window: 22 bars for RSK estimation (monthly lag, same as HAR-RV).
// ============================================================

namespace omm::buyer {

struct RealizedSkewnessSignalConfig {
    AlphaSignalConfig base;    // window=50, z_entry=1.5, z_exit=0.5
    int rsk_window = 22;       // bars for realized skewness estimation
};

class RealizedSkewnessSignal : public IAlphaSignal {
public:
    RealizedSkewnessSignal(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       /*rough_engine*/,
        RealizedSkewnessSignalConfig cfg = {}
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
        if (rv_idx_ < cfg_.rsk_window) return;

        constexpr double ANNUAL = 252.0 * 390.0;

        // Third standardized moment of returns
        double sum2 = 0.0, sum3 = 0.0;
        for (int i = 0; i < cfg_.rsk_window; ++i) {
            double r = rv_ring_[(rv_idx_ - 1 - i) % 22];
            sum2 += r * r;
            sum3 += r * r * r;
        }

        double rv_bar = sum2 / cfg_.rsk_window;     // per-bar mean squared return

        double mean_r3 = sum3 / cfg_.rsk_window;
        // Guard: flat market — RSK ill-defined; publish valid=false to keep round-robin intact
        double rsk = 0.0;
        if (rv_bar >= 1e-12)
            rsk = mean_r3 / std::pow(rv_bar, 1.5);
        bool rsk_valid = (rv_bar >= 1e-12);

        history_.push_back(rsk);
        if ((int)history_.size() > cfg_.base.window)
            history_.pop_front();

        bool   window_full = ((int)history_.size() == cfg_.base.window);
        double zscore      = 0.0;

        if (window_full) {
            auto [mean, std_dev] = rolling_stats(history_);
            if (std_dev > 1e-12)
                zscore = (rsk - mean) / std_dev;
        }

        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full && rsk_valid;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = rv_bar * ANNUAL;   // diagnostics: annualized RV
        snap.raw_spread              = rsk;
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
    RealizedSkewnessSignalConfig                         cfg_;

    double rv_ring_[22] = {};
    int    rv_idx_      = 0;
    double last_spot_   = 0.0;

    std::deque<double> history_;
};

} // namespace omm::buyer
