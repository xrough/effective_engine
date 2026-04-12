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
// File: LogIVRVRatioSignal.hpp  (demo/cpp/signal/)
// Role: Multiplicative Volatility Risk Premium signal —
//       log(IV / RV5), z-scored.
//
// Signal:
//   rv5        = sqrt( sum(r²_last5) / 5 * 252 * 390 )
//   log_ratio  = log( IV / max(rv5, 1e-4) )
//   zscore     = (log_ratio − rolling_mean) / rolling_std
//
// Why the log ratio instead of IV² − RV²:
//   The additive form IV²−RV² has vol-level dependence: in a 10% vol
//   environment, a 2pt gap looks the same as in a 30% environment.
//   The log ratio form log(IV/RV) scales naturally with the vol level
//   (it is the log-normal model's VRP), is closer to the true
//   risk-neutral to physical measure change, and has been shown to
//   have better out-of-sample forecasting properties.
//
// Literature:
//   Todorov (2019, JF) — nonparametric spot vol from options
//   Martin & Wagner (2019, RFS) — expected return on a stock via SVIX
//   Andersen, Fusari & Todorov (2020, JF) — short-maturity options
//
// Positive z-score → IV unusually rich vs recent RV (in log scale).
// Uses a 5-bar RV window, matching the existing VRPAlphaSignal.
// ============================================================

namespace omm::buyer {

struct LogIVRVRatioSignalConfig {
    AlphaSignalConfig base;    // window=50, z_entry=1.5, z_exit=0.5
    int rv_window = 5;         // bars for RV estimation (5-bar, same as VRPAlphaSignal)
};

class LogIVRVRatioSignal : public IAlphaSignal {
public:
    LogIVRVRatioSignal(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       /*rough_engine*/,
        LogIVRVRatioSignalConfig cfg = {}
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
        if (rv_idx_ < cfg_.rv_window) return;

        constexpr double ANNUAL = 252.0 * 390.0;

        // 5-bar realized vol (same as VRPAlphaSignal pattern)
        double ss = 0.0;
        for (int i = 0; i < cfg_.rv_window; ++i) {
            double r = rv_ring_[(rv_idx_ - 1 - i) % 22];
            ss += r * r;
        }
        double rv5 = std::sqrt(ss / cfg_.rv_window * ANNUAL);

        // Floor RV to avoid log(0) / log(very small number)
        constexpr double RV_FLOOR = 1e-4;   // 1bp annualised vol
        double rv_safe = std::max(rv5, RV_FLOOR);

        // Log ratio: the multiplicative VRP measure
        double log_ratio = std::log(iv.atm_implied_vol / rv_safe);

        history_.push_back(log_ratio);
        if ((int)history_.size() > cfg_.base.window)
            history_.pop_front();

        bool   window_full = ((int)history_.size() == cfg_.base.window);
        double zscore      = 0.0;

        if (window_full) {
            auto [mean, std_dev] = rolling_stats(history_);
            if (std_dev > 1e-12)
                zscore = (log_ratio - mean) / std_dev;
        }

        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full && rv_idx_ >= cfg_.rv_window;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = rv5 * rv5;   // diagnostics: RV5 variance
        snap.raw_spread              = log_ratio;
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
    LogIVRVRatioSignalConfig                             cfg_;

    double rv_ring_[22] = {};
    int    rv_idx_      = 0;
    double last_spot_   = 0.0;

    std::deque<double> history_;
};

} // namespace omm::buyer
