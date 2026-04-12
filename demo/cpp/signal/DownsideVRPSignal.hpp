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
// File: DownsideVRPSignal.hpp  (demo/cpp/signal/)
// Role: Downside Variance Risk Premium signal — IV² vs downside
//       realized semivariance (RS⁻), z-scored.
//
// Signal:
//   RS⁻       = Σ r²·1(r<0) over rv_window bars, annualized
//   dvp_spread = IV² − RS⁻_ann
//   zscore     = (dvp_spread − rolling_mean) / rolling_std
//
// Interpretation: dvp_spread > 0 → IV is rich even relative to the
// downside-only RV (the pure fear premium). A high positive z-score
// signals expensive left-tail protection.
//
// Literature:
//   Feunou, Jahan-Parvar & Okou (2018, JFE) — VRP decomposition
//   Kilic & Shaliastovich (2019, JFE) — downside vs total VRP
//   Bollerslev, Patton & Quaedvlieg (2022) — semivariance forecasting
//
// Why distinct from VRPAlphaSignal (IV² − RV5²):
//   VRPAlphaSignal uses total RV = RS⁺ + RS⁻.
//   This signal uses only RS⁻. The spread IV² − RS⁻ is larger than
//   the standard VRP and responds more strongly to fear regimes,
//   since left-tail returns dominate RS⁻ during stress periods.
//
// Window: 22 bars (≈ 3.4 minutes of intraday data at 1-min bars,
//   or roughly one trading session at 5-min bars) for semivariance.
// ============================================================

namespace omm::buyer {

struct DownsideVRPSignalConfig {
    AlphaSignalConfig base;    // window=50, z_entry=1.5, z_exit=0.5
    int rv_window = 22;        // bars for downside semivariance estimation
};

class DownsideVRPSignal : public IAlphaSignal {
public:
    DownsideVRPSignal(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       /*rough_engine*/,
        DownsideVRPSignalConfig cfg = {}
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

        // Downside semivariance: sum only squared negative returns
        double rs_minus = 0.0;
        for (int i = 0; i < cfg_.rv_window; ++i) {
            double r = rv_ring_[(rv_idx_ - 1 - i) % 22];
            if (r < 0.0) rs_minus += r * r;
        }

        // Annualize using full window denominator (preserves scale vs RV)
        double rs_minus_ann = rs_minus / cfg_.rv_window * ANNUAL;

        double iv_sq      = iv.atm_implied_vol * iv.atm_implied_vol;
        double dvp_spread = iv_sq - rs_minus_ann;

        history_.push_back(dvp_spread);
        if ((int)history_.size() > cfg_.base.window)
            history_.pop_front();

        bool   window_full = ((int)history_.size() == cfg_.base.window);
        double zscore      = 0.0;

        if (window_full) {
            auto [mean, std_dev] = rolling_stats(history_);
            if (std_dev > 1e-12)
                zscore = (dvp_spread - mean) / std_dev;
        }

        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = rs_minus_ann;   // diagnostics: annualized RS⁻
        snap.raw_spread              = dvp_spread;
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
    DownsideVRPSignalConfig                              cfg_;

    double rv_ring_[22] = {};
    int    rv_idx_      = 0;
    double last_spot_   = 0.0;

    std::deque<double> history_;
};

} // namespace omm::buyer
