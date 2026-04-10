#pragma once
#include <deque>
#include <cmath>
#include <memory>
#include <numeric>
#include <chrono>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "modules/buyer/IAlphaSignal.hpp"
#include "core/analytics/RoughVolPricingEngine.hpp"
#include "core/analytics/ImpliedVarianceExtractor.hpp"
#include "core/domain/InstrumentFactory.hpp"
#include "VarianceAlphaSignal.hpp"   // for AlphaSignalConfig
#include "analytics/RoughVolSmilePredictor.hpp"

// ============================================================
// File: SkewCurvatureAlphaSignal.hpp  (demo/cpp/)
// Role: Extended alpha signal incorporating rough-vol predicted
//       smile skew and curvature alongside the base variance spread.
//
// Signal components:
//   1. Variance z-score (weight 0.70): z((IV² − xi0·T))
//      Same as VarianceAlphaSignal — captures vol risk premium.
//   2. Curvature z-score (weight 0.30): z(realized_vol_of_vol − model_vol_of_vol)
//      Captures misalignment between market option convexity and the
//      rough model's predicted vol-of-vol (η × σ₀).
//
// Combined: zscore = 0.70 × var_z + 0.30 × curv_z
// Published as SignalSnapshotEvent.zscore so StrategyController
// picks it up without modification.
//
// Diagnostic accessors expose the decomposed z-scores and smile
// predictions for the ExtendedPnLTracker report.
// ============================================================

namespace omm::buyer {

struct SkewCurvatureSignalConfig {
    AlphaSignalConfig base;              // window, z_entry/exit, z_cap, base_vega
    double var_weight      = 0.70;       // weight on variance component
    double curvature_weight = 0.30;      // weight on curvature component
};

class SkewCurvatureAlphaSignal : public IAlphaSignal {
public:
    SkewCurvatureAlphaSignal(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       rough_engine,
        SkewCurvatureSignalConfig cfg = {}
    )
        : bus_(std::move(bus))
        , extractor_(std::move(extractor))
        , rough_(std::move(rough_engine))
        , cfg_(cfg)
    {}

    void register_handlers() override {
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_option_quote(e); }
        );
    }

    void on_market_data(const events::MarketDataEvent&) override {}

    // ── Diagnostic accessors (for ExtendedPnLTracker summary) ──
    double last_predicted_skew_slope() const { return last_skew_slope_; }
    double last_predicted_curvature()  const { return last_curvature_; }
    double last_var_zscore()           const { return last_var_z_; }
    double last_curvature_zscore()     const { return last_curv_z_; }

private:
    void on_option_quote(const events::OptionMidQuoteEvent& e) {
        analytics::ImpliedVariancePoint iv = extractor_->last_point();
        if (!iv.valid) return;

        auto   params = rough_->get_params();
        double T      = iv.time_to_expiry;

        // ── 1. Variance spread ────────────────────────────────
        double rough_fv   = params.xi0 * T;
        double var_spread = iv.atm_implied_variance - rough_fv;
        var_history_.push_back(var_spread);
        if ((int)var_history_.size() > cfg_.base.window)
            var_history_.pop_front();

        // ── 2. Curvature: track sigma changes vs model vol-of-vol
        double sigma_now = iv.atm_implied_vol;
        if (sigma_prev_ > 0.0) {
            sigma_change_history_.push_back(sigma_now - sigma_prev_);
            if ((int)sigma_change_history_.size() > cfg_.base.window)
                sigma_change_history_.pop_front();
        }
        sigma_prev_ = sigma_now;

        // ── 3. Smile predictions ──────────────────────────────
        demo::SmilePrediction smile = predictor_.predict(params, T);
        last_skew_slope_ = smile.skew_slope;
        last_curvature_  = smile.curvature;

        bool window_full = ((int)var_history_.size() == cfg_.base.window);
        double var_z  = 0.0;
        double curv_z = 0.0;

        if (window_full) {
            // Variance z-score
            auto [vm, vstd] = rolling_stats(var_history_);
            if (vstd > 1e-12)
                var_z = (var_spread - vm) / vstd;

            // Curvature z-score: realized vol-of-vol vs model prediction
            if ((int)sigma_change_history_.size() >= 5) {
                double realized_vov = rolling_std(sigma_change_history_);
                // Model vol-of-vol per tick: η × σ₀ × sqrt(dt) where dt≈1min
                double dt_annual   = 1.0 / (252.0 * 390.0);
                double model_vov   = params.eta * smile.atm_vol * std::sqrt(dt_annual);
                double curv_spread = realized_vov - model_vov;

                curv_spread_history_.push_back(curv_spread);
                if ((int)curv_spread_history_.size() > cfg_.base.window)
                    curv_spread_history_.pop_front();

                if ((int)curv_spread_history_.size() == cfg_.base.window) {
                    auto [cm, cstd] = rolling_stats(curv_spread_history_);
                    if (cstd > 1e-12)
                        curv_z = (curv_spread - cm) / cstd;
                }
            }
        }

        last_var_z_  = var_z;
        last_curv_z_ = curv_z;
        double combined_z = cfg_.var_weight * var_z + cfg_.curvature_weight * curv_z;

        // ── 4. ATM vega for position sizing ──────────────────
        double atm_vega = 0.0;
        {
            double spot  = e.underlying;
            double K_atm = std::round(spot);
            auto expiry_tp = std::chrono::system_clock::now()
                + std::chrono::hours(static_cast<int>(T * 365.0 * 24.0));
            auto call_tmp = domain::InstrumentFactory::make_call("SC_ATM", K_atm, expiry_tp);
            atm_vega = rough_->price(*call_tmp, spot).vega;
        }

        // ── 5. Publish as standard SignalSnapshotEvent ────────
        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = rough_fv;
        snap.raw_spread              = var_spread;
        snap.zscore                  = combined_z;   // combined — consumed by StrategyController
        snap.calibration_ok          = true;
        snap.atm_vega                = atm_vega;
        snap.atm_spot                = e.underlying;
        bus_->publish(snap);
    }

    // ── Rolling statistics helpers ────────────────────────────
    std::pair<double, double> rolling_stats(const std::deque<double>& d) const {
        double mean = std::accumulate(d.begin(), d.end(), 0.0) / (double)d.size();
        double var  = 0.0;
        for (double v : d) var += (v - mean) * (v - mean);
        return { mean, std::sqrt(var / (double)d.size()) };
    }

    double rolling_std(const std::deque<double>& d) const {
        return rolling_stats(d).second;
    }

    // ── Members ───────────────────────────────────────────────
    std::shared_ptr<events::EventBus>                    bus_;
    std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor_;
    std::shared_ptr<domain::RoughVolPricingEngine>       rough_;
    SkewCurvatureSignalConfig                            cfg_;
    demo::RoughVolSmilePredictor                         predictor_;

    std::deque<double>  var_history_;
    std::deque<double>  sigma_change_history_;
    std::deque<double>  curv_spread_history_;

    double sigma_prev_     = 0.0;
    double last_skew_slope_ = 0.0;
    double last_curvature_  = 0.0;
    double last_var_z_      = 0.0;
    double last_curv_z_     = 0.0;
};

} // namespace omm::buyer
