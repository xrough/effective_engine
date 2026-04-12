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
#include "core/domain/InstrumentFactory.hpp"

// ============================================================
// File: IVAutocorrSignal.hpp  (demo/cpp/signal/)
// Role: Rolling lag-1 autocorrelation of IV changes as a
//       mean-reversion timing signal, z-scored.
//
// Signal:
//   Δiv_t     = IV_t − IV_{t-1}    (first difference of ATM IV)
//   ρ_1       = Pearson corr(Δiv_t, Δiv_{t-1}) over autocorr_window bars
//   zscore    = (ρ_1 − rolling_mean) / rolling_std
//
// Interpretation:
//   ρ_1 > 0  → IV changes are positively autocorrelated (trending).
//              Mean reversion is less likely right now.
//   ρ_1 < 0  → IV changes are negatively autocorrelated (reverting).
//              The trend has broken; a long straddle trade may capture
//              the mean reversion.
//   A highly negative z-score of ρ_1 signals an unusually strong
//   mean-reversion regime relative to the recent rolling history.
//
// Note: this is different from VolOfVolSignal (rolling std of IV
//   changes). VoV captures magnitude; this signal captures the sign
//   of serial dependence. They are complementary.
//
// Literature:
//   Cao, Han & Tong (2019, RFS) — volatility of volatility and tail premia
//   Da, Shen & Xiu (2022) — IV predictability
//   Brunner, Ehm & Schmitt (2022) — implied volatility dynamics
//
// Guard: if variance of IV changes is near zero (flat IV), the
//   autocorrelation is ill-defined and the signal skips the bar.
// ============================================================

namespace omm::buyer {

struct IVAutocorrSignalConfig {
    AlphaSignalConfig base;          // window=50, z_entry=1.5, z_exit=0.5
    int autocorr_window = 22;        // bars for lag-1 autocorrelation estimation
};

class IVAutocorrSignal : public IAlphaSignal {
public:
    IVAutocorrSignal(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       rough_engine,
        IVAutocorrSignalConfig cfg = {}
    ) : bus_(std::move(bus))
      , extractor_(std::move(extractor))
      , rough_(std::move(rough_engine))
      , cfg_(cfg) {}

    void register_handlers() override {
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { on_market_data(e); }
        );
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_option_quote(e); }
        );
    }

    // Track rv_idx_ so warmup is synchronized with ring-buffer signals
    // (no log-return stored — the counter is used only for warmup gating).
    void on_market_data(const events::MarketDataEvent& e) override {
        if (last_spot_sync_ > 0.0 && e.underlying_price > 0.0)
            ++rv_idx_;
        last_spot_sync_ = e.underlying_price;
    }

    void on_option_quote(const events::OptionMidQuoteEvent& e) {
        analytics::ImpliedVariancePoint iv = extractor_->last_point();
        if (!iv.valid || iv.atm_implied_vol <= 0.0) return;

        double iv_now = iv.atm_implied_vol;

        // Accumulate IV differences: Δiv = iv_t − iv_{t-1}
        if (last_iv_ > 0.0) {
            iv_changes_.push_back(iv_now - last_iv_);
            // Keep one extra element so lag-1 pairing spans the full window
            if ((int)iv_changes_.size() > cfg_.autocorr_window + 1)
                iv_changes_.pop_front();
        }
        last_iv_ = iv_now;

        // Use rv_idx_ for warmup gating (synchronized with ring-buffer signals).
        // Must publish valid=false (not plain-return) to keep CompositeAlphaSignal
        // round-robin counter synchronized with the other sub-signals.
        if (rv_idx_ < cfg_.autocorr_window) {
            events::SignalSnapshotEvent snap;
            snap.ts = e.timestamp;  snap.valid = false;  snap.zscore = 0.0;
            snap.atm_implied_variance = iv.atm_implied_variance;
            snap.rough_forecast_variance = 0.0;  snap.raw_spread = 0.0;
            snap.calibration_ok = true;  snap.atm_vega = 0.0;  snap.atm_spot = e.underlying;
            bus_->publish(snap);
            return;
        }
        // Also need enough IV changes accumulated
        if ((int)iv_changes_.size() < cfg_.autocorr_window) {
            // Publish valid=false to keep round-robin intact
            events::SignalSnapshotEvent snap;
            snap.ts = e.timestamp;  snap.valid = false;  snap.zscore = 0.0;
            snap.atm_implied_variance = iv.atm_implied_variance;
            snap.rough_forecast_variance = 0.0;  snap.raw_spread = 0.0;
            snap.calibration_ok = true;  snap.atm_vega = 0.0;  snap.atm_spot = e.underlying;
            bus_->publish(snap);
            return;
        }

        int n = cfg_.autocorr_window;

        // Mean of IV changes in the window
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
            sum += iv_changes_[iv_changes_.size() - n + i];
        double mean_chg = sum / n;

        // Lag-1 covariance and variance
        double cov_lag = 0.0, var_chg = 0.0;
        for (int i = 1; i < n; ++i) {
            double di   = iv_changes_[iv_changes_.size() - n + i    ] - mean_chg;
            double di_1 = iv_changes_[iv_changes_.size() - n + i - 1] - mean_chg;
            cov_lag += di * di_1;
            var_chg += di * di;
        }
        // Also include the first element in variance (not in covariance)
        {
            double d0 = iv_changes_[iv_changes_.size() - n] - mean_chg;
            var_chg  += d0 * d0;
        }

        // Guard: flat IV — autocorr ill-defined; publish valid=false to keep round-robin intact
        bool   autocorr_valid = (var_chg >= 1e-18);
        double autocorr = autocorr_valid ? (cov_lag / var_chg) : 0.0;

        autocorr_history_.push_back(autocorr);
        if ((int)autocorr_history_.size() > cfg_.base.window)
            autocorr_history_.pop_front();

        bool   window_full = ((int)autocorr_history_.size() == cfg_.base.window);
        double zscore      = 0.0;

        if (window_full) {
            auto [mean, std_dev] = rolling_stats(autocorr_history_);
            if (std_dev > 1e-12)
                zscore = (autocorr - mean) / std_dev;
        }

        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full && autocorr_valid;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = var_chg / (n * n);  // diagnostics: variance per step²
        snap.raw_spread              = autocorr;
        snap.zscore                  = zscore;
        snap.calibration_ok          = true;
        snap.atm_spot                = e.underlying;

        // ATM vega for StrategyController position sizing — same pattern as SkewCurvatureAlphaSignal
        {
            double T   = iv.time_to_expiry;
            double spot = e.underlying;
            auto expiry_tp = std::chrono::system_clock::now()
                + std::chrono::hours(static_cast<int>(T * 365.0 * 24.0));
            auto call_tmp = domain::InstrumentFactory::make_call("IVAC_ATM", std::round(spot), expiry_tp);
            snap.atm_vega = rough_ ? rough_->price(*call_tmp, spot).vega : 0.0;
        }

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
    std::shared_ptr<domain::RoughVolPricingEngine>       rough_;
    IVAutocorrSignalConfig                               cfg_;

    double             last_iv_       = -1.0;  // sentinel: -1 = uninitialized
    double             last_spot_sync_= 0.0;  // for rv_idx_ warmup counter
    int                rv_idx_        = 0;    // synchronized with ring-buffer signals
    std::deque<double> iv_changes_;           // consecutive IV first differences
    std::deque<double> autocorr_history_;     // rolling z-score window for ρ_1
};

} // namespace omm::buyer
