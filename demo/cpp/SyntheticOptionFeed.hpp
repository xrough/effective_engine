#pragma once
#include <cmath>
#include <memory>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"

// ============================================================
// File: SyntheticOptionFeed.hpp  (demo/cpp/)
// Role: simulation adapter — converts MarketDataEvent into
//       OptionMidQuoteEvent for an ATM straddle.
//
// This is demo infrastructure only. BuyerModule, DeltaHedger,
// and AlphaPnLTracker have no knowledge of this class — they
// only consume the OptionMidQuoteEvents it emits.
//
// Replacing this with a real market feed requires no changes
// to any engine component.
//
// Method: BS formula prices ATM call + put at current spot.
//   σ = 0.25, r = 0.05, T = 30/365 (fixed for demo)
//   strike = round(spot) to nearest integer
// ============================================================

namespace omm::demo {

class SyntheticOptionFeed {
public:
    explicit SyntheticOptionFeed(
        std::shared_ptr<events::EventBus> bus,
        double sigma = 0.25,
        double r     = 0.05,
        double T     = 30.0 / 365.0
    ) : bus_(std::move(bus)), sigma_(sigma), r_(r), T_(T) {}

    void register_handlers() {
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { on_market_data(e); }
        );
    }

private:
    void on_market_data(const events::MarketDataEvent& e) {
        double S = e.underlying_price;
        double K = std::round(S);  // ATM strike = nearest integer

        // 近似：用短期价格变动模拟 IV 的轻微波动（负相关效应）
        // σ_eff = σ_base * (1 - 0.4 * Δ_return)，Δ_return 有正有负
        double sigma_eff = sigma_;
        if (last_price_ > 0.0) {
            double ret = (S - last_price_) / last_price_;
            sigma_eff = std::max(0.05, sigma_ * (1.0 - 0.4 * ret));
        }
        last_price_ = S;

        double call_mid = bs_price(S, K, sigma_eff, true);
        double put_mid  = bs_price(S, K, sigma_eff, false);

        if (call_mid <= 0.0 || put_mid <= 0.0) return;

        events::OptionMidQuoteEvent call_evt;
        call_evt.instrument_id  = "AAPL_ATM_CALL";
        call_evt.mid_price      = call_mid;
        call_evt.underlying     = S;
        call_evt.strike         = K;
        call_evt.time_to_expiry = T_;
        call_evt.is_call        = true;
        call_evt.timestamp      = e.timestamp;
        bus_->publish(call_evt);

        events::OptionMidQuoteEvent put_evt;
        put_evt.instrument_id  = "AAPL_ATM_PUT";
        put_evt.mid_price      = put_mid;
        put_evt.underlying     = S;
        put_evt.strike         = K;
        put_evt.time_to_expiry = T_;
        put_evt.is_call        = false;
        put_evt.timestamp      = e.timestamp;
        bus_->publish(put_evt);
    }

    double bs_price(double S, double K, double sigma, bool is_call) const {
        auto norm_cdf = [](double x) {
            return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        };
        double d1 = (std::log(S / K) + (r_ + 0.5 * sigma * sigma) * T_)
                    / (sigma * std::sqrt(T_));
        double d2 = d1 - sigma * std::sqrt(T_);
        if (is_call)
            return S * norm_cdf(d1) - K * std::exp(-r_ * T_) * norm_cdf(d2);
        else
            return K * std::exp(-r_ * T_) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }

    std::shared_ptr<events::EventBus> bus_;
    double sigma_;
    double r_;
    double T_;
    double last_price_ = 0.0;
};

} // namespace omm::demo
