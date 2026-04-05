#pragma once
#include <cmath>
#include <mutex>
#include <memory>
#include <stdexcept>
#include "../../core/events/EventBus.hpp"
#include "../../core/events/Events.hpp"

// ============================================================
// 文件：ImpliedVarianceExtractor.hpp
// 职责：从 OptionMidQuoteEvent 中反推 ATM 隐含波动率和隐含方差。
//
// 方法：Black-Scholes 二分法（50次迭代，σ ∈ [1e-4, 5.0]）
// 输出：ImpliedVariancePoint（供 VarianceAlphaSignal 消费）
// ============================================================

namespace omm::buyer {

// ImpliedVariancePoint — 单次隐含方差提取结果
struct ImpliedVariancePoint {
    bool   valid               = false;
    double strike              = 0.0;
    double atm_implied_vol     = 0.0;
    double atm_implied_variance = 0.0;   // σ²_atm
    double total_implied_variance = 0.0; // σ²_atm * T
    double time_to_expiry      = 0.0;
};

class ImpliedVarianceExtractor {
public:
    explicit ImpliedVarianceExtractor(
        std::shared_ptr<events::EventBus> bus,
        double r = 0.05
    ) : bus_(std::move(bus)), r_(r) {}

    void register_handlers() {
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_quote(e); }
        );
    }

    // 线程安全地读取最新提取结果
    ImpliedVariancePoint last_point() const {
        std::lock_guard<std::mutex> lk(mu_);
        return last_;
    }

private:
    void on_quote(const events::OptionMidQuoteEvent& e) {
        if (e.time_to_expiry <= 0.0 || e.mid_price <= 0.0) return;

        double vol = bs_implied_vol(e.underlying, e.strike,
                                    e.time_to_expiry, r_,
                                    e.mid_price, e.is_call);
        ImpliedVariancePoint pt;
        pt.valid                 = (vol > 0.0);
        pt.strike                = e.strike;
        pt.atm_implied_vol       = vol;
        pt.atm_implied_variance  = vol * vol;
        pt.total_implied_variance = vol * vol * e.time_to_expiry;
        pt.time_to_expiry        = e.time_to_expiry;

        std::lock_guard<std::mutex> lk(mu_);
        last_ = pt;
    }

    // BS 隐含波动率：二分法，50次迭代
    double bs_implied_vol(double S, double K, double T, double r,
                           double mid, bool is_call) const {
        auto norm_cdf = [](double x) {
            return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        };
        auto bs_price = [&](double sigma) -> double {
            double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
                        / (sigma * std::sqrt(T));
            double d2 = d1 - sigma * std::sqrt(T);
            if (is_call)
                return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
            else
                return K * std::exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
        };

        double lo = 1e-4, hi = 5.0;
        if (bs_price(lo) > mid || bs_price(hi) < mid) return 0.0;  // 边界检查

        for (int iter = 0; iter < 50; ++iter) {
            double mid_vol = 0.5 * (lo + hi);
            if (bs_price(mid_vol) < mid)
                lo = mid_vol;
            else
                hi = mid_vol;
        }
        return 0.5 * (lo + hi);
    }

    std::shared_ptr<events::EventBus> bus_;
    double                            r_;
    ImpliedVariancePoint              last_;
    mutable std::mutex                mu_;
};

} // namespace omm::buyer
