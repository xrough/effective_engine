#pragma once
#include <memory>
#include <iostream>
#include <iomanip>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "core/analytics/ImpliedVarianceExtractor.hpp"
#include "core/analytics/RoughVolPricingEngine.hpp"

// ============================================================
// File: RoughVolCalibrator.hpp  (demo/cpp/)
// Role: EWMA online calibration of xi0 from observed ATM implied variance.
//
// Signal identity:
//   raw_spread = σ²_atm(market) − xi0 * T
//   With a fixed xi0, the spread is just "IV minus a constant".
//   With EWMA-calibrated xi0, the spread genuinely measures
//   deviation from the model's rolling forward variance estimate.
//
// Design:
//   - Subscribes to OptionMidQuoteEvent (same bus as VarianceAlphaSignal)
//   - Computes xi0_spot = σ²_atm / T  (instantaneous forward variance)
//   - EWMA-smooths to xi0_ewma_
//   - Updates RoughVolPricingEngine with a 1-tick lag so the signal
//     always compares today's IV against the *prior* estimate
//
// Guards:
//   - T_MIN = 7/365: near-expiry ticks are skipped to prevent xi0_spot explosion
//   - xi0 floored at 1e-6 before writing to prevent downstream sqrt(0)
// ============================================================

namespace omm::demo {

class RoughVolCalibrator {
public:
    RoughVolCalibrator(
        std::shared_ptr<events::EventBus>                    bus,
        std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor,
        std::shared_ptr<domain::RoughVolPricingEngine>       rough_engine,
        double lambda  = 0.15,  // EWMA 衰减率 (~6-tick 半衰期)
        bool   verbose = false  // set true to log per-tick calibration
    )
        : bus_(std::move(bus))
        , extractor_(std::move(extractor))
        , rough_engine_(std::move(rough_engine))
        , lambda_(lambda)
        , verbose_(verbose)
    {
        // 用当前 xi0 初始化 EWMA，避免冷启动跳跃
        xi0_ewma_ = rough_engine_->get_params().xi0;
        xi0_prev_ = xi0_ewma_;
    }

    void register_handlers() {
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_option_quote(e); }
        );
    }

private:
    static constexpr double T_MIN      = 7.0 / 365.0;  // 1周到期下限
    static constexpr double XI0_FLOOR  = 1e-6;

    void on_option_quote(const events::OptionMidQuoteEvent& /*e*/) {
        auto iv = extractor_->last_point();
        if (!iv.valid) return;

        // 近到期保护：跳过 T < 1周 的 tick
        if (iv.time_to_expiry < T_MIN) return;

        // 点估计：σ²_atm / T  →  即时前向方差
        double xi0_spot = iv.atm_implied_variance / iv.time_to_expiry;

        // EWMA 平滑
        xi0_ewma_ = (1.0 - lambda_) * xi0_ewma_ + lambda_ * xi0_spot;

        // 1-tick 滞后写入引擎：信号读取的始终是上一期估计
        domain::RoughVolParams p = rough_engine_->get_params();
        p.xi0 = std::max(xi0_prev_, XI0_FLOOR);
        rough_engine_->update_params(p);

        if (verbose_)
            std::cout << "[RoughVolCalibrator] xi0_ewma=" << std::fixed
                      << std::setprecision(5) << xi0_ewma_
                      << "  xi0_spot=" << xi0_spot
                      << "  applied=" << p.xi0 << "\n";

        xi0_prev_ = xi0_ewma_;
    }

    std::shared_ptr<events::EventBus>                    bus_;
    std::shared_ptr<analytics::ImpliedVarianceExtractor> extractor_;
    std::shared_ptr<domain::RoughVolPricingEngine>       rough_engine_;

    double lambda_;
    bool   verbose_;
    double xi0_ewma_;
    double xi0_prev_;
};

} // namespace omm::demo
