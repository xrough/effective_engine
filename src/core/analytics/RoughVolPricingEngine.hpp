#pragma once
#include <mutex>
#include "../domain/Instrument.hpp"
#include "PricingEngine.hpp"

// ============================================================
// File：RoughVolPricingEngine.hpp
// Rough Bergomi first order skew correction for European options
// Reference：Bayer, Friz, Gatheral (2016), §4.2
//
//   - update_params() support hot injection of calibration results
//   - mutex protection
//   - Realize IPricingEngine
// ============================================================

namespace omm::domain {

// RoughVolParams — Rough Bergomi parameters
struct RoughVolParams {
    double H    = 0.10;    // Hurst 
    double eta  = 1.50;    // vol-of-vol
    double rho  = -0.70;   // spot-vol 
    double xi0  = 0.0625;  // t=0 forward variance
};

class RoughVolPricingEngine final : public IPricingEngine {
public:
    explicit RoughVolPricingEngine(
        RoughVolParams params = RoughVolParams{},
        double r = 0.05 // interest rate (annualized)
    );

    // price() — 粗糙波动率偏斜修正定价（实现 IPricingEngine 接口）
    PriceResult price(
        const Option& option,
        double underlying_price
    ) const override;

    // update_params() — 热注入校准结果（Phase 2 完成后调用）
    // 线程安全：内部使用 mutex
    void update_params(const RoughVolParams& params);

    // 查询当前参数（供日志/报告使用）
    RoughVolParams get_params() const;

private:
    // bs_price_and_delta() — 标准 BS 公式（复用 PricingEngine 逻辑）
    //   sigma — 偏斜调整后的波动率 σ_K
    PriceResult bs_price_and_delta(
        double S, double K, double T, double sigma, bool is_call
    ) const;

    // compute_skew_adjusted_vol() — 计算偏斜修正波动率 σ_K
    double compute_skew_adjusted_vol(double K, double S, double T) const;

    mutable std::mutex params_mutex_; 
    RoughVolParams     params_;       
    double             r_;            
};

} // namespace omm::domain
