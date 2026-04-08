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

    // price() — realizing IPricingEngine interface
    PriceResult price(
        const Option& option,
        double underlying_price
    ) const override;

    // update_params() — hot injection of calibration
    void update_params(const RoughVolParams& params);
    // get_params() — query current parameters for logging/monitoring
    RoughVolParams get_params() const;

private:
    // bs_price_and_delta() — standard BS formula with full Greeks
    PriceResult bs_price_and_delta(
        double S, double K, double T, double sigma, bool is_call,
        double sigma_impl = -1.0  // 调用方传入的ATM vol供vega日志使用；默认使用sigma
    ) const;

    // compute_skew_adjusted_vol() — adjust vol
    double compute_skew_adjusted_vol(double K, double S, double T) const;

    mutable std::mutex params_mutex_;  // mutual exclusion for params access
    RoughVolParams     params_;       
    double             r_;            
};

} // namespace omm::domain
