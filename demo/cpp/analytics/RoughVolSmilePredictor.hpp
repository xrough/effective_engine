#pragma once
#include <cmath>
#include <algorithm>
#include "core/analytics/RoughVolPricingEngine.hpp"

// ============================================================
// File: RoughVolSmilePredictor.hpp  (demo/cpp/)
// Role: Analytical smile shape predictions from calibrated
//       Rough Bergomi / Rough Heston parameters.
//
// Reference: Bayer, Friz, Gatheral (2016) small-time expansion
//   σ_imp(k) ≈ σ₀ + (ρη/4σ₀)·T^{H−0.5}·k + (η²(2−3ρ²)/24σ₀³)·T^{2H−1}·k²
//
// Gate results (AAPL 2026-04-08, Rough Heston best fit):
//   H=0.01, ν=0.838, ρ=−0.507, v0=0.077
//   Skew scales as T^{−0.49} (extremely steep for short-dated)
//   Curvature scales as T^{−0.98} (large convexity from vol-of-vol)
// ============================================================

namespace omm::demo {

struct SmilePrediction {
    double atm_vol;        // sqrt(xi0) — model ATM vol
    double skew_slope;     // dσ/dk = ρη/(4σ₀) × T^{H−0.5}  (negative for ρ<0)
    double curvature;      // d²σ/dk² = η²(2−3ρ²)/(24σ₀³) × T^{2H−1}
    double term_exponent;  // H − 0.5 (diagnostic: sign/magnitude of power-law)
};

class RoughVolSmilePredictor {
public:
    static constexpr double T_MIN = 7.0 / 365.0;  // 7-day floor

    SmilePrediction predict(const domain::RoughVolParams& p, double T) const {
        T = std::max(T, T_MIN);
        double sigma0 = std::sqrt(std::max(p.xi0, 1e-8));
        double H_exp  = p.H - 0.5;   // < 0 in rough regime → power-law blow-up

        // Skew slope coefficient (negative for ρ < 0, i.e. equity put skew)
        double skew_slope = (p.rho * p.eta) / (4.0 * sigma0)
                            * std::pow(T, H_exp);

        // Curvature (smile convexity): always positive for η > 0
        // For calibrated ρ=−0.507: 2 − 3ρ² = 2 − 3×0.257 = 1.229
        double rho_sq    = p.rho * p.rho;
        double curvature = (p.eta * p.eta * (2.0 - 3.0 * rho_sq))
                           / (24.0 * sigma0 * sigma0 * sigma0)
                           * std::pow(T, 2.0 * p.H - 1.0);

        return SmilePrediction{ sigma0, skew_slope, curvature, H_exp };
    }
};

} // namespace omm::demo
