#include "RoughVolPricingEngine.hpp"
#include <algorithm> 
#include <cmath>     
#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================
// File: RoughVolPricingEngine.cpp
// Bergomi-Guyon 2nd-order asymptotic expansion for implied vol.
//
// Implied volatility formula (Bergomi-Guyon 2012 / Fukasawa 2011):
//
//   σ_ATM = √ξ₀                                        ATM vol
//   ψ(T)  = ρ·η·Γ(H+0.5) / (2√π) · T^{H-0.5}         skew slope
//   χ(T)  = ψ(T)² / σ_ATM                              smile curvature
//   k     = log(K/S)                                    log-moneyness
//
//   σ_K   = max(σ_ATM + ψ·k + (χ/2)·k², 0.001)        2nd-order smile
//
// The curvature term χ/2·k² (Fukasawa 2011, Thm 4.1) is the
// leading-order smile convexity for any rough vol model.
// It ensures σ(k) is symmetric around ATM and always concave-up,
// capturing the observed parabolic shape of short-dated smiles.
//
// Compared to 1st-order (linear in k):
//   OTM call (k>0, ρ<0):  ψ·k < 0 (lower vol), χ/2·k² > 0 (partial offset)
//   OTM put  (k<0, ρ<0):  ψ·k > 0 (higher vol), χ/2·k² > 0 (further boost)
//   → asymmetric smile with put skew, consistent with equity markets
// ============================================================

namespace omm::domain {

static double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

RoughVolPricingEngine::RoughVolPricingEngine(RoughVolParams params, double r)
    : params_(params), r_(r) {}

// ── compute_skew_adjusted_vol() ───────────────────────────────
// Bergomi-Guyon (2012) §4.2 / Bayer-Friz-Gatheral (2016)
double RoughVolPricingEngine::compute_skew_adjusted_vol(
    double K, double S, double T) const
{
    // parameters copy for thread safety (minimize lock scope)
    RoughVolParams p;
    {
        std::lock_guard<std::mutex> lk(params_mutex_);
        p = params_;
    }

    // σ_ATM = √ξ₀ apprximately
    double sigma_atm = std::sqrt(p.xi0);

    // ψ(T) = ρ·η·Γ(H+0.5)/(2√π) · T^{H-0.5}
    double gamma_val = std::tgamma(p.H + 0.5);
    double psi = p.rho * p.eta * gamma_val
                 / (2.0 * std::sqrt(M_PI))
                 * std::pow(T, p.H - 0.5);

    // log-moneyness k = log(K/S)
    double k = std::log(K / S);

    // smile curvature χ = ψ²/σ_ATM  (Fukasawa 2011, leading-order convexity)
    double chi = (psi * psi) / sigma_atm;

    // 2nd-order implied vol: σ_K = σ_ATM + ψ·k + (χ/2)·k²
    double sigma_k = sigma_atm + psi * k + 0.5 * chi * k * k;
    return std::max(sigma_k, 0.001);
}

// ── bs_price_and_delta() ─────────────────────────────────────
PriceResult RoughVolPricingEngine::bs_price_and_delta(
    double S, double K, double T, double sigma, bool is_call, double /*sigma_impl*/) const
{
    double d1 = (std::log(S / K) + (r_ + 0.5 * sigma * sigma) * T)
                / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    double Nd1  = norm_cdf(d1);
    double Nd2  = norm_cdf(d2);
    double n_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);  // N'(d1)

    double theo  = 0.0;
    double delta = 0.0;
    double theta = 0.0;

    if (is_call) {
        theo  = S * Nd1 - K * std::exp(-r_ * T) * Nd2;
        delta = Nd1;
        theta = -(S * sigma * n_d1) / (2.0 * std::sqrt(T))
                - r_ * K * std::exp(-r_ * T) * Nd2;
    } else {
        theo  = K * std::exp(-r_ * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
        delta = Nd1 - 1.0;
        theta = -(S * sigma * n_d1) / (2.0 * std::sqrt(T))
                + r_ * K * std::exp(-r_ * T) * norm_cdf(-d2);
    }

    PriceResult result;
    result.theo  = std::max(0.0, theo);
    result.delta = delta;
    result.gamma = n_d1 / (S * sigma * std::sqrt(T));
    result.vega  = S * std::sqrt(T) * n_d1;
    result.theta = theta;
    return result;
}

// ── price() ──────────────────────────────────────────────────
// main pricing method implementing IPricingEngine interface
PriceResult RoughVolPricingEngine::price(
    const Option& option,
    double underlying_price) const
{
    const double S = underlying_price;
    const double K = option.strike();

    // T: time to maturity in years under system clock
    auto now = std::chrono::system_clock::now();
    double T = std::chrono::duration<double>(option.expiry() - now).count()
               / (365.0 * 24.0 * 3600.0);
    T = std::max(T, 1e-6);

    // ajusted vol for strike K
    double sigma_atm = std::sqrt(params_.xi0);  // ATM vol for logging
    double sigma_k   = compute_skew_adjusted_vol(K, S, T);

    // adjustment for logging
    double skew_adj = sigma_k - sigma_atm;

    std::cout << "[Rough Volatility] " << option.id()
              << "  σ_ATM=" << std::fixed << std::setprecision(3) << sigma_atm
              << "  σ_K="   << sigma_k
              << "  adj=" << std::showpos << std::setprecision(3) << skew_adj
              << " (linear+convexity)"
              << std::noshowpos << "\n";

    bool is_call = (option.option_type() == OptionType::Call);
    return bs_price_and_delta(S, K, T, sigma_k, is_call, sigma_atm);
}

// ── update_params() ──────────────────────────────────────────
void RoughVolPricingEngine::update_params(const RoughVolParams& params) {
    std::lock_guard<std::mutex> lk(params_mutex_);
    params_ = params;
}

// ── get_params() ─────────────────────────────────────────────
RoughVolParams RoughVolPricingEngine::get_params() const {
    std::lock_guard<std::mutex> lk(params_mutex_);
    return params_;
}

} // namespace omm::domain