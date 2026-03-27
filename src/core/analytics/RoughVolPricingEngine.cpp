#include "RoughVolPricingEngine.hpp"
#include <algorithm> 
#include <cmath>     
#include <chrono>
#include <iostream>
#include <iomanip>

// ============================================================
// File：RoughVolPricingEngine.cpp
// Bergomi-Guyon first order skew correction for European options
//
// Explicit formula for skew-adjusted volatility:
//   σ_ATM = √ξ₀
//   ψ(T)  = ρ·η·Γ(H+0.5) / (2√π) · T^{H-0.5}   
//   k     = log(K/S)                               
//   σ_K   = max(σ_ATM + ψ·k, 0.001)               
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

    // log moneyness k = log(K/S)
    double k = std::log(K / S);

    // σ_K = σ_ATM + ψ·k skew adjustment with floor
    double sigma_k = sigma_atm + psi * k;
    return std::max(sigma_k, 0.001);
}

// ── bs_price_and_delta() ─────────────────────────────────────
PriceResult RoughVolPricingEngine::bs_price_and_delta(
    double S, double K, double T, double sigma, bool is_call) const
{
    double d1 = (std::log(S / K) + (r_ + 0.5 * sigma * sigma) * T)
                / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    double theo  = 0.0;
    double delta = 0.0;

    if (is_call) {
        theo  = S * norm_cdf(d1) - K * std::exp(-r_ * T) * norm_cdf(d2);
        delta = norm_cdf(d1);
    } else {
        theo  = K * std::exp(-r_ * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
        delta = norm_cdf(d1) - 1.0;
    }

    theo = std::max(0.0, theo);
    return PriceResult{theo, delta};
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
    double sigma_atm = std::sqrt(params_.xi0);  // 仅用于日志，无锁读取近似值
    double sigma_k   = compute_skew_adjusted_vol(K, S, T);

    // adjustment for logging
    double skew_adj = sigma_k - sigma_atm;

    std::cout << "[Rough Volatility] " << option.id()
              << "  σ_ATM=" << std::fixed << std::setprecision(3) << sigma_atm
              << "  σ_K="   << sigma_k
              << "  skew adjustment=" << std::showpos << std::setprecision(3) << skew_adj
              << std::noshowpos << "\n";

    bool is_call = (option.option_type() == OptionType::Call);
    return bs_price_and_delta(S, K, T, sigma_k, is_call);
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