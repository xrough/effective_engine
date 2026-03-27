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
// 输入：行权价 K，标的价 S，到期年限 T
// 输出：偏斜修正后的 BS 波动率 σ_K
// 公式来源：Bergomi-Guyon (2012) §4.2 / Bayer-Friz-Gatheral (2016)
double RoughVolPricingEngine::compute_skew_adjusted_vol(
    double K, double S, double T) const
{
    // 参数快照（持锁）
    RoughVolParams p;
    {
        std::lock_guard<std::mutex> lk(params_mutex_);
        p = params_;
    }

    // σ_ATM = √ξ₀（平值隐含波动率近似）
    double sigma_atm = std::sqrt(p.xi0);

    // 偏斜斜率 ψ(T) = ρ·η·Γ(H+0.5)/(2√π) · T^{H-0.5}
    //   Γ(H+0.5)：当 H=0.1 → Γ(0.6) ≈ 1.489
    double gamma_val = std::tgamma(p.H + 0.5);
    double psi = p.rho * p.eta * gamma_val
                 / (2.0 * std::sqrt(M_PI))
                 * std::pow(T, p.H - 0.5);

    // 对数货币度 k = log(K/S)
    double k = std::log(K / S);

    // 偏斜修正：σ_K = σ_ATM + ψ·k，截断到最低 0.001
    double sigma_k = sigma_atm + psi * k;
    return std::max(sigma_k, 0.001);
}

// ── bs_price_and_delta() ─────────────────────────────────────
// 标准 Black-Scholes 公式（与 PricingEngine.cpp 一致的 d1/d2 逻辑）
// 此处接收已偏斜修正的 sigma，是 RoughVol 定价的最后一步。
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
// 主定价接口：偏斜修正波动率 → BS 公式 → {theo, delta}
PriceResult RoughVolPricingEngine::price(
    const Option& option,
    double underlying_price) const
{
    const double S = underlying_price;
    const double K = option.strike();

    // 计算到期年限 T
    auto now = std::chrono::system_clock::now();
    double T = std::chrono::duration<double>(option.expiry() - now).count()
               / (365.0 * 24.0 * 3600.0);
    T = std::max(T, 1e-6);

    // 偏斜修正波动率
    double sigma_atm = std::sqrt(params_.xi0);  // 仅用于日志，无锁读取近似值
    double sigma_k   = compute_skew_adjusted_vol(K, S, T);

    // 偏斜调整量（供日志观察）
    double skew_adj = sigma_k - sigma_atm;

    std::cout << "[粗糙波动率] " << option.id()
              << "  σ_ATM=" << std::fixed << std::setprecision(3) << sigma_atm
              << "  σ_K="   << sigma_k
              << "  偏斜调整=" << std::showpos << std::setprecision(3) << skew_adj
              << std::noshowpos << "\n";

    bool is_call = (option.option_type() == OptionType::Call);
    return bs_price_and_delta(S, K, T, sigma_k, is_call);
}

// ── update_params() ──────────────────────────────────────────
// 热注入校准结果（Phase 2 完成后由 main.cpp 调用）
// 线程安全：mutex 保护
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
