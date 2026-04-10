#include "PricingEngine.hpp"
#include <algorithm> // std::max
#include <cmath>     // std::log, std::sqrt, std::exp, std::erf
#include <chrono>    // 用于计算到期年份 T

// 模式：策略模式（Strategy Pattern）
// SimplePricingEngine 是 IPricingEngine 的一个具体策略实现。
// 替换定价模型时，只需新建类并实现 IPricingEngine，然后在 main.cpp 中替换注入实例。
namespace omm::domain {

PriceResult SimplePricingEngine::price(
    const Option& option,
    double underlying_price) const {

    double theo  = 0.0;
    double delta = 0.0;

    if (option.option_type() == OptionType::Call) {
        // 看涨期权内在价值：max(0, S - K)
        // 当标的价格 > 行权价时，该期权处于实值状态（In-the-Money），内在价值为 S-K
        // 当标的价格 <= 行权价时，期权处于虚值状态（Out-of-the-Money），内在价值为 0
        theo  = std::max(0.0, underlying_price - option.strike());
        // Delta 桩值：真实 Black-Scholes 中应为 N(d₁)（标准正态 CDF）
        // 此处简化为固定 +0.5，表示期权价格约随标的资产价格以 0.5 的比例变动
        delta = +0.5;
    } else {
        // 看跌期权内在价值：max(0, K - S)
        // 当行权价 > 标的价格时，持有看跌期权有价值（可以高价卖出低价资产）
        theo  = std::max(0.0, option.strike() - underlying_price);
        // Delta 桩值：真实 Black-Scholes 中应为 N(d₁) - 1（约为 -0.5 at-the-money）
        delta = -0.5;
    }

    return PriceResult{theo, delta};
}

PriceResult SimplePricingEngine::price_at_iv(
    double /*S*/, double /*K*/, double /*T_sim*/,
    double /*sigma_market*/, bool is_call) const
{
    // Intrinsic-value engine has no meaningful vol-based pricing.
    // Return stub with fixed delta (same as price()).
    return PriceResult{0.0, is_call ? 0.5 : -0.5};
}

// ============================================================
// BlackScholesPricingEngine 实现
// ============================================================

// N(x) — 标准正态累积分布函数（CDF），使用 std::erf 近似计算
// 公式：N(x) = 0.5 * (1 + erf(x / sqrt(2)))
static double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

BlackScholesPricingEngine::BlackScholesPricingEngine(double vol, double r)
    : vol_(vol), r_(r) {}

void BlackScholesPricingEngine::set_vol(double vol) {
    vol_ = vol;  // 运行时更新波动率（由 ParameterStore 注入校准结果）
}

double BlackScholesPricingEngine::get_vol() const {
    return vol_;
}

PriceResult BlackScholesPricingEngine::price(
    const Option& option,
    double underlying_price) const {

    const double S = underlying_price;        // 标的资产当前价格
    const double K = option.strike();          // 行权价
    const double sigma = vol_;                 // 年化波动率

    // 计算到期年限 T（从当前时刻到期权到期日）
    auto now = std::chrono::system_clock::now();
    auto time_to_expiry = option.expiry() - now;
    double T = std::chrono::duration<double>(time_to_expiry).count()
               / (365.0 * 24.0 * 3600.0); // 转换为年
    T = std::max(T, 1e-6); // 防止 T=0 导致除零

    // 计算 d₁ 和 d₂（Black-Scholes 核心公式）
    double d1 = (std::log(S / K) + (r_ + 0.5 * sigma * sigma) * T)
                / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    double theo  = 0.0;
    double delta = 0.0;

    if (option.option_type() == OptionType::Call) {
        // 看涨期权定价：Call = S·N(d₁) - K·e^(-rT)·N(d₂)
        theo  = S * norm_cdf(d1) - K * std::exp(-r_ * T) * norm_cdf(d2);
        // 看涨期权 Delta = N(d₁)（[0, 1] 范围内）
        delta = norm_cdf(d1);
    } else {
        // 看跌期权定价：Put = K·e^(-rT)·N(-d₂) - S·N(-d₁)
        theo  = K * std::exp(-r_ * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
        // 看跌期权 Delta = N(d₁) - 1（[-1, 0] 范围内）
        delta = norm_cdf(d1) - 1.0;
    }

    theo = std::max(0.0, theo); // 期权理论价格不得为负
    return PriceResult{theo, delta};
}

PriceResult BlackScholesPricingEngine::price_at_iv(
    double S, double K, double T_sim,
    double sigma_market, bool is_call) const
{
    const double sigma = sigma_market;
    double d1 = (std::log(S / K) + (r_ + 0.5 * sigma * sigma) * T_sim)
                / (sigma * std::sqrt(T_sim));
    double d2 = d1 - sigma * std::sqrt(T_sim);

    double theo  = 0.0;
    double delta = 0.0;
    const double n_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);

    if (is_call) {
        theo  = S * norm_cdf(d1) - K * std::exp(-r_ * T_sim) * norm_cdf(d2);
        delta = norm_cdf(d1);
    } else {
        theo  = K * std::exp(-r_ * T_sim) * norm_cdf(-d2) - S * norm_cdf(-d1);
        delta = norm_cdf(d1) - 1.0;
    }

    PriceResult result;
    result.theo  = std::max(0.0, theo);
    result.delta = delta;
    result.gamma = n_d1 / (S * sigma * std::sqrt(T_sim));
    result.vega  = S * std::sqrt(T_sim) * n_d1;
    result.theta = -(S * sigma * n_d1) / (2.0 * std::sqrt(T_sim))
                   + (is_call ? -1.0 : 1.0)
                     * r_ * K * std::exp(-r_ * T_sim)
                     * (is_call ? norm_cdf(d2) : norm_cdf(-d2));
    return result;
}

} // namespace omm::domain
