#include "PricingEngine.hpp"
#include <algorithm> // std::max

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

} // namespace omm::domain
