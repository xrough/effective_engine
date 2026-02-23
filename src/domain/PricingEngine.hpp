#pragma once
#include "Instrument.hpp"

// ============================================================
// 文件：PricingEngine.hpp
// 职责：定义期权定价引擎的抽象接口及 MVP 简化实现。
//
// 模式：策略模式（Strategy Pattern）
//   IPricingEngine 是策略接口（Strategy Interface），抽象"如何定价"。
//   SimplePricingEngine 是具体策略（Concrete Strategy），提供简化的内在价值定价。
//
//   策略模式的优势：
//   - 报价引擎（QuoteEngine）和对冲器（DeltaHedger）只依赖 IPricingEngine 接口，
//     不感知具体实现。
//   - 替换定价模型（例如将来接入完整的 Black-Scholes）只需：
//       1. 新建一个实现了 IPricingEngine 的类
//       2. 在 main.cpp 中替换注入的实例
//       3. 其他所有代码零改动
//   这是"开闭原则（Open/Closed Principle）"的典型应用。
// ============================================================

namespace omm::domain {

// ============================================================
// PriceResult — 定价结果值对象
//
// theo  — 理论公允价值（做市商基于模型估算的"中间价"）
// delta — 期权价格对标的资产价格的一阶敏感度（dV/dS）
//          看涨期权：[0, +1]  标的上涨则期权价值上升
//          看跌期权：[-1, 0]  标的上涨则期权价值下降
// ============================================================
struct PriceResult {
    double theo;   // 理论价格（美元）
    double delta;  // Delta 值（无量纲）
};

// ============================================================
// IPricingEngine — 策略接口（抽象基类）
//
// 所有定价模型必须实现此接口。
// ============================================================
class IPricingEngine {
public:
    virtual ~IPricingEngine() = default;

    // price() — 核心定价方法
    //
    // 参数：
    //   option           — 待定价的期权合约（含行权价、到期日、看涨/看跌）
    //   underlying_price — 标的资产当前市场价格
    //
    // 返回：PriceResult（theo + delta）
    virtual PriceResult price(
        const Option& option,
        double        underlying_price
    ) const = 0;
};

// ============================================================
// SimplePricingEngine — 简化定价引擎（MVP 具体策略）
//
// 定价逻辑（内在价值，无时间价值）：
//   看涨 theo  = max(0, S - K)  （只有实值部分，无时间溢价）
//   看跌 theo  = max(0, K - S)
//   Delta      = +0.5（看涨） / -0.5（看跌）（固定桩值，真实应为 N(d₁)）
//
// 注意：此实现有意简化，目的是展示策略模式的接口结构，
//       而非追求定价精度。替换为 Black-Scholes 只需实现 IPricingEngine
//       并在 main.cpp 注入新策略。
// ============================================================
class SimplePricingEngine final : public IPricingEngine {
public:
    PriceResult price(
        const Option& option,
        double        underlying_price
    ) const override;
};

} // namespace omm::domain
