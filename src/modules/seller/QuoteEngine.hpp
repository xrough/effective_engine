#pragma once
#include <memory>
#include <vector>
#include "../../core/events/EventBus.hpp"
#include "../../core/events/Events.hpp"
#include "../../core/domain/Instrument.hpp"
#include "../../core/analytics/PricingEngine.hpp"

// ============================================================
// 文件：QuoteEngine.hpp
// 职责：实现做市商的报价生成策略（Sell-Side Quoting Strategy）。
//
// 模式：策略模式（Strategy Pattern）
//   IQuoteStrategy 是策略接口，定义"如何生成报价"的抽象。
//   QuoteEngine 是具体策略实现：基于理论价格 ± 固定价差生成买卖报价。
//
//   未来可以无缝替换为其他策略，例如：
//   - InventorySkewQuoteEngine：根据持仓方向调整报价偏斜
//   - VolSurfaceQuoteEngine：基于隐含波动率曲面动态调整价差
//   替换只需在 main.cpp 中注入不同的 IQuoteStrategy 实例。
//
// 工作流程：
//   1. 接收 MarketDataEvent（行情更新）
//   2. 对每个期权合约调用 IPricingEngine::price()（委托给定价策略）
//   3. 计算 bid = theo - half_spread，ask = theo + half_spread
//   4. 发布 QuoteGeneratedEvent
// ============================================================

namespace omm::application {

// ============================================================
// IQuoteStrategy — 报价策略接口
//
// 所有报价算法必须实现此接口。
// QuoteEngine 依赖此接口 + IPricingEngine，两者均可独立替换。
// ============================================================
class IQuoteStrategy {
public:
    virtual ~IQuoteStrategy() = default;

    // 响应行情更新，生成并发布报价
    virtual void on_market_data(const events::MarketDataEvent& event) = 0;
};

// ============================================================
// QuoteEngine — 固定价差报价引擎（具体策略）
//
// 报价公式：
//   bid = theo - half_spread   （做市商愿意买入的价格）
//   ask = theo + half_spread   （做市商愿意卖出的价格）
//   总价差（spread）= 2 × half_spread
//
// 默认 half_spread = 0.05，即总价差为 $0.10。
// ============================================================
class QuoteEngine final : public IQuoteStrategy {
public:
    QuoteEngine(
        std::shared_ptr<events::EventBus>            bus,
        std::shared_ptr<domain::IPricingEngine>      pricing_engine,
        std::vector<std::shared_ptr<domain::Option>> options,
        double half_spread = 0.05
    );

    // 向 EventBus 注册 MarketDataEvent 处理器，应在 main.cpp 连线阶段调用
    void register_handlers();

    // 策略接口实现：处理行情更新，为每个期权生成并发布报价
    void on_market_data(const events::MarketDataEvent& event) override;

private:
    std::shared_ptr<events::EventBus>            bus_;            // 事件总线（注入）
    std::shared_ptr<domain::IPricingEngine>      pricing_engine_; // 定价策略（注入）
    std::vector<std::shared_ptr<domain::Option>> options_;        // 待报价的期权合约列表
    double                                       half_spread_;    // 单边价差（美元）
};

} // namespace omm::application
