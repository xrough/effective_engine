#pragma once
#include "../events/Events.hpp"

// ============================================================
// 文件：IQuoteStrategy.hpp
// 职责：报价策略纯接口 — 定义"如何响应行情更新并生成报价"。
//
// 设计原则：
//   - 纯行为契约，不持有 EventBus 引用，不含 register_handlers()
//   - 具体实现（QuoteEngine）负责事件总线的订阅/发布
//   - 买方/卖方模块均可通过此接口扩展自定义报价逻辑
// ============================================================

namespace omm::core {

class IQuoteStrategy {
public:
    virtual ~IQuoteStrategy() = default;

    // 响应行情更新，生成报价（具体实现可发布 QuoteGeneratedEvent）
    virtual void on_market_data(const events::MarketDataEvent& event) = 0;
};

} // namespace omm::core
