#pragma once
#include "../events/Events.hpp"

// ============================================================
// 文件：IAlphaSignal.hpp
// 职责：Alpha 信号纯接口 — 定义"如何从行情中产生交易信号"。
//
// 设计原则：
//   - 纯行为契约，不含 EventBus 或 register_handlers()
//   - 具体实现负责在内部发布 SignalGeneratedEvent（待 Phase 3 添加）
//   - 买方模块专用；卖方不需要 Alpha 信号
// ============================================================

namespace omm::core {

class IAlphaSignal {
public:
    virtual ~IAlphaSignal() = default;

    // 响应行情更新，计算信号（具体实现在内部发布 SignalGeneratedEvent）
    virtual void on_market_data(const events::MarketDataEvent& event) = 0;
};

} // namespace omm::core
