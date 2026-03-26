#pragma once
#include "../events/Events.hpp"

// ============================================================
// 文件：IHedgeStrategy.hpp
// 职责：对冲策略纯接口 — 定义"如何响应成交与行情更新执行对冲"。
//
// 设计原则：
//   - 纯行为契约，不含 EventBus 或 register_handlers()
//   - 具体实现（DeltaHedger、GammaHedger 等）负责总线订阅
//   - 仅卖方模块需要对冲；买方使用独立的头寸管理逻辑
// ============================================================

namespace omm::core {

class IHedgeStrategy {
public:
    virtual ~IHedgeStrategy() = default;

    // 成交后触发：更新持仓，检查对冲条件
    virtual void on_fill(const events::FillEvent& event) = 0;

    // 行情更新后触发：刷新用于 Delta 计算的标的价格
    virtual void on_market_data(const events::MarketDataEvent& event) = 0;
};

} // namespace omm::core
