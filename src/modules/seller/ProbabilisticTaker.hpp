#pragma once
#include <memory>
#include <random>
#include "../../core/events/EventBus.hpp"
#include "../../core/events/Events.hpp"

// ============================================================
// 文件：ProbabilisticTaker.hpp
// 职责：模拟市场中的主动成交方（Aggressive Taker / Market Participant）。
//
// 行为逻辑：
//   - 订阅 QuoteGeneratedEvent（做市商每次报价后触发）
//   - 以固定概率（默认 10%）随机决定是否"成交"
//   - 若决定成交，随机选择 Hit Bid（客户卖出）或 Lift Ask（客户买入）
//   - 发布 TradeExecutedEvent，触发持仓更新和 Delta 对冲检查
//
// 设计说明：
//   本组件是事件链的中间环节，同时扮演"观察者"（订阅 QuoteGeneratedEvent）
//   和"发布者"（发布 TradeExecutedEvent）的双重角色，完美体现
//   事件驱动架构中组件间的完全解耦：ProbabilisticTaker 不持有任何对
//   QuoteEngine 或 DeltaHedger 的引用。
// ============================================================

namespace omm::infrastructure {

class ProbabilisticTaker {
public:
    // 构造函数参数：
    //   bus              — 事件总线（注入）
    //   trade_probability — 每条报价触发成交的概率（默认 10%）
    //   seed             — 随机数种子（固定种子确保仿真结果可重现）
    explicit ProbabilisticTaker( //explicit:禁止隐式类型转换，构造函数只能通过显式调用来创建对象
        std::shared_ptr<events::EventBus> bus,
        double       trade_probability = 0.10,
        unsigned int seed              = 42
    );

    // ----------------------------------------------------------
    // register_handlers() — 向 EventBus 注册事件处理器
    //
    // 模式：观察者 — 将 on_quote_generated() 注册为
    //   QuoteGeneratedEvent 的观察者。
    // 应在 main.cpp 的"连线阶段"调用一次。
    // ----------------------------------------------------------
    void register_handlers();

private:
    // QuoteGeneratedEvent 的处理函数
    void on_quote_generated(const events::QuoteGeneratedEvent& event);

    std::shared_ptr<events::EventBus>  bus_;              // 事件总线（注入）
    double                             trade_probability_; // 成交概率
    std::mt19937                       rng_;              // Mersenne Twister 随机数引擎
    std::uniform_real_distribution<>   dist_;             // [0, 1) 均匀分布
    std::uniform_int_distribution<>    side_dist_;        // {0, 1} 用于随机选择方向
};

} // namespace omm::infrastructure
