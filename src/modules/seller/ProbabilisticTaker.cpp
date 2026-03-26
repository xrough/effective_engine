#include "ProbabilisticTaker.hpp"
#include <iostream>
#include <iomanip>

namespace omm::infrastructure {

ProbabilisticTaker::ProbabilisticTaker(
    std::shared_ptr<events::EventBus> bus,
    double       trade_probability,
    unsigned int seed)
    : bus_(std::move(bus))
    , trade_probability_(trade_probability)
    , rng_(seed)                              // 固定种子，确保每次仿真结果一致
    , dist_(0.0, 1.0)                         // [0, 1) 均匀分布
    , side_dist_(0, 1) {}                     // {0, 1} 用于随机选择买/卖方向

void ProbabilisticTaker::register_handlers() {
    // 模式：观察者 — 将本组件注册为 QuoteGeneratedEvent 的订阅者
    bus_->subscribe<events::QuoteGeneratedEvent>(
        [this](const events::QuoteGeneratedEvent& evt) {
            this->on_quote_generated(evt);
        }
    );
}

void ProbabilisticTaker::on_quote_generated(
    const events::QuoteGeneratedEvent& event) {

    // 独立掷骰子：以 trade_probability_ 的概率决定是否触发成交
    double roll = dist_(rng_);
    if (roll >= trade_probability_) {
        // 本次不成交，静默跳过
        return;
    }

    // 随机选择成交方向（我方视角，Phase 2 统一语义）：
    //   0 → Hit Bid  → 客户卖出 → 我们买入（Side::Buy）→ 持仓增加
    //   1 → Lift Ask → 客户买入 → 我们卖出（Side::Sell）→ 持仓减少
    bool hit_bid = (side_dist_(rng_) == 0);
    events::Side side  = hit_bid ? events::Side::Buy  : events::Side::Sell;
    double       price = hit_bid ? event.bid_price     : event.ask_price;

    // 构造统一成交事件（FillEvent，我方视角）
    events::FillEvent fill{
        event.instrument_id, // 合约 ID 继承自报价事件
        side,
        price,
        1,                   // 固定成交量：1 手（MVP 简化）
        "customer_taker",    // 填单方标识
        event.timestamp
    };

    std::cout << "[成交模拟] "
              << event.instrument_id
              << "  " << (hit_bid ? "Hit Bid(我方买入)" : "Lift Ask(我方卖出)")
              << " @ $" << std::fixed << std::setprecision(4) << price
              << "  (掷骰结果: " << std::setprecision(2) << roll
              << " < " << trade_probability_ << ")\n";

    // 发布成交事件，触发 PortfolioService 和 DeltaHedger 的处理逻辑
    bus_->publish(fill);
}

} // namespace omm::infrastructure
