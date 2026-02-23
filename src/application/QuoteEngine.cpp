#include "QuoteEngine.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm> // std::max

// 模式：策略模式（Strategy Pattern）
// QuoteEngine 是 IQuoteStrategy 的一个具体实现。
// 其内部的 pricing_engine_ 又是另一个策略接口，体现了策略的可嵌套组合。
namespace omm::application {

QuoteEngine::QuoteEngine(
    std::shared_ptr<events::EventBus>            bus,
    std::shared_ptr<domain::IPricingEngine>      pricing_engine,
    std::vector<std::shared_ptr<domain::Option>> options,
    double half_spread)
    : bus_(std::move(bus))
    , pricing_engine_(std::move(pricing_engine))
    , options_(std::move(options))
    , half_spread_(half_spread) {}

void QuoteEngine::register_handlers() {
    // 模式：观察者 — 将本组件注册为 MarketDataEvent 的订阅者
    bus_->subscribe<events::MarketDataEvent>(
        [this](const events::MarketDataEvent& evt) {
            this->on_market_data(evt);
        }
    );
}

void QuoteEngine::on_market_data(const events::MarketDataEvent& event) {
    // 对投资组合中每个期权合约生成报价
    for (const auto& option : options_) {
        // --------------------------------------------------
        // 模式：策略 — 委托给注入的 IPricingEngine 进行定价
        // QuoteEngine 不关心是 Black-Scholes 还是简化模型
        // --------------------------------------------------
        domain::PriceResult result =
            pricing_engine_->price(*option, event.underlying_price);

        // 套用固定价差公式：bid = theo - ε，ask = theo + ε
        double bid = result.theo - half_spread_;
        double ask = result.theo + half_spread_;

        // 防止报价出现负数（期权价格不能为负）
        bid = std::max(0.0, bid);
        ask = std::max(0.0, ask);

        // 构造并发布报价事件
        events::QuoteGeneratedEvent quote{
            option->id(),
            bid,
            ask,
            event.timestamp
        };

        std::cout << "[报价引擎] " << option->id()
                  << "  Bid=$" << std::fixed << std::setprecision(4) << bid
                  << "  Ask=$" << ask
                  << "  (theo=$" << result.theo
                  << "  Δ=" << result.delta << ")\n";

        // 发布报价，触发 ProbabilisticTaker 的处理逻辑
        bus_->publish(quote);
    }
}

} // namespace omm::application
