#include "PortfolioService.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

namespace omm::application {

PortfolioService::PortfolioService(
    std::shared_ptr<events::EventBus>            bus,
    std::shared_ptr<domain::IPricingEngine>      pricing_engine,
    std::vector<std::shared_ptr<domain::Option>> options,
    std::string                                  underlying_id,
    std::string                                  account_id)
    : bus_(std::move(bus))
    , pricing_engine_(std::move(pricing_engine))
    , underlying_id_(std::move(underlying_id))
    , account_id_(std::move(account_id))
    , portfolio_(account_id_, options)
    , last_price_(150.0) {}

void PortfolioService::register_handlers() {
    // 订阅统一成交事件（来自 ProbabilisticTaker、BrokerAdapter、DeltaHedger 对冲）
    bus_->subscribe<events::FillEvent>(
        [this](const events::FillEvent& evt) {
            this->on_fill(evt);
        }
    );

    // 订阅行情事件，用于 mark-to-market 重新估值
    bus_->subscribe<events::MarketDataEvent>(
        [this](const events::MarketDataEvent& evt) {
            this->on_market(evt);
        }
    );
}

void PortfolioService::on_fill(const events::FillEvent& event) {
    // ── 步骤 1：更新持仓（我方视角，无需视角转换）──────────
    portfolio_.applyFill(event);

    // ── 步骤 2：按最新市价重新估值 ───────────────────────
    portfolio_.markToMarket(*pricing_engine_, last_price_);

    // ── 步骤 3：发布持仓快照供风控应用消费 ───────────────
    publish_snapshot();
}

void PortfolioService::on_market(const events::MarketDataEvent& event) {
    // ── 步骤 1：更新标的价格 ──────────────────────────────
    last_price_ = event.underlying_price;

    // ── 步骤 2：按新价格重新估值 ─────────────────────────
    portfolio_.markToMarket(*pricing_engine_, last_price_);

    // ── 步骤 3：有持仓时发布快照（避免空仓噪音）─────────
    auto metrics = portfolio_.computeMetrics(*pricing_engine_, last_price_);
    if (metrics.delta != 0.0 || metrics.realized_pnl != 0.0
        || metrics.unrealized_pnl != 0.0) {
        events::PortfolioUpdateEvent update{
            account_id_,
            metrics,
            std::chrono::system_clock::now()
        };
        bus_->publish(update);
    }
}

void PortfolioService::publish_snapshot() {
    auto metrics = portfolio_.computeMetrics(*pricing_engine_, last_price_);

    std::cout << "[持仓服务|" << account_id_ << "] 快照:"
              << "  已实现盈亏=$" << std::fixed << std::setprecision(2)
              << metrics.realized_pnl
              << "  未实现盈亏=$" << metrics.unrealized_pnl
              << "  组合Δ=" << std::setprecision(3) << metrics.delta << "\n";

    events::PortfolioUpdateEvent update{
        account_id_,
        metrics,
        std::chrono::system_clock::now()
    };
    bus_->publish(update);
}

} // namespace omm::application
