#include "DeltaHedger.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>    // std::abs, std::round

// 模式：策略模式 — DeltaHedger 是可替换的风险管理策略
// 模式：命令模式 — 通过 OrderSubmittedEvent 封装对冲订单意图
namespace omm::application {

DeltaHedger::DeltaHedger(
    std::shared_ptr<events::EventBus>            bus,
    std::shared_ptr<domain::PositionManager>     position_manager,
    std::shared_ptr<domain::IPricingEngine>      pricing_engine,
    std::vector<OptionEntry>                     options,
    const std::string&                           underlying_id,
    double delta_threshold,
    HedgeMode mode)
    : bus_(std::move(bus))
    , position_manager_(std::move(position_manager))
    , pricing_engine_(std::move(pricing_engine))
    , options_(std::move(options))
    , underlying_id_(underlying_id)
    , delta_threshold_(delta_threshold)
    , last_price_(150.0)
    , hedge_mode_(mode) {}

void DeltaHedger::register_handlers() {
    // FillEvent — record positions ONLY; do NOT compute delta or hedge here.
    // Deferring the rebalance to MarketDataEvent ensures that all legs of a
    // multi-leg strategy (e.g. call + put straddle) are fully recorded before
    // we compute portfolio delta.  Without this deferral, the call fill fires
    // a buy-hedge and the put fill (same bar, same price) fires a sell-hedge,
    // netting to $0 PnL regardless of subsequent spot moves.
    bus_->subscribe<events::FillEvent>(
        [this](const events::FillEvent& evt) { this->on_fill(evt); }
    );
    // MarketDataEvent — once-per-bar rebalance after all bar fills settle.
    bus_->subscribe<events::MarketDataEvent>(
        [this](const events::MarketDataEvent& evt) { this->on_market_data(evt); }
    );
}

void DeltaHedger::update_market_price(double price) {
    last_price_ = price;
}

// ── on_fill: position accounting only ───────────────────────
void DeltaHedger::on_fill(const events::FillEvent& event) {
    position_manager_->on_fill(event);

    // Hedge fills update the underlying position but must not trigger another
    // rebalance cycle.
    if (event.producer == "hedge_order" || event.instrument_id == underlying_id_) {
        return;
    }

    needs_rebalance_ = true;
}

// ── on_market_data: once-per-bar delta rebalance ─────────────
void DeltaHedger::on_market_data(const events::MarketDataEvent& event) {
    last_price_ = event.underlying_price;

    if (!needs_rebalance_) return;
    needs_rebalance_ = false;

    auto delta_map       = compute_delta_map(last_price_);
    double portfolio_delta = position_manager_->compute_portfolio_delta(delta_map);

    if (std::abs(portfolio_delta) <= delta_threshold_) return;

    int hedge_qty = static_cast<int>(std::round(std::abs(portfolio_delta)));
    events::Side hedge_side = (portfolio_delta > 0.0)
        ? events::Side::Sell
        : events::Side::Buy;

    std::cout << "[Delta对冲] *** 触发对冲！"
              << (hedge_side == events::Side::Buy ? " 买入 " : " 卖出 ")
              << hedge_qty << " 股 " << underlying_id_
              << "  (Delta=" << std::fixed << std::setprecision(2)
              << portfolio_delta << "  spot=" << last_price_ << ")\n";

    events::OrderSubmittedEvent order{
        underlying_id_,
        hedge_side,
        hedge_qty,
        events::OrderType::Market
    };
    order.reference_price = last_price_;
    order.producer = "hedge_order";
    order.timestamp = event.timestamp;
    bus_->publish(order);
}

std::unordered_map<std::string, double>
DeltaHedger::compute_delta_map(double current_price) const {
    std::unordered_map<std::string, double> deltas;

    // Underlying delta is always 1.0
    deltas[underlying_id_] = 1.0;

    // options_ is vector<pair<fill_id, option_ptr>>.
    // Key by fill_id so it matches PositionManager keys ("ATM_CALL" etc.)
    const bool use_market_state = (market_iv_ > 0.0 && market_T_ > 0.0);
    for (const auto& entry : options_) {
        const std::string&                     fill_id = entry.first;
        const std::shared_ptr<domain::Option>& option  = entry.second;
        domain::PriceResult result;
        if (use_market_state) {
            const bool is_call =
                (option->option_type() == domain::OptionType::Call);
            result = (hedge_mode_ == HedgeMode::ROUGH_DELTA)
                ? pricing_engine_->price_with_rough_delta(
                      current_price, option->strike(),
                      market_T_, market_iv_, is_call)
                : pricing_engine_->price_at_iv(
                      current_price, option->strike(),
                      market_T_, market_iv_, is_call);
        } else {
            result = pricing_engine_->price(*option, current_price);
        }
        deltas[fill_id] = result.delta;
    }

    return deltas;
}

} // namespace omm::application
