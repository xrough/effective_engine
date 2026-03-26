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
    std::vector<std::shared_ptr<domain::Option>> options,
    const std::string&                           underlying_id,
    double delta_threshold)
    : bus_(std::move(bus))
    , position_manager_(std::move(position_manager))
    , pricing_engine_(std::move(pricing_engine))
    , options_(std::move(options))
    , underlying_id_(underlying_id)
    , delta_threshold_(delta_threshold)
    , last_price_(150.0) {} // 初始价格假设（在第一条行情到达前的备用值）

void DeltaHedger::register_handlers() {
    // 模式：观察者 — 订阅 FillEvent（统一成交事件）
    bus_->subscribe<events::FillEvent>(
        [this](const events::FillEvent& evt) {
            this->on_fill(evt);
        }
    );
}

void DeltaHedger::update_market_price(double price) {
    // 由 main.cpp 中的 MarketDataEvent 订阅 lambda 调用
    // 确保 DeltaHedger 始终使用最新的标的资产价格计算 Delta
    last_price_ = price;
}

void DeltaHedger::on_fill(
    const events::FillEvent& event) {

    // --------------------------------------------------------
    // 第一步：更新持仓
    // 直接调用 PositionManager，而不是重新发布事件，
    // 避免对冲成交引发新的 FillEvent 导致无限递归。
    // --------------------------------------------------------
    position_manager_->on_fill(event);

    // --------------------------------------------------------
    // 第二步：计算组合总 Delta 敞口
    // Delta 映射表包含：
    //   - 标的资产：Delta 固定为 1.0（每股股价变动 $1，头寸价值变动 $1）
    //   - 各期权：由 IPricingEngine 计算（策略模式委托）
    // --------------------------------------------------------
    auto delta_map = compute_delta_map(last_price_);
    double portfolio_delta = position_manager_->compute_portfolio_delta(delta_map);

    std::cout << "[Delta对冲] 组合 Delta = "
              << std::fixed << std::setprecision(3) << portfolio_delta
              << "  (阈值 ±" << delta_threshold_ << ")\n";

    // --------------------------------------------------------
    // 第三步：判断是否需要对冲
    // --------------------------------------------------------
    if (std::abs(portfolio_delta) <= delta_threshold_) {
        std::cout << "[Delta对冲] Delta 在阈值内，无需对冲\n";
        return;
    }

    // --------------------------------------------------------
    // 计算对冲参数：
    //   组合 Delta 为正（净多头）→ 卖出标的资产以降低 Delta
    //   组合 Delta 为负（净空头）→ 买入标的资产以提升 Delta
    // --------------------------------------------------------
    int hedge_qty = static_cast<int>(std::round(std::abs(portfolio_delta)));
    events::Side hedge_side = (portfolio_delta > 0.0)
        ? events::Side::Sell  // Delta 过多：卖出标的
        : events::Side::Buy;  // Delta 过少：买入标的

    // --------------------------------------------------------
    // 模式：命令 — 将对冲指令封装为 OrderSubmittedEvent
    // DeltaHedger 作为"命令发起者（Invoker）"，不直接执行订单，
    // 只负责发布命令。OrderRouter（接收者/Receiver）决定如何执行。
    // --------------------------------------------------------
    events::OrderSubmittedEvent order{
        underlying_id_,
        hedge_side,
        hedge_qty,
        events::OrderType::Market // 使用市价单以确保立即成交
    };

    std::cout << "[Delta对冲] *** 触发对冲！"
              << " 发送市价单: "
              << (hedge_side == events::Side::Buy ? "买入" : "卖出")
              << " " << hedge_qty << " 股 " << underlying_id_
              << "  (当前 Delta=" << portfolio_delta << ")\n";

    // 发布对冲命令到 EventBus
    bus_->publish(order);

    // --------------------------------------------------------
    // MVP 简化：直接更新持仓（模拟市价单立即成交）
    // 不经过 EventBus，避免触发新一轮 FillEvent → DeltaHedger 的递归调用
    //
    // Phase 2 统一语义：FillEvent.side 表示我方方向，无需视角转换。
    //   hedge_side == Buy  → 我们买入标的 → PM 持仓 += qty（正确增加）
    //   hedge_side == Sell → 我们卖出标的 → PM 持仓 -= qty（正确减少）
    // --------------------------------------------------------
    events::FillEvent hedge_fill{
        underlying_id_,
        hedge_side,    // 直接使用我方对冲方向，无需翻转
        last_price_,
        hedge_qty,
        "hedge_order",
        event.timestamp
    };
    position_manager_->on_fill(hedge_fill);

    std::cout << "[Delta对冲] 对冲成交已直接记账（绕过事件总线，防止递归）\n";
}

std::unordered_map<std::string, double>
DeltaHedger::compute_delta_map(double current_price) const {
    std::unordered_map<std::string, double> deltas;

    // 标的资产的 Delta 恒为 1.0（每单位持仓对应每股 $1 的 Delta 敞口）
    deltas[underlying_id_] = 1.0;

    // 调用定价策略计算每个期权的 Delta
    for (const auto& option : options_) {
        domain::PriceResult result = pricing_engine_->price(*option, current_price);
        deltas[option->id()] = result.delta;
    }

    return deltas;
}

} // namespace omm::application
