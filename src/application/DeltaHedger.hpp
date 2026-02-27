#pragma once
#include <memory>
#include <vector>
#include <unordered_map> // HashMap 用于存储合约 ID 到 Delta 的映射
#include <string>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"
#include "../domain/Instrument.hpp"
#include "../domain/PositionManager.hpp"
#include "../domain/PricingEngine.hpp"

// ============================================================
// 文件：DeltaHedger.hpp
// 职责：实现 Delta 中性对冲策略（风险管理工作流）。
//
// 模式：策略模式（Strategy Pattern）
//   DeltaHedger 是一个可替换的风险对冲策略。
//   未来可替换为 GammaHedger（Gamma 对冲）或 VegaHedger（Vega 对冲）。
//
// 模式：命令模式（Command Pattern）
//   发布 OrderSubmittedEvent 将"提交对冲订单"的意图封装为命令对象，
//   DeltaHedger（命令发起者/Invoker）与订单执行方（OrderRouter/Receiver）完全解耦。
//
// 工作流程（每次成交后触发）：
//   1. 收到 TradeExecutedEvent → 更新持仓（通过 PositionManager）
//   2. 计算组合总 Delta 敞口：
//      Δ_portfolio = Δ_标的 × 持仓_标的 + Σ(Δ_期权_i × 持仓_期权_i)
//   3. 若 |Δ_portfolio| > 阈值（delta_threshold_）：
//      - 发布 OrderSubmittedEvent（市价单，方向与 Delta 敞口相反）
//      - 直接更新 PositionManager（模拟对冲成交，避免事件重入）
// ============================================================

namespace omm::application {

class DeltaHedger {
public:
    DeltaHedger(
        std::shared_ptr<events::EventBus>            bus,
        std::shared_ptr<domain::PositionManager>     position_manager,
        std::shared_ptr<domain::IPricingEngine>      pricing_engine, //计算Delta需要用到定价引擎
        std::vector<std::shared_ptr<domain::Option>> options,
        const std::string&                           underlying_id,
        double delta_threshold = 0.5  // MVP 低阈值，确保仿真中快速触发对冲
    );

    // 向 EventBus 注册 TradeExecutedEvent 处理器，应在 main.cpp 连线阶段调用
    void register_handlers();
    //初始化，在其中subscribe事件处理器

    // 更新当前标的资产市场价格（供 DeltaHedger 计算 Delta 时使用）
    // 通过订阅 MarketDataEvent 的 lambda 调用
    void update_market_price(double price);

private:
    // TradeExecutedEvent 的处理函数
    void on_trade_executed(const events::TradeExecutedEvent& event);

    // 构建各合约的 Delta 映射表（用于传入 PositionManager::compute_portfolio_delta)
    // snapshot of current deltas for all instruments
    std::unordered_map<std::string, double> compute_delta_map(
        double current_price
    ) const; //const成员函数，承诺不修改对象状态

    std::shared_ptr<events::EventBus>            bus_;             // 事件总线
    std::shared_ptr<domain::PositionManager>     position_manager_; // 持仓管理（注入）
    std::shared_ptr<domain::IPricingEngine>      pricing_engine_;  // 定价策略（注入）
    std::vector<std::shared_ptr<domain::Option>> options_;         // 期权列表
    std::string                                  underlying_id_;   // 标的资产 ID
    double                                       delta_threshold_; // 对冲触发阈值
    double                                       last_price_;      // 最近已知标的价格
};

} // namespace omm::application
