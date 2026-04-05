#pragma once
#include <cmath>
#include <memory>
#include <optional>
#include <iostream>
#include "../../core/events/EventBus.hpp"
#include "../../core/events/Events.hpp"
#include "../../core/domain/RiskMetrics.hpp"
#include "../../core/interfaces/IEntryPolicy.hpp"
#include "VarianceAlphaSignal.hpp"

// ============================================================
// 文件：StrategyController.hpp
// 职责：方差 Alpha 策略控制器，实现 IEntryPolicy 接口。
//
// 状态机：Flat → Live → Cooldown → Flat
//   Flat     → Live     : |zscore| > z_entry AND valid AND calibration_ok
//   Live     → Cooldown : |zscore| < z_exit  OR 持仓超时  OR 止损触发
//   Cooldown → Flat     : 冷却期结束（cooldown_bars）
//
// v1 仅支持多空 ATM 前端跨式期权（LongFrontVariance / ShortFrontVariance）。
// 仓位 vega = base_vega * min(|zscore| / z_cap, 1.0)
//   zscore > 0  → 市场隐含方差偏贵 → short front variance（卖出跨式）
//   zscore < 0  → 市场隐含方差偏廉 → long  front variance（买入跨式）
// ============================================================

namespace omm::buyer {

enum class StrategyState { Flat, Live, Cooldown };

enum class TradeType { None, LongFrontVariance, ShortFrontVariance };

struct StrategyControllerConfig {
    int    max_holding_bars = 100;   // 最大持仓周期数（超时强制离场）
    double stop_loss_vega   = 2000.0; // vega 侧止损阈值（美元）
    int    cooldown_bars    = 20;    // 离场后冷却周期数
};

class StrategyController : public core::IEntryPolicy {
public:
    StrategyController(
        std::shared_ptr<events::EventBus> bus,
        AlphaSignalConfig signal_cfg,
        StrategyControllerConfig ctrl_cfg = {}
    ) : bus_(std::move(bus))
      , signal_cfg_(signal_cfg)
      , ctrl_cfg_(ctrl_cfg) {}

    void register_handlers() {
        bus_->subscribe<events::SignalSnapshotEvent>(
            [this](const events::SignalSnapshotEvent& s) { on_signal(s); }
        );
    }

    // IEntryPolicy 接口：根据最新信号和风险指标决定入场
    // 返回 nullopt 表示跳过（Flat/Cooldown 状态或信号无效）
    std::optional<core::OrderRequest> evaluate(
        const events::MarketDataEvent& /*market*/,
        const domain::RiskMetrics&    /*metrics*/
    ) override {
        if (state_ != StrategyState::Live) return std::nullopt;
        if (!latest_signal_.valid)          return std::nullopt;

        double zscore    = latest_signal_.zscore;
        double scale     = std::min(std::abs(zscore) / signal_cfg_.z_cap, 1.0);
        double target_vega = signal_cfg_.base_vega * scale;

        // 发布跨式主腿（认购）
        core::OrderRequest req;
        req.instrument_id = "ATM_CALL";  // 由执行层解析为实际合约
        req.side          = (current_trade_ == TradeType::LongFrontVariance)
                            ? events::Side::Buy : events::Side::Sell;
        req.quantity      = target_vega;  // v1 以 vega 为单位，执行层换算手数
        req.limit_price   = 0.0;          // 市价单
        req.strategy_id   = "VarianceAlpha";
        return req;
    }

    StrategyState state()        const { return state_; }
    TradeType     current_trade() const { return current_trade_; }

private:
    void on_signal(const events::SignalSnapshotEvent& sig) {
        latest_signal_ = sig;
        ++bars_in_state_;

        switch (state_) {
            case StrategyState::Flat:
                try_enter(sig);
                break;

            case StrategyState::Live:
                try_exit(sig);
                break;

            case StrategyState::Cooldown:
                if (--cooldown_remaining_ <= 0) {
                    state_ = StrategyState::Flat;
                    bars_in_state_ = 0;
                    std::cout << "[StrategyController] 冷却结束 → Flat\n";
                }
                break;
        }

        // 发布入场/离场指令（非空则提交订单）
        if (state_ == StrategyState::Live) {
            auto req = evaluate(events::MarketDataEvent{}, domain::RiskMetrics{});
            if (req.has_value()) {
                // 同时提交认沽腿（跨式第二腿）
                core::OrderRequest put_req = req.value();
                put_req.instrument_id = "ATM_PUT";
                bus_->publish(events::OrderSubmittedEvent{
                    req.value().instrument_id,
                    req.value().side,
                    static_cast<int>(req.value().quantity),
                    events::OrderType::Market
                });
                bus_->publish(events::OrderSubmittedEvent{
                    put_req.instrument_id,
                    put_req.side,
                    static_cast<int>(put_req.quantity),
                    events::OrderType::Market
                });
            }
        }
    }

    void try_enter(const events::SignalSnapshotEvent& sig) {
        if (!sig.valid || !sig.calibration_ok) return;
        if (std::abs(sig.zscore) <= signal_cfg_.z_entry) return;

        state_ = StrategyState::Live;
        bars_in_state_ = 0;
        current_trade_ = (sig.zscore > 0)
                         ? TradeType::ShortFrontVariance  // 隐含方差偏贵 → 做空
                         : TradeType::LongFrontVariance;  // 隐含方差偏廉 → 做多

        std::cout << "[StrategyController] Flat → Live"
                  << "  交易类型=" << (current_trade_ == TradeType::LongFrontVariance
                                        ? "LongFrontVariance" : "ShortFrontVariance")
                  << "  zscore=" << sig.zscore << "\n";
    }

    void try_exit(const events::SignalSnapshotEvent& sig) {
        bool signal_exit  = (std::abs(sig.zscore) < signal_cfg_.z_exit);
        bool timeout_exit = (bars_in_state_ >= ctrl_cfg_.max_holding_bars);

        if (signal_exit || timeout_exit) {
            std::cout << "[StrategyController] Live → Cooldown"
                      << (signal_exit ? "  (信号回归)" : "  (持仓超时)") << "\n";
            state_ = StrategyState::Cooldown;
            current_trade_ = TradeType::None;
            cooldown_remaining_ = ctrl_cfg_.cooldown_bars;
            bars_in_state_ = 0;
        }
    }

    std::shared_ptr<events::EventBus> bus_;
    AlphaSignalConfig                 signal_cfg_;
    StrategyControllerConfig          ctrl_cfg_;
    StrategyState                     state_               = StrategyState::Flat;
    TradeType                         current_trade_       = TradeType::None;
    int                               bars_in_state_       = 0;
    int                               cooldown_remaining_  = 0;
    events::SignalSnapshotEvent       latest_signal_       = {};
};

} // namespace omm::buyer
