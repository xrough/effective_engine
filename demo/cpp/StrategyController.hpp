#pragma once
#include <cmath>
#include <memory>
#include <optional>
#include <iostream>
#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "core/domain/RiskMetrics.hpp"
#include "core/interfaces/IEntryPolicy.hpp"
#include "VarianceAlphaSignal.hpp"

// ============================================================
// 文件：StrategyController.hpp  (demo/cpp/)
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
    int    max_holding_bars   = 100;    // 最大持仓周期数（超时强制离场）
    double stop_loss_vega     = 2000.0; // vega 侧止损阈值（美元）
    int    cooldown_bars      = 20;     // 离场后冷却周期数
    int    max_position       = 200;    // 最大持仓（硬上限，防止 vega 跳跃导致仓位暴涨）
    double min_vega_notional  = 5.0;    // ATM vega 下限（$），低于此值跳过入场
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

    // Called by demo before BuyerModule::install()
    void register_handlers() {
        bus_->subscribe<events::SignalSnapshotEvent>(
            [this](const events::SignalSnapshotEvent& s) { on_signal(s); }
        );
    }

    // IEntryPolicy 接口：入场逻辑已迁移至 try_enter()，此处仅做接口合规
    std::optional<core::OrderRequest> evaluate(
        const events::MarketDataEvent& /*market*/,
        const domain::RiskMetrics&    /*metrics*/
    ) override {
        return std::nullopt;  // 订单发布由 try_enter() 直接处理
    }

    StrategyState state()         const { return state_; }
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

    }

    void try_enter(const events::SignalSnapshotEvent& sig) {
        if (!sig.valid || !sig.calibration_ok) return;
        if (std::abs(sig.zscore) <= signal_cfg_.z_entry) return;

        // ATM vega 下限保护：vega 过小时跳过（近到期或零波动率 tick）
        if (sig.atm_vega < ctrl_cfg_.min_vega_notional) {
            std::cout << "[StrategyController] 跳过入场：ATM vega 过小 ("
                      << std::fixed << std::setprecision(2) << sig.atm_vega << ")\n";
            return;
        }

        state_ = StrategyState::Live;
        bars_in_state_ = 0;
        current_trade_ = (sig.zscore > 0)
                         ? TradeType::ShortFrontVariance
                         : TradeType::LongFrontVariance;

        // 基于信念度和 ATM vega 的仓位缩放，受硬上限约束
        double conviction = std::min(std::abs(sig.zscore) / signal_cfg_.z_cap, 1.0);
        int raw_qty = std::max(1, static_cast<int>(
                          signal_cfg_.base_vega * conviction / sig.atm_vega));
        int qty = std::min(raw_qty, ctrl_cfg_.max_position);

        std::cout << "[StrategyController] Flat → Live"
                  << "  交易类型=" << (current_trade_ == TradeType::LongFrontVariance
                                        ? "LongFrontVariance" : "ShortFrontVariance")
                  << "  zscore=" << std::fixed << std::setprecision(3) << sig.zscore
                  << "  conviction=" << std::setprecision(2) << conviction
                  << "  atm_vega=" << std::setprecision(2) << sig.atm_vega
                  << "  qty=" << qty << "\n";

        auto side = (current_trade_ == TradeType::LongFrontVariance)
                    ? events::Side::Buy : events::Side::Sell;

        // 入场时提交一次跨式两腿订单（仅在状态转换时触发）
        bus_->publish(events::OrderSubmittedEvent{
            "ATM_CALL", side, qty, events::OrderType::Market
        });
        bus_->publish(events::OrderSubmittedEvent{
            "ATM_PUT",  side, qty, events::OrderType::Market
        });
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
