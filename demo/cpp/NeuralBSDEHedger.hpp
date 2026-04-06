#pragma once

// ============================================================
// File: NeuralBSDEHedger.hpp  (demo/cpp/)
// Role: Neural BSDE-based hedge strategy for the alpha pipeline.
//
// BSDE 状态向量（7维）：[tau, log(S/K), V_t, U1, U2, U3, U4]
//   tau       = 到期剩余时间（年）
//   log(S/K)  = 对入场行权价的对数货币性
//   V_t       = LiftedHestonStateEstimator::get_V()（在线 OU 积分）
//   U1..U4    = LiftedHestonStateEstimator::get_U()（在线 OU 积分）
//
// Z_spot → delta 转换（Black-Scholes 极限恒等式）：
//   delta_call = Z_spot / (sigma * S)   sigma = sqrt(V_t)
//   delta_put  = delta_call - 1          (看跌-看涨平价)
//   net_delta  = call_qty * delta_call + put_qty * delta_put
// ============================================================

#ifdef BUILD_ONNX_DEMO

#include <memory>
#include <string>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"
#include "core/domain/PositionManager.hpp"
#include "core/analytics/RoughVolPricingEngine.hpp"
#include "OnnxInference.hpp"
#include "LiftedHestonStateEstimator.hpp"

namespace omm::demo {

class NeuralBSDEHedger {
public:
    NeuralBSDEHedger(
        std::shared_ptr<events::EventBus>               bus,
        std::shared_ptr<::demo::OnnxInference>          onnx,
        std::shared_ptr<domain::RoughVolPricingEngine>  rough_engine,
        std::shared_ptr<domain::PositionManager>        position_mgr,
        std::shared_ptr<LiftedHestonStateEstimator>     state_est,
        const std::string& underlying_id,
        const std::string& call_id,
        const std::string& put_id,
        double             strike_entry,
        std::chrono::system_clock::time_point T_expiry,
        double             threshold = 0.3
    )
        : bus_(std::move(bus))
        , onnx_(std::move(onnx))
        , rough_engine_(std::move(rough_engine))
        , position_mgr_(std::move(position_mgr))
        , state_est_(std::move(state_est))
        , underlying_id_(underlying_id)
        , call_id_(call_id)
        , put_id_(put_id)
        , K_entry_(strike_entry)
        , T_expiry_(T_expiry)
        , threshold_(threshold)
    {}

    void register_handlers() {
        bus_->subscribe<events::FillEvent>(
            [this](const events::FillEvent& e) { on_fill(e); }
        );
        bus_->subscribe<events::MarketDataEvent>(
            [this](const events::MarketDataEvent& e) { on_market_data(e); }
        );
    }

private:
    void on_fill(const events::FillEvent& event) {
        position_mgr_->on_fill(event);
        // 标的对冲成交不触发重算，防止 SimpleExecSim → FillEvent 形成递归
        if (event.instrument_id == underlying_id_) return;
        recompute_hedge();
    }

    void on_market_data(const events::MarketDataEvent& event) {
        spot_ = event.underlying_price;

        // ── 推进 LRH 状态估计器 ──────────────────────────────
        // dt 使用固定 tick 间隔近似（生产替换为实际时间差）
        constexpr double DT_ANNUAL = 1.0 / (252.0 * 390.0);
        double V_new = rough_engine_->get_params().xi0;  // xi0 作为 V_t 代理
        state_est_->update(DT_ANNUAL, V_new);

        recompute_hedge();
    }

    void recompute_hedge() {
        // ── 查询当前期权持仓 ──────────────────────────────────
        int call_qty = position_mgr_->get_position(call_id_);
        int put_qty  = position_mgr_->get_position(put_id_);
        if (call_qty == 0 && put_qty == 0) return;

        // ── 构建原始状态向量（使用在线 OU 因子） ─────────────
        auto now = std::chrono::system_clock::now();
        double tau = std::chrono::duration<double>(T_expiry_ - now).count()
                     / (365.25 * 24.0 * 3600.0);
        if (tau <= 0.0) return;
        tau = std::max(tau, 1e-4);

        double log_moneyness = std::log(spot_ / K_entry_);
        double V_t           = state_est_->get_V();          // 在线重构方差
        auto   U             = state_est_->get_U();          // [U1, U2, U3, U4]

        std::vector<float> raw_state = {
            static_cast<float>(tau),
            static_cast<float>(log_moneyness),
            static_cast<float>(V_t),
            static_cast<float>(U[0]),
            static_cast<float>(U[1]),
            static_cast<float>(U[2]),
            static_cast<float>(U[3])
        };

        // ── 归一化（使用 OnnxInference 中已加载的参数） ──────
        const auto& mean  = onnx_->norm_mean();
        const auto& stdev = onnx_->norm_std();
        std::vector<float> norm_state(7);
        for (int i = 0; i < 7; ++i) {
            norm_state[i] = (raw_state[i] - mean[i]) / (stdev[i] + 1e-8f);
        }

        // ── ONNX 推理 → Z_spot ──────────────────────────────
        auto signal  = onnx_->run(norm_state);
        float Z_spot = signal.Z_spot;

        // ── Z_spot → delta（BS 极限恒等式） ─────────────────
        double sigma = std::sqrt(std::max(V_t, 1e-6));
        double denom = sigma * spot_;
        if (denom < 1e-10) return;

        double delta_call = static_cast<double>(Z_spot) / denom;
        double delta_put  = delta_call - 1.0;   // 看跌-看涨平价

        // ── 组合目标对冲量（标的股数，取整后与当前持仓比较） ──
        double target_hedge     = call_qty * delta_call + put_qty * delta_put;
        int    target_round     = static_cast<int>(std::round(target_hedge));
        int    hedge_qty        = std::abs(target_round - current_hedge_qty_);

        std::cout << "[BSDE对冲]  Z_spot=" << std::fixed << std::setprecision(4) << Z_spot
                  << "  Δ_call=" << std::setprecision(3) << delta_call
                  << "  target=" << target_round
                  << "  current=" << current_hedge_qty_
                  << "  Δ_qty=" << (target_round - current_hedge_qty_) << "\n";

        if (hedge_qty <= static_cast<int>(threshold_)) return;

        // ── 提交对冲订单（命令模式，与 DeltaHedger 一致） ────
        auto hedge_side = (target_round > current_hedge_qty_)
            ? events::Side::Buy    // 需要更多多头
            : events::Side::Sell;  // 需要更多空头

        // 先更新内部对冲量，防止 EventBus 同步递归重入
        current_hedge_qty_ = target_round;

        events::OrderSubmittedEvent order{
            underlying_id_,
            hedge_side,
            hedge_qty,
            events::OrderType::Market
        };
        bus_->publish(order);

        auto U_log = state_est_->get_U();
        std::cout << "[BSDE对冲] *** 触发对冲！"
                  << (hedge_side == events::Side::Buy ? "买入" : "卖出")
                  << " " << hedge_qty << " 股 " << underlying_id_
                  << "  推理延迟=" << signal.latency_us << " μs"
                  << "  U=[" << std::fixed << std::setprecision(4)
                  << U_log[0] << "," << U_log[1] << "," << U_log[2] << "," << U_log[3] << "]\n";
    }

    // ── 成员变量 ───────────────────────────────────────────────
    std::shared_ptr<events::EventBus>               bus_;
    std::shared_ptr<::demo::OnnxInference>           onnx_;
    std::shared_ptr<domain::RoughVolPricingEngine>   rough_engine_;
    std::shared_ptr<domain::PositionManager>         position_mgr_;
    std::shared_ptr<LiftedHestonStateEstimator>      state_est_;

    std::string underlying_id_;
    std::string call_id_;
    std::string put_id_;
    double      K_entry_;
    std::chrono::system_clock::time_point T_expiry_;
    double      threshold_;

    double spot_              = 150.0;  // 最新标的价格
    int    current_hedge_qty_ = 0;      // 当前标的对冲净持仓
};

} // namespace omm::demo

#else // !BUILD_ONNX_DEMO ─────────────────────────────────────

// 无 ONNX Runtime 时的空壳：确保头文件始终可包含
#include <iostream>
#include <memory>
#include <string>
#include <chrono>

namespace omm::demo {

class NeuralBSDEHedger {
public:
    template<typename... Args>
    explicit NeuralBSDEHedger(Args&&...) {
        std::cout << "[NeuralBSDEHedger] 警告：BUILD_ONNX_DEMO 未启用，"
                     "此对冲器为空壳，不执行任何操作。\n";
    }
    void register_handlers() {}
};

} // namespace omm::demo

#endif // BUILD_ONNX_DEMO
