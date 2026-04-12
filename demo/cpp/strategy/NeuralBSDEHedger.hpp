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
#include "execution/OnnxInference.hpp"
#include "analytics/LiftedHestonStateEstimator.hpp"

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

    // Called by the backtest loop (same contract as DeltaHedger::set_market_state).
    // atm_iv: market implied vol at ATM; T_sim: time-to-expiry in years.
    void set_market_state(double atm_iv, double T_sim) {
        tau_from_data_ = T_sim;
        // V_t proxy: use market ATM implied vol squared.
        // atm_iv ≈ 0.14–0.15 → V_t ≈ 0.02, consistent with training distribution
        // (training V_mean=0.025, V_std=0.070). The EWMA xi0 from rough_engine
        // accumulates to ~0.38 (5σ OOD) and must NOT be used here.
        atm_iv_ = atm_iv;
    }

private:
    void on_fill(const events::FillEvent& e) {
        position_mgr_->on_fill(e);
        // 标的对冲成交不触发重算，防止 SimpleExecSim → FillEvent 形成递归
        if (e.instrument_id == underlying_id_) return;
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

        // ── tau: from market data feed (not wall clock) ───────────────
        // NeuralBSDEHedger operates on historical data whose options may
        // have already expired relative to the current wall clock.
        // set_market_state() is called each bar with the actual tau from the feed.
        double tau = tau_from_data_;
        if (tau <= 0.0) return;
        tau = std::max(tau, 1e-4);

        double log_moneyness = std::log(spot_ / K_entry_);

        // V_t: use ATM implied vol squared from market feed.
        // atm_iv ≈ 0.14–0.15 → V_t ≈ 0.02, in-distribution (training mean=0.025, std=0.070).
        // The EWMA rough_engine xi0 accumulates to ~0.38 (5σ OOD) and must NOT be used.
        double V_t = atm_iv_ * atm_iv_;
        V_t = std::max(V_t, 1e-6);

        // All U factors are zeroed at inference. Rationale:
        // - The lrh_delta training target (BS delta PnL) is independent of U, so the
        //   optimal Z should depend only on (tau, logm, V_t). The model nonetheless
        //   learned U-correlated artifacts during training (U correlates with tau
        //   along LRH paths), causing Z → 0 for any U ≠ 0 even within-distribution.
        // - U[0]/U[1] (slow factors) read 0.02 at inference vs training mean ~0.002,
        //   shifting Z from 4.7 to -0.04. Setting U=0 recovers Z ≈ N(d1)*σ*K_train.
        // - U[2]/U[3] (fast factors, λ=215/10000): exponentially decay to ≈0 within
        //   each training step (dt_train=0.02 → λ*dt >> 1), so are already ≈0 in
        //   training data. The online estimator accumulates them with dt_tick<<1.
        std::vector<float> raw_state = {
            static_cast<float>(tau),
            static_cast<float>(log_moneyness),
            static_cast<float>(V_t),
            0.0f,   // U[0]: zeroed — see above
            0.0f,   // U[1]: zeroed — see above
            0.0f,   // U[2]: fast factor (λ=215), ≈0 in training
            0.0f    // U[3]: fast factor (λ=10000), ≈0 in training
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

        // ── Z_spot → delta ───────────────────────────────────
        // The model was trained with nominal strike K_train (stored in normalization.json).
        // Z_spot ≈ N(d1)·σ·K_train  (BM-space hedge coefficient at training scale).
        // Dividing by σ·K_train recovers delta = N(d1), regardless of current spot price.
        double sigma   = std::sqrt(std::max(V_t, 1e-6));
        double K_train = static_cast<double>(onnx_->k_train());
        double denom   = sigma * K_train;
        if (denom < 1e-10) return;

        double delta_call = static_cast<double>(Z_spot) / denom;
        double delta_put  = delta_call - 1.0;   // 看跌-看涨平价

        // ── 组合目标对冲量（标的股数，取整后与当前持仓比较） ──
        // Underlying hedge position = -option_delta (short when options are net long).
        // delta_call = N(d1) > 0 for long call; hedge must SHORT to cancel.
        // Equivalent to DeltaHedger: portfolio_delta > 0 → Sell underlying.
        double target_hedge     = -(call_qty * delta_call + put_qty * delta_put);
        int    target_round     = static_cast<int>(std::round(target_hedge));
        int    hedge_qty        = std::abs(target_round - current_hedge_qty_);

        std::cout << "[BSDE对冲]  Z_spot=" << std::fixed << std::setprecision(4) << Z_spot
                  << "  Δ_call=" << std::setprecision(3) << delta_call
                  << "  target=" << target_round
                  << "  current=" << current_hedge_qty_
                  << "  Δ_qty=" << (target_round - current_hedge_qty_) << "\n";

        if (hedge_qty <= static_cast<int>(threshold_)) return;

        // ── 提交对冲订单（直接发布 FillEvent，与 DeltaHedger 模式一致）────
        // Use FillEvent(producer="hedge_order") so ExtendedPnLTracker correctly
        // credits/debits delta_hedge_pnl_, identical to DeltaHedger::on_market_data.
        // Do NOT publish OrderSubmittedEvent — that would route through SimpleExecSim
        // and produce producer="alpha_exec", which is excluded from delta_hedge_pnl_.
        auto hedge_side = (target_round > current_hedge_qty_)
            ? events::Side::Buy    // 需要更多多头
            : events::Side::Sell;  // 需要更多空头

        // 先更新内部对冲量，防止 EventBus 同步递归重入
        current_hedge_qty_ = target_round;

        events::FillEvent hedge_fill{
            underlying_id_,
            hedge_side,
            spot_,
            hedge_qty,
            "hedge_order",
            std::chrono::system_clock::now()
        };
        position_mgr_->on_fill(hedge_fill);   // 更新持仓（不触发 recompute_hedge）
        bus_->publish(hedge_fill);             // ExtendedPnLTracker 通过此获取对冲 PnL

        std::cout << "[BSDE对冲] *** 触发对冲！"
                  << (hedge_side == events::Side::Buy ? "买入" : "卖出")
                  << " " << hedge_qty << " 股 " << underlying_id_
                  << "  推理延迟=" << signal.latency_us << " μs\n";
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
    double      threshold_;

    double spot_              = 150.0;  // 最新标的价格
    double tau_from_data_     = 0.25;   // time-to-expiry from market data (set via set_market_state)
    double atm_iv_            = 0.15;   // ATM implied vol from market feed; V_t = atm_iv_^2
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
