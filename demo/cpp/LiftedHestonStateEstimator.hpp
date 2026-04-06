#pragma once
#include <array>
#include <cmath>
#include <algorithm>

// ============================================================
// File: LiftedHestonStateEstimator.hpp  (demo/cpp/)
// Role: Online integration of the Markovian lift factors U_k for the
//       Lifted Rough Heston model, driven by observed variance changes.
//
// Purpose:
//   NeuralBSDEHedger needs the 7D state [tau, log(S/K), V_t, U1..U4]
//   every tick. Without this estimator, U1..U4=0 always (t=0 approx),
//   which is increasingly wrong as the option ages. This estimator
//   integrates U_k forward using observed V_t as a proxy for the
//   unobservable vol BM dW2.
//
// Model (from LiftedHestonSim.cpp, exponential integrator):
//   V_t   = V0 + sum_k(c_k * U_k)
//   dU_k  = -lam_k * U_k * dt + kappa*(theta - V)*dt + xi*sqrt(V)*dW2
//
// dW2 inversion from observed V_t change:
//   sum_ce   = sum_k(c_k * xi * sqrt(V_safe) * exp(-lam_k*dt/2))
//   drift_V  = sum_k(c_k * (1-exp(-lam_k*dt))/lam_k * kappa*(theta-V))
//   dW2_hat  = (V_new - V_prev - drift_V) / sum_ce
//   (clamped ±5, zero if |sum_ce| < 1e-10)
//
// Guards:
//   V_FLOOR = 1e-6  clamped before every sqrt(V)
//   |dW2_hat| clamped to ±5 to suppress outlier spikes
// ============================================================

namespace omm::demo {

class LiftedHestonStateEstimator {
public:
    // LRH params from normalization.json model_params
    LiftedHestonStateEstimator(
        double kappa  = 0.3,
        double theta  = 0.04,
        double xi     = 0.5,
        double V0     = 0.04
    )
        : kappa_(kappa), theta_(theta), xi_(xi), V0_(V0)
        , V_curr_(V0)
    {
        U_.fill(0.0);
    }

    // Call on each MarketDataEvent tick with the new variance estimate.
    // dt: time step in years (e.g. 1/252/390 for 1-second ticks)
    // V_new: current variance proxy (xi0 from RoughVolPricingEngine)
    void update(double dt, double V_new) {
        if (dt <= 0.0) return;

        constexpr double V_FLOOR = 1e-6;
        double V_safe = std::max(V_curr_, V_FLOOR);

        // ── 计算 sum_ce（dW2 推断分母） ──────────────────────
        double sum_ce    = 0.0;
        double drift_V   = 0.0;
        for (int k = 0; k < 4; ++k) {
            double decay  = std::exp(-LAM[k] * dt);
            double inv_l  = (LAM[k] > 1e-12) ? (1.0 / LAM[k]) : dt;
            sum_ce  += C[k] * xi_ * std::sqrt(V_safe) * std::exp(-LAM[k] * dt * 0.5);
            drift_V += C[k] * (1.0 - decay) * inv_l * kappa_ * (theta_ - V_safe);
        }

        // ── dW2 推断（分母保护 + 截断） ───────────────────────
        double dW2_hat = 0.0;
        if (std::abs(sum_ce) >= 1e-10) {
            dW2_hat = (V_new - V_curr_ - drift_V) / sum_ce;
            dW2_hat = std::max(-5.0, std::min(5.0, dW2_hat));  // 截断离群值
        }

        // ── U_k 步进（指数积分器，与 LiftedHestonSim.cpp L124-131 一致） ──
        for (int k = 0; k < 4; ++k) {
            double decay = std::exp(-LAM[k] * dt);
            double inv_l = (LAM[k] > 1e-12) ? (1.0 / LAM[k]) : dt;
            double drift = (1.0 - decay) * inv_l * kappa_ * (theta_ - V_safe);
            double diff  = xi_ * std::sqrt(V_safe) * std::exp(-LAM[k] * dt * 0.5) * dW2_hat;
            U_[k] = decay * U_[k] + drift + diff;
        }

        // ── V 重构并钳位 ──────────────────────────────────────
        double V_recon = V0_;
        for (int k = 0; k < 4; ++k) V_recon += C[k] * U_[k];
        V_curr_ = std::max(V_recon, V_FLOOR);
    }

    // 当前 Markovian 提升因子
    std::array<double, 4> get_U() const { return U_; }

    // 当前方差估计（重构值）
    double get_V() const { return V_curr_; }

private:
    // LRH 常数（来自 LiftedHestonSim.cpp 默认值）
    static constexpr double C[4]   = {0.6796910829880118,  1.7847209579297232,
                                       11.339598592234626,  41.010764478435718};
    static constexpr double LAM[4] = {0.1, 4.641588833612779,
                                       215.4434690031883, 10000.0};

    double kappa_, theta_, xi_, V0_;
    double V_curr_;
    std::array<double, 4> U_;
};

} // namespace omm::demo
