#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace demo {

// ============================================================
// 提升粗糙Heston模型参数
// ============================================================
struct LiftedHestonParams {
    double S0    = 100.0;  // 初始股价
    double K     = 100.0;  // 行权价（欧式看涨期权）
    double T     = 1.0;    // 到期时间（年）
    double r     = 0.05;   // 无风险利率
    double V0    = 0.04;   // 初始方差（=xi0）
    double kappa = 0.3;    // 均值回归速度
    double theta = 0.04;   // 长期方差
    double xi    = 0.5;    // 波动率的波动率
    double rho   = -0.7;   // 现货-波动率相关系数
    double H     = 0.1;    // Hurst指数
    int    m     = 4;      // Markovian提升因子数量
};

// ============================================================
// 仿真批次输出
// state_tensor: [n_paths * n_times * state_dim]，行主序
//   每个时间步的状态 = [tau, log(S_t/K), V_t, U^1..U^m]
// dW1:          [n_paths * n_steps]，现货BM增量
// terminal_payoff: [n_paths]，折现欧式看涨期权收益
// tau_grid:     [n_times]，到期时间网格
// ============================================================
struct SimulationBatch {
    std::vector<float> state_tensor;    // 归一化状态
    std::vector<float> dW1;             // 现货BM增量（未归一化，用于BSDE训练）
    std::vector<float> terminal_payoff; // 折现收益
    std::vector<float> tau_grid;        // 到期时间
    int n_paths    = 0;
    int n_times    = 0;  // = n_steps + 1（含初始时刻）
    int n_steps    = 0;
    int state_dim  = 0;  // = 3 + m：tau, log(S/K), V_t, U^1..U^m
};

// ============================================================
// 归一化统计（每个特征维度的均值和标准差）
// ============================================================
struct NormStats {
    std::vector<float> mean;  // [state_dim]
    std::vector<float> std;   // [state_dim]
    int state_dim = 0;
};

// ============================================================
// 提升粗糙Heston仿真器
//
// 使用Markovian提升指数积分器。
// H=0.1, m=4 的核权重已预先计算并硬编码。
// ============================================================
class LiftedHestonSimulator {
public:
    explicit LiftedHestonSimulator(LiftedHestonParams params);

    // 生成批次路径
    // antithetic=true: 生成n_paths/2条基础路径 + n_paths/2条对立路径（方差缩减）
    SimulationBatch generate_batch(int n_paths, int n_steps,
                                   uint64_t seed,
                                   bool antithetic = true) const;

    // 从校准批次计算归一化统计并保存到JSON
    // 应使用独立的校准数据集（seed=0）调用此函数
    NormStats compute_norm_stats(const SimulationBatch& cal_batch) const;
    void save_normalization_json(const NormStats& stats,
                                 const std::string& path) const;

    const LiftedHestonParams& params() const { return params_; }

private:
    // 应用归一化到state_tensor（原地操作）
    void normalize_batch(SimulationBatch& batch, const NormStats& stats) const;

    LiftedHestonParams params_;

    // H=0.1, m=4 的预计算核权重（定义见 LiftedHestonSim.cpp）
    // 来源: markovian_lift_weights(hurst=0.1, n_factors=4)
    static const int    M_DEFAULT;
    static const double C_DEFAULT[4];
    static const double LAMBDA_DEFAULT[4];
};

} // namespace demo
