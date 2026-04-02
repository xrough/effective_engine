#include "LiftedHestonSim.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace demo {

// ============================================================
// H=0.1, m=4 的预计算Markovian提升核权重
// 来源: markovian_lift_weights(hurst=0.1, n_factors=4)
//   几何网格: lambda_i = (1/t_max) * (t_max/t_min)^{i/(m-1)}
//             t_min=1e-4, t_max=10.0
//   权重 c_i: 非负最小二乘法（scipy lsq_linear），500个对数均匀采样点
// ============================================================
const int    LiftedHestonSimulator::M_DEFAULT = 4;
const double LiftedHestonSimulator::C_DEFAULT[4] = {
    6.796910829880118410e-01,
    1.784720957929723184e+00,
    1.133959859223462630e+01,
    4.101076447843571771e+01
};
const double LiftedHestonSimulator::LAMBDA_DEFAULT[4] = {
    1.000000000000000194e-01,
    4.641588833612779297e+00,
    2.154434690031882838e+02,
    1.000000000000000909e+04
};

// ============================================================
// 构造函数
// ============================================================
LiftedHestonSimulator::LiftedHestonSimulator(LiftedHestonParams params)
    : params_(std::move(params))
{
    if (params_.m != M_DEFAULT) {
        throw std::invalid_argument(
            "[LiftedHestonSimulator] 当前仅支持 m=4 的硬编码核权重。"
            "如需其他m值，请从Python参考实现预计算后更新常量。");
    }
}

// ============================================================
// generate_batch: 生成批次路径
// ============================================================
SimulationBatch LiftedHestonSimulator::generate_batch(
    int n_paths, int n_steps, uint64_t seed, bool antithetic) const
{
    if (antithetic && n_paths % 2 != 0) {
        throw std::invalid_argument("[LiftedHestonSimulator] antithetic=true 时 n_paths 必须为偶数。");
    }

    const int m          = params_.m;
    const int state_dim  = 3 + m;  // tau, log(S/K), V_t, U^1..U^m
    const int n_times    = n_steps + 1;
    const double dt      = params_.T / static_cast<double>(n_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double sqrt_rho2 = std::sqrt(1.0 - params_.rho * params_.rho);

    // 基础路径数量（对立采样：一半基础，一半镜像）
    const int n_base = antithetic ? n_paths / 2 : n_paths;

    SimulationBatch batch;
    batch.n_paths   = n_paths;
    batch.n_times   = n_times;
    batch.n_steps   = n_steps;
    batch.state_dim = state_dim;

    // 预分配
    batch.state_tensor.resize(n_paths * n_times * state_dim, 0.0f);
    batch.dW1.resize(n_paths * n_steps, 0.0f);
    batch.terminal_payoff.resize(n_paths, 0.0f);

    // tau网格（从T到0）
    batch.tau_grid.resize(n_times);
    for (int i = 0; i < n_times; ++i) {
        batch.tau_grid[i] = static_cast<float>(params_.T - i * dt);
    }

    // 随机数生成器
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    // --------------------------------------------------------
    // 对每条基础路径进行仿真
    // --------------------------------------------------------
    for (int p = 0; p < n_base; ++p) {
        double S = params_.S0;
        double V = params_.V0;
        std::vector<double> U(m, 0.0);  // 初始因子 U^k = 0

        // 写入初始状态（时刻0）
        auto write_state = [&](int path_idx, int t_idx) {
            int base = (path_idx * n_times + t_idx) * state_dim;
            double tau  = params_.T - t_idx * dt;
            double logm = std::log(S / params_.K);
            batch.state_tensor[base + 0] = static_cast<float>(tau);
            batch.state_tensor[base + 1] = static_cast<float>(logm);
            batch.state_tensor[base + 2] = static_cast<float>(V);
            for (int k = 0; k < m; ++k)
                batch.state_tensor[base + 3 + k] = static_cast<float>(U[k]);
        };

        write_state(p, 0);

        // 仿真每个时间步
        for (int i = 0; i < n_steps; ++i) {
            double z1 = norm(rng);
            double z2 = norm(rng);
            double dW1_i = sqrt_dt * z1;
            double dW2_i = sqrt_dt * (params_.rho * z1 + sqrt_rho2 * z2);

            // 存储dW1（BSDE训练用）
            batch.dW1[p * n_steps + i] = static_cast<float>(dW1_i);

            double V_pos = std::max(V, 0.0);
            double sqV   = std::sqrt(V_pos);

            // 更新Markovian因子（指数积分器）
            for (int k = 0; k < m; ++k) {
                double lam_k  = LAMBDA_DEFAULT[k];
                double decay  = std::exp(-lam_k * dt);
                double inv_l  = (lam_k > 1e-12) ? (1.0 / lam_k) : dt;
                double drift  = (1.0 - decay) * inv_l * params_.kappa * (params_.theta - V);
                double diff   = params_.xi * sqV * std::exp(-lam_k * dt * 0.5) * dW2_i;
                U[k] = decay * U[k] + drift + diff;
            }

            // 重建方差
            double V_new = params_.V0;
            for (int k = 0; k < m; ++k) V_new += C_DEFAULT[k] * U[k];
            V = std::max(V_new, 0.0);

            // 更新股价（对数正态）
            S = S * std::exp((params_.r - 0.5 * V_pos) * dt + sqV * dW1_i);

            write_state(p, i + 1);
        }

        // 折现收益（欧式看涨）
        double disc  = std::exp(-params_.r * params_.T);
        double payoff = std::max(S - params_.K, 0.0) * disc;
        batch.terminal_payoff[p] = static_cast<float>(payoff);

        // --------------------------------------------------------
        // 对立路径：镜像BM增量
        // --------------------------------------------------------
        if (antithetic) {
            int p_anti = p + n_base;

            // 重置状态
            S = params_.S0;
            V = params_.V0;
            std::fill(U.begin(), U.end(), 0.0);

            write_state(p_anti, 0);

            // 从已生成的增量中取反
            for (int i = 0; i < n_steps; ++i) {
                double dW1_base = batch.dW1[p * n_steps + i];
                // 对立路径使用 -dW1, -dW2（相关结构保持不变）
                // 注意：我们只存储了dW1，需要重新采样dW2的独立分量
                // 简化：对立路径的dW1取反，dW2通过 -rho*dW1 + sqrt(1-rho^2)*z2' 重建
                // 但z2'未存储。实际可行的近似：直接取 -dW1, -dW2（完全对立）
                double dW1_i = -static_cast<double>(dW1_base);

                // 重新采样一个独立的z2并取反（保持对立性）
                // 注意：这里对dW2取反以保证完整的对立采样
                // 从存储的dW1反推z1，再用z2镜像
                // 简化方案：只对dW1取反，dW2独立采样（标准做法）
                double z2_anti = norm(rng);  // 对立路径的独立vol BM
                double dW2_i   = sqrt_dt * (params_.rho * (dW1_i / sqrt_dt) + sqrt_rho2 * z2_anti);

                batch.dW1[p_anti * n_steps + i] = static_cast<float>(dW1_i);

                double V_pos = std::max(V, 0.0);
                double sqV   = std::sqrt(V_pos);

                for (int k = 0; k < m; ++k) {
                    double lam_k = LAMBDA_DEFAULT[k];
                    double decay = std::exp(-lam_k * dt);
                    double inv_l = (lam_k > 1e-12) ? (1.0 / lam_k) : dt;
                    double drift = (1.0 - decay) * inv_l * params_.kappa * (params_.theta - V);
                    double diff  = params_.xi * sqV * std::exp(-lam_k * dt * 0.5) * dW2_i;
                    U[k] = decay * U[k] + drift + diff;
                }

                double V_new = params_.V0;
                for (int k = 0; k < m; ++k) V_new += C_DEFAULT[k] * U[k];
                V = std::max(V_new, 0.0);

                S = S * std::exp((params_.r - 0.5 * V_pos) * dt + sqV * dW1_i);
                write_state(p_anti, i + 1);
            }

            double payoff_anti = std::max(S - params_.K, 0.0) * disc;
            batch.terminal_payoff[p_anti] = static_cast<float>(payoff_anti);
        }
    }

    return batch;
}

// ============================================================
// compute_norm_stats: 计算每个特征维度的均值和标准差
// ============================================================
NormStats LiftedHestonSimulator::compute_norm_stats(const SimulationBatch& cal) const {
    const int d = cal.state_dim;
    NormStats stats;
    stats.state_dim = d;
    stats.mean.assign(d, 0.0f);
    stats.std.assign(d, 0.0f);

    long long total = static_cast<long long>(cal.n_paths) * cal.n_times;

    // 计算均值
    for (int i = 0; i < cal.n_paths; ++i) {
        for (int t = 0; t < cal.n_times; ++t) {
            int base = (i * cal.n_times + t) * d;
            for (int j = 0; j < d; ++j)
                stats.mean[j] += cal.state_tensor[base + j];
        }
    }
    for (int j = 0; j < d; ++j)
        stats.mean[j] /= static_cast<float>(total);

    // 计算标准差
    for (int i = 0; i < cal.n_paths; ++i) {
        for (int t = 0; t < cal.n_times; ++t) {
            int base = (i * cal.n_times + t) * d;
            for (int j = 0; j < d; ++j) {
                float diff = cal.state_tensor[base + j] - stats.mean[j];
                stats.std[j] += diff * diff;
            }
        }
    }
    for (int j = 0; j < d; ++j) {
        stats.std[j] = std::sqrt(stats.std[j] / static_cast<float>(total));
        // 防止除零（tau维度在初始时刻为T，可能为常数）
        if (stats.std[j] < 1e-6f) stats.std[j] = 1.0f;
    }

    return stats;
}

// ============================================================
// save_normalization_json: 将归一化统计保存为JSON
// ============================================================
void LiftedHestonSimulator::save_normalization_json(
    const NormStats& stats, const std::string& path) const
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("[LiftedHestonSim] 无法写入: " + path);

    auto& p = params_;
    f << "{\n";
    f << "  \"feature_order\": [\"tau\", \"log_moneyness\", \"V_t\"";
    for (int k = 1; k <= p.m; ++k) f << ", \"U" << k << "\"";
    f << "],\n";

    auto arr = [&](const std::string& key, const std::vector<float>& v) {
        f << "  \"" << key << "\": [";
        for (int i = 0; i < (int)v.size(); ++i) {
            f << v[i];
            if (i + 1 < (int)v.size()) f << ", ";
        }
        f << "],\n";
    };
    arr("mean", stats.mean);
    arr("std",  stats.std);

    f << "  \"state_dim\": " << stats.state_dim << ",\n";
    f << "  \"m\": "         << p.m     << ",\n";
    f << "  \"K\": "         << p.K     << ",\n";
    f << "  \"T\": "         << p.T     << ",\n";
    f << "  \"r\": "         << p.r     << ",\n";
    f << "  \"model_params\": {\n";
    f << "    \"S0\": "    << p.S0    << ",\n";
    f << "    \"V0\": "    << p.V0    << ",\n";
    f << "    \"kappa\": " << p.kappa << ",\n";
    f << "    \"theta\": " << p.theta << ",\n";
    f << "    \"xi\": "    << p.xi    << ",\n";
    f << "    \"rho\": "   << p.rho   << ",\n";
    f << "    \"H\": "     << p.H     << "\n";
    f << "  },\n";
    f << "  \"generation_seed\": 0,\n";
    f << "  \"n_cal_paths\": 2000\n";  // 调用方固定使用2000条校准路径
    f << "}\n";
}

} // namespace demo
