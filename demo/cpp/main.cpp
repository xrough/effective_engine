#include "LiftedHestonSim.hpp"
#include "NpyWriter.hpp"
#ifdef BUILD_ONNX_DEMO
#include "OnnxInference.hpp"
#endif

#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
// 辅助函数：打印批次统计
// ============================================================
static void print_batch_stats(const demo::SimulationBatch& batch, const std::string& label) {
    const int d = batch.state_dim;

    // 计算 S_T（末时刻 log-moneyness 反推）和 V_T 的均值/标准差
    double mean_logm = 0, mean_V = 0, mean_payoff = 0;
    for (int p = 0; p < batch.n_paths; ++p) {
        int last_t = batch.n_times - 1;
        int base   = (p * batch.n_times + last_t) * d;
        mean_logm   += batch.state_tensor[base + 1];  // log(S_T/K)
        mean_V      += batch.state_tensor[base + 2];  // V_T
        mean_payoff += batch.terminal_payoff[p];
    }
    mean_logm   /= batch.n_paths;
    mean_V      /= batch.n_paths;
    mean_payoff /= batch.n_paths;

    std::cout << "[" << label << "]\n"
              << "  路径数:          " << batch.n_paths  << "\n"
              << "  时间步数:        " << batch.n_steps  << "\n"
              << "  状态维度:        " << batch.state_dim << "\n"
              << "  E[log(S_T/K)]:   " << std::fixed << std::setprecision(4) << mean_logm << "\n"
              << "  E[V_T]:          " << mean_V     << "\n"
              << "  E[折现收益]:     " << mean_payoff << "\n";
}

// ============================================================
// 辅助函数：计算BS解析价格（验证用）
// ============================================================
static double bs_call_price(double S, double K, double T, double r, double sigma) {
    if (T <= 0) return std::max(S - K, 0.0);
    auto norm_cdf = [](double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    };
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

int main() {
    // --------------------------------------------------------
    // 配置
    // --------------------------------------------------------
    demo::LiftedHestonParams params;
    params.S0    = 100.0;
    params.K     = 100.0;
    params.T     = 1.0;
    params.r     = 0.05;
    params.V0    = 0.04;   // 初始方差（对应 σ_0 ≈ 20%）
    params.kappa = 0.3;
    params.theta = 0.04;
    params.xi    = 0.5;
    params.rho   = -0.7;
    params.H     = 0.1;
    params.m     = 4;

    const int  N_CAL_PATHS   = 2000;   // 校准数据集（用于归一化）
    const int  N_TRAIN_PATHS = 10000;  // 训练数据集
    const int  N_OOS_PATHS   = 2000;   // 样本外基准测试数据集
    const int  N_STEPS       = 50;     // 时间步数
    const uint64_t SEED_CAL   = 0;
    const uint64_t SEED_TRAIN = 42;
    const uint64_t SEED_OOS   = 99;

    // artifacts目录（相对于demo/运行）
    const std::string ARTIFACTS = "artifacts";
    fs::create_directories(ARTIFACTS);

    std::cout << "==========================================================\n"
              << "  提升粗糙Heston + 深度BSDE Demo\n"
              << "==========================================================\n\n";

    // --------------------------------------------------------
    // 步骤1：构建仿真器
    // --------------------------------------------------------
    demo::LiftedHestonSimulator sim(params);
    std::cout << "[步骤1] 仿真器已初始化 (H=" << params.H
              << ", m=" << params.m << ")\n\n";

    // --------------------------------------------------------
    // 步骤2：生成校准数据集 → 计算归一化统计
    // --------------------------------------------------------
    std::cout << "[步骤2] 生成校准批次 (seed=" << SEED_CAL
              << ", n_paths=" << N_CAL_PATHS << ")...\n";
    auto cal_batch = sim.generate_batch(N_CAL_PATHS, N_STEPS, SEED_CAL, /*antithetic=*/true);
    print_batch_stats(cal_batch, "校准批次");

    auto norm_stats = sim.compute_norm_stats(cal_batch);
    std::string norm_path = ARTIFACTS + "/normalization.json";
    sim.save_normalization_json(norm_stats, norm_path);
    std::cout << "  归一化统计已保存: " << norm_path << "\n\n";

    // --------------------------------------------------------
    // 步骤3：生成训练批次（归一化后保存）
    // --------------------------------------------------------
    std::cout << "[步骤3] 生成训练批次 (seed=" << SEED_TRAIN
              << ", n_paths=" << N_TRAIN_PATHS << ", antithetic=true)...\n";
    auto train_batch = sim.generate_batch(N_TRAIN_PATHS, N_STEPS, SEED_TRAIN, /*antithetic=*/true);

    // 使用校准统计归一化训练批次
    // 注意：将归一化逻辑嵌入到state_tensor写入之前（这里在保存前手动应用）
    {
        int d = train_batch.state_dim;
        for (int i = 0; i < (int)train_batch.state_tensor.size(); ++i) {
            int j = i % d;
            train_batch.state_tensor[i] =
                (train_batch.state_tensor[i] - norm_stats.mean[j]) / norm_stats.std[j];
        }
    }

    // 同样归一化OOS批次（先生成后归一化）
    auto oos_batch = sim.generate_batch(N_OOS_PATHS, N_STEPS, SEED_OOS, /*antithetic=*/false);

    print_batch_stats(train_batch, "训练批次（归一化后）");

    // 保存训练批次
    {
        int np = train_batch.n_paths, nt = train_batch.n_times, sd = train_batch.state_dim;
        demo::save_npy_float32(ARTIFACTS + "/training_states.npy",
                               train_batch.state_tensor, {np, nt, sd});
        demo::save_npy_float32(ARTIFACTS + "/training_dW1.npy",
                               train_batch.dW1, {np, train_batch.n_steps});
        demo::save_npy_float32(ARTIFACTS + "/training_payoff.npy",
                               train_batch.terminal_payoff, {np});
        demo::save_npy_float32(ARTIFACTS + "/training_tau_grid.npy",
                               train_batch.tau_grid, {nt});
        std::cout << "  训练批次已保存至 " << ARTIFACTS << "/\n\n";
    }

    // 保存样本外批次（未归一化，用于基准测试时手动归一化）
    {
        int np = oos_batch.n_paths, ns = oos_batch.n_steps;
        demo::save_npy_float32(ARTIFACTS + "/oos_states_raw.npy",
                               oos_batch.state_tensor,
                               {np, oos_batch.n_times, oos_batch.state_dim});
        demo::save_npy_float32(ARTIFACTS + "/oos_dW1.npy",
                               oos_batch.dW1, {np, ns});
        demo::save_npy_float32(ARTIFACTS + "/oos_payoff.npy",
                               oos_batch.terminal_payoff, {np});
    }

    // --------------------------------------------------------
    // 验证检查：BS理论价格 vs Monte Carlo均值收益
    // --------------------------------------------------------
    double bs_price = bs_call_price(params.S0, params.K, params.T,
                                    params.r, std::sqrt(params.V0));
    double mc_mean  = 0;
    for (float v : oos_batch.terminal_payoff) mc_mean += v;
    mc_mean /= oos_batch.n_paths;

    std::cout << "[验证] BS解析价格 (σ=" << std::sqrt(params.V0) << "): "
              << std::fixed << std::setprecision(4) << bs_price << "\n"
              << "[验证] OOS MC均值收益: " << mc_mean
              << "  (注：LRH价格应与BS不同)\n\n";

    // --------------------------------------------------------
    // 步骤4-5: ONNX推理（仅在 BUILD_ONNX_DEMO=ON 时编译）
    // --------------------------------------------------------
#ifdef BUILD_ONNX_DEMO
    const std::string onnx_path = ARTIFACTS + "/neural_bsde.onnx";

    // 检查模型文件是否存在
    if (!fs::exists(onnx_path)) {
        std::cout << "[步骤4] 未找到ONNX模型: " << onnx_path << "\n"
                  << "  请先运行Python训练流程:\n"
                  << "    python python/trainer.py --config python/configs/bs_validation.yaml\n"
                  << "    python python/trainer.py --config python/configs/lifted_rough_heston.yaml\n"
                  << "    python python/export.py --checkpoint artifacts/checkpoints/best.pt\n\n";
    } else {
        std::cout << "[步骤4] 加载ONNX模型: " << onnx_path << "\n";
        demo::OnnxInference infer(onnx_path, ARTIFACTS + "/normalization.json");

        // 冒烟测试：对第一条路径的第一个时刻运行推理
        {
            int d = oos_batch.state_dim;
            std::vector<float> state0(oos_batch.state_tensor.begin(),
                                      oos_batch.state_tensor.begin() + d);
            // 应用归一化
            for (int j = 0; j < d; ++j)
                state0[j] = (state0[j] - norm_stats.mean[j]) / norm_stats.std[j];

            auto sig = infer.run(state0);
            std::cout << "  冒烟测试 — Y=" << sig.Y
                      << ", Z_spot=" << sig.Z_spot
                      << ", 延迟=" << sig.latency_us << " μs\n\n";
        }

        // --------------------------------------------------------
        // 步骤5: 样本外基准测试
        // --------------------------------------------------------
        std::cout << "[步骤5] 样本外基准测试 (n_paths=" << N_OOS_PATHS << ")...\n";

        // BS delta对冲 PnL
        std::vector<double> pnl_bs(N_OOS_PATHS, 0.0);
        // 神经网络对冲 PnL
        std::vector<double> pnl_nn(N_OOS_PATHS, 0.0);

        auto norm_cdf = [](double x) {
            return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
        };
        double sigma_bs = std::sqrt(params.V0);  // BS基准用初始波动率

        int d = oos_batch.state_dim;
        for (int p = 0; p < N_OOS_PATHS; ++p) {
            for (int i = 0; i < N_STEPS; ++i) {
                int base = (p * oos_batch.n_times + i) * d;
                float tau_i    = oos_batch.state_tensor[base + 0];
                float logm_i   = oos_batch.state_tensor[base + 1];  // log(S_t/K)

                // 当前和下一时刻的 S_t
                float logm_next = oos_batch.state_tensor[
                    (p * oos_batch.n_times + i + 1) * d + 1];
                double dS = params.K * (std::exp(logm_next) - std::exp(logm_i));

                // BS delta
                double d1 = (static_cast<double>(logm_i)
                             + (params.r + 0.5 * sigma_bs * sigma_bs) * tau_i)
                            / (sigma_bs * std::sqrt(std::max((double)tau_i, 1e-6)));
                double Z_bs = norm_cdf(d1);
                pnl_bs[p] += Z_bs * dS;

                // 神经网络 hedge signal
                std::vector<float> state_i(oos_batch.state_tensor.begin() + base,
                                            oos_batch.state_tensor.begin() + base + d);
                for (int j = 0; j < d; ++j)
                    state_i[j] = (state_i[j] - norm_stats.mean[j]) / norm_stats.std[j];

                auto sig = infer.run(state_i);
                pnl_nn[p] += sig.Z_spot * static_cast<float>(dS);
            }
            // 扣减收益
            double payoff = oos_batch.terminal_payoff[p];
            pnl_bs[p] -= payoff;
            pnl_nn[p] -= payoff;
        }

        // 统计
        auto stats = [&](const std::vector<double>& v) -> std::pair<double,double> {
            double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double var  = 0;
            for (double x : v) var += (x - mean) * (x - mean);
            return {mean, std::sqrt(var / v.size())};
        };

        auto [bs_mean, bs_std] = stats(pnl_bs);
        auto [nn_mean, nn_std] = stats(pnl_nn);
        auto lat = infer.get_latency_stats();

        std::cout << "\n  ========================================\n"
                  << "  对冲方法         PnL均值    PnL标准差\n"
                  << "  ----------------------------------------\n"
                  << std::fixed << std::setprecision(4)
                  << "  BS Delta对冲     " << bs_mean << "   " << bs_std << "\n"
                  << "  Neural BSDE      " << nn_mean << "   " << nn_std << "\n"
                  << "  ========================================\n"
                  << "  推理延迟  p50=" << std::setprecision(1) << lat.p50_us
                  << " μs,  p99=" << lat.p99_us << " μs\n\n";
    }
#else
    std::cout << "[步骤4-5] ONNX推理已禁用。\n"
              << "  运行Python训练后，使用以下命令重新构建:\n"
              << "    cmake .. -DBUILD_ONNX_DEMO=ON -DONNXRUNTIME_ROOT=$HOME/onnxruntime\n"
              << "    make\n\n";
#endif

    std::cout << "==========================================================\n"
              << "  路径生成完成。下一步:\n"
              << "  1. cd demo\n"
              << "  2. python python/trainer.py --config python/configs/bs_validation.yaml\n"
              << "  3. python python/trainer.py --config python/configs/lifted_rough_heston.yaml\n"
              << "  4. python python/export.py --checkpoint artifacts/checkpoints/best.pt\n"
              << "==========================================================\n";

    return 0;
}
