#pragma once
#include <vector>
#include <functional>
#include <utility>

// ============================================================
// 文件：CalibrationEngine.hpp
// 职责：收集模型预测值与市场观测值，通过优化算法校准模型参数。
//
// 校准目标（来自 Risk_Calibration.md §2.6）：
//   最小化均方误差（MSE）损失函数：
//     L(θ) = mean[(price_model(θ) - price_market)²]
//
// 优化算法：黄金分割搜索（Golden-Section Search）
//   - 1 维参数优化（此处为隐含波动率 σ）
//   - 区间收缩法，无需梯度，收敛稳健
//   - 时间复杂度：O(log((b-a)/ε))，约 50 次迭代达到 1e-6 精度
//
// 使用流程：
//   1. observe(market_price, model_price)  — 每个 tick 调用，积累观测
//   2. solve(vol_lo, vol_hi, loss_fn)      — 全局校准，返回最优参数
// ============================================================

namespace omm::domain {

// Observation — 单次价格观测（市场价 vs 模型当前预测价）
struct Observation {
    double market_price; // 从市场（或"真实"模型）观测到的期权价格
    double model_price;  // 待校准模型在当前参数下的预测价格
};

class CalibrationEngine {
public:
    CalibrationEngine() = default;

    // observe() — 记录一次价格观测
    //   在回测循环中每个 tick 调用：
    //     engine.observe(market_engine.price(...).theo,
    //                    model_engine.price(...).theo)
    void observe(double market_price, double model_price);

    // solve() — 运行黄金分割搜索，返回最优参数值
    //   参数：
    //     lo      — 参数搜索下界（如 σ_min = 0.01）
    //     hi      — 参数搜索上界（如 σ_max = 1.0）
    //     loss_fn — 损失函数：接受参数值 θ，返回对应损失（MSE）
    //               由调用方（BacktestCalibrationApp）提供，内部使用 pricing engine 重新评估
    //     tol     — 收敛精度（默认 1e-6）
    //   返回：使 loss_fn 最小的参数值
    double solve(
        double lo,
        double hi,
        std::function<double(double)> loss_fn,
        double tol = 1e-6
    ) const;

    // mse() — 计算当前所有观测的均方误差（供外部检查用）
    double mse() const;

    // observation_count() — 已积累的观测数量
    int observation_count() const;

private:
    std::vector<Observation> observations_; // 历史价格观测序列
};

} // namespace omm::domain
