#include "CalibrationEngine.hpp"
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>

// ============================================================
// CalibrationEngine 实现
//
// 黄金分割搜索（Golden-Section Search）原理：
//   给定单峰函数 f 在区间 [a, b] 上的最小值，通过以黄金比例
//   φ = (√5 - 1) / 2 ≈ 0.618 收缩区间，每次迭代减少约 38.2% 的搜索范围。
//   不需要梯度信息，适合非光滑或高计算代价的损失函数。
// ============================================================

namespace omm::domain {

void CalibrationEngine::observe(double market_price, double model_price) {
    // 记录一次观测：市场价格与模型预测价格
    observations_.push_back(Observation{market_price, model_price});
}

double CalibrationEngine::solve(
    double lo,
    double hi,
    std::function<double(double)> loss_fn,
    double tol) const {

    if (observations_.empty()) {
        std::cout << "[校准引擎] 警告：无观测数据，返回初始参数中点\n";
        return (lo + hi) / 2.0;
    }

    std::cout << "[校准引擎] 开始黄金分割搜索，参数范围 ["
              << std::fixed << std::setprecision(4) << lo
              << ", " << hi << "]，观测数: " << observations_.size() << "\n";

    // 黄金比例倒数：φ = (√5 - 1) / 2 ≈ 0.618
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;

    // 初始化两个内部探测点（区间三等分的黄金分割点）
    double c = hi - phi * (hi - lo); // 左探测点（较小值）
    double d = lo + phi * (hi - lo); // 右探测点（较大值）

    int iterations = 0;

    while ((hi - lo) > tol) {
        ++iterations;
        double fc = loss_fn(c); // 左点损失
        double fd = loss_fn(d); // 右点损失

        if (fc < fd) {
            // 最优解在 [lo, d]，收缩右边界
            hi = d;
            d  = c;
            c  = hi - phi * (hi - lo);
        } else {
            // 最优解在 [c, hi]，收缩左边界
            lo = c;
            c  = d;
            d  = lo + phi * (hi - lo);
        }
    }

    double best_param = (lo + hi) / 2.0;

    std::cout << "[校准引擎] 搜索完成，迭代次数: " << iterations
              << "  最优参数: " << std::fixed << std::setprecision(6)
              << best_param
              << "  最终损失(MSE): " << std::setprecision(8) << loss_fn(best_param)
              << "\n";

    return best_param;
}

double CalibrationEngine::mse() const {
    if (observations_.empty()) return 0.0;

    // MSE = mean[(model_price - market_price)²]
    double sum_sq = 0.0;
    for (const auto& obs : observations_) {
        double err = obs.model_price - obs.market_price;
        sum_sq += err * err;
    }
    return sum_sq / static_cast<double>(observations_.size());
}

int CalibrationEngine::observation_count() const {
    return static_cast<int>(observations_.size());
}

} // namespace omm::domain
