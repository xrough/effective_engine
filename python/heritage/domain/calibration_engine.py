# ============================================================
# 文件：calibration_engine.py
# 职责：收集模型预测值与市场观测值，通过黄金分割搜索校准模型参数。
#
# 校准目标：最小化均方误差（MSE）损失函数：
#   L(θ) = mean[(price_model(θ) - price_market)²]
#
# 优化算法：黄金分割搜索（Golden-Section Search）
#   - 1 维参数优化（隐含波动率 σ）
#   - 区间收缩法，无需梯度，收敛稳健
#   - 黄金比例：φ = (√5 - 1) / 2 ≈ 0.618
# ============================================================

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class Observation:
    """单次价格观测（市场价 vs 模型当前预测价）。"""
    market_price: float  # 从市场（或"真实"模型）观测到的期权价格
    model_price:  float  # 待校准模型在当前参数下的预测价格


class CalibrationEngine:
    """参数校准引擎（黄金分割搜索优化器）。"""

    def __init__(self) -> None:
        self._observations: list[Observation] = []  # 历史价格观测序列

    def observe(self, market_price: float, model_price: float) -> None:
        """记录一次价格观测。

        在回测循环中每个 tick 调用：
          engine.observe(market_engine.price(...).theo,
                         model_engine.price(...).theo)
        """
        self._observations.append(Observation(market_price, model_price))

    def solve(
        self,
        lo: float,
        hi: float,
        loss_fn: Callable[[float], float],
        tol: float = 1e-6,
    ) -> float:
        """运行黄金分割搜索，返回使 loss_fn 最小的参数值。

        Args:
            lo:      参数搜索下界（如 σ_min = 0.01）
            hi:      参数搜索上界（如 σ_max = 1.0）
            loss_fn: 损失函数：接受参数值 θ，返回对应损失（MSE）
            tol:     收敛精度（默认 1e-6）

        Returns:
            使 loss_fn 最小的参数值
        """
        if not self._observations:
            print("[校准引擎] 警告：无观测数据，返回参数中点")
            return (lo + hi) / 2.0

        print(
            f"[校准引擎] 开始黄金分割搜索，参数范围 [{lo:.4f}, {hi:.4f}]，"
            f"观测数: {len(self._observations)}"
        )

        # 黄金比例倒数：φ = (√5 - 1) / 2 ≈ 0.618
        phi = (math.sqrt(5.0) - 1.0) / 2.0

        # 初始化两个内部探测点
        c = hi - phi * (hi - lo)  # 左探测点
        d = lo + phi * (hi - lo)  # 右探测点

        iterations = 0
        while (hi - lo) > tol:
            iterations += 1
            fc = loss_fn(c)
            fd = loss_fn(d)
            if fc < fd:
                # 最优解在 [lo, d]，收缩右边界
                hi = d
                d  = c
                c  = hi - phi * (hi - lo)
            else:
                # 最优解在 [c, hi]，收缩左边界
                lo = c
                c  = d
                d  = lo + phi * (hi - lo)

        best_param = (lo + hi) / 2.0
        print(
            f"[校准引擎] 搜索完成，迭代次数: {iterations}"
            f"  最优参数: {best_param:.6f}"
            f"  最终损失(MSE): {loss_fn(best_param):.8f}"
        )
        return best_param

    def mse(self) -> float:
        """计算当前所有观测的均方误差（供外部检查用）。"""
        if not self._observations:
            return 0.0
        total = sum(
            (obs.model_price - obs.market_price) ** 2
            for obs in self._observations
        )
        return total / len(self._observations)

    def observation_count(self) -> int:
        """已积累的观测数量。"""
        return len(self._observations)
