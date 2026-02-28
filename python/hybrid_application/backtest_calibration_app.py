# ============================================================
# 文件：hybrid_application/backtest_calibration_app.py
# 职责：回测与参数校准应用（混合模式）— 逻辑与
#       application/backtest_calibration_app.py 完全相同，
#       但使用 C++ omm_core 的所有核心对象。
#
# 对比纯 Python 版本的差异：
#   - 事件订阅：backtest_bus.subscribe_market_data(fn)
#   - 事件发布：main_bus.publish_param_update(event)
#   - market_engine / model_engine 均为 omm_core.BlackScholesPricingEngine
#   - calibrator 为 omm_core.CalibrationEngine（黄金分割搜索在 C++ 执行）
#   - loss_fn 是 Python 闭包，通过 pybind11/functional.h 传入 C++ solve()
#
# 事件订阅：MarketDataEvent（来自隔离的 backtest_bus 上的 C++ MarketDataAdapter）
# 事件发布：ParamUpdateEvent（发布到主总线 main_bus，由 C++ ParameterStore 接收）
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
import datetime

import omm_core  # C++ 扩展模块（核心层）


@dataclass
class _RawObs:
    """原始观测缓存（供 finalize() 在不同 σ 下重新计算模型价格）。"""
    underlying_price: float     # 该 tick 的标的资产价格
    option:           object    # omm_core.Option 对象
    market_price:     float     # 市场（高精度 BS vol=0.25）报价


class BacktestCalibrationApp:
    """回测与参数校准应用（混合模式 — Python 逻辑 + C++ 核心对象）。

    核心流程：
      market_engine（vol=0.25）生成"真实"市场价 → 与 model_engine（vol=0.15 初始）比对
      → CalibrationEngine 黄金分割搜索最优 σ → ParamUpdateEvent 发布到主总线
    """

    def __init__(
        self,
        backtest_bus:   omm_core.EventBus,
        main_bus:       omm_core.EventBus,
        market_engine:  omm_core.BlackScholesPricingEngine,  # vol=0.25，模拟真实市场
        model_engine:   omm_core.BlackScholesPricingEngine,  # vol=0.15，待校准
        options:        list,                                 # list[omm_core.Option]
        calibrator:     omm_core.CalibrationEngine,
        model_id:       str,
    ) -> None:
        self._backtest_bus  = backtest_bus
        self._main_bus      = main_bus
        self._market_engine = market_engine
        self._model_engine  = model_engine
        self._options       = options
        self._calibrator    = calibrator
        self._model_id      = model_id
        self._tick_count    = 0
        self._raw_obs: list[_RawObs] = []

    def register_handlers(self) -> None:
        """在回测专用 C++ EventBus 上注册行情处理器（与主仿真完全隔离）。"""
        self._backtest_bus.subscribe_market_data(self._on_market)

    def _on_market(self, event: omm_core.MarketDataEvent) -> None:
        """行情事件处理器（回测重放时调用）。

        对每个期权：
          market_price = C++ market_engine.price(opt, S).theo
          model_price  = C++ model_engine.price(opt, S).theo
          C++ calibrator.observe(market_price, model_price)
          缓存 _RawObs 供 finalize() 的 loss_fn 使用
        """
        self._tick_count += 1
        S = event.underlying_price

        for opt in self._options:
            market_price = self._market_engine.price(opt, S).theo
            model_price  = self._model_engine.price(opt, S).theo

            self._calibrator.observe(market_price, model_price)
            self._raw_obs.append(_RawObs(S, opt, market_price))

            diff = model_price - market_price
            print(
                f"[回测|混合|tick{self._tick_count:2d}] {opt.id}"
                f"  市场价=${market_price:.4f}"
                f"  模型价=${model_price:.4f}"
                f"  偏差=${diff:+.4f}"
            )

    def finalize(self) -> float:
        """运行参数校准（C++ 黄金分割搜索），发布 ParamUpdateEvent，
        返回最优隐含波动率。

        loss_fn 是 Python 闭包，通过 pybind11/functional.h 无缝传入
        C++ CalibrationEngine::solve()，实现跨语言的参数优化。
        """
        print("\n[回测校准|混合] ──── 开始参数优化 ────")
        print(
            f"[回测校准|混合] 模型 ID: {self._model_id}"
            f"  原始观测数: {len(self._raw_obs)}"
            f"  初始 MSE (vol=0.15): {self._calibrator.mse():.6f}"
        )

        # ── 损失函数：Python 闭包 → 通过 pybind11 传入 C++ solve() ──
        # 每次调用更新 C++ model_engine.vol 并重新评估所有期权价格
        def loss_fn(sigma: float) -> float:
            self._model_engine.set_vol(sigma)  # 更新 C++ 引擎波动率
            if not self._raw_obs:
                return 0.0
            sum_sq = sum(
                (self._model_engine.price(obs.option, obs.underlying_price).theo
                 - obs.market_price) ** 2
                for obs in self._raw_obs
            )
            return sum_sq / len(self._raw_obs)

        # ── 调用 C++ 黄金分割搜索（φ = (√5-1)/2，约 29 次迭代）────
        best_vol = self._calibrator.solve(0.01, 1.0, loss_fn)

        self._model_engine.set_vol(best_vol)
        error_pct = abs(best_vol - 0.25) / 0.25 * 100.0

        print(
            f"[回测校准|混合] ✅ 校准完成！\n"
            f"             初始波动率: 0.1500\n"
            f"             校准波动率: {best_vol:.4f}\n"
            f"             目标波动率: 0.2500（市场真实值）\n"
            f"             校准误差:   {error_pct:.4f}%"
        )

        # ── 构造 C++ ParamUpdateEvent 并发布到主总线 ───────────────
        update = omm_core.ParamUpdateEvent()
        update.model_id   = self._model_id
        update.params     = {"vol": best_vol, "r": 0.05}
        update.updated_at = datetime.datetime.now()
        print(
            f"[回测校准|混合] 📤 发布 ParamUpdateEvent → C++ ParameterStore"
            f"  模型: {self._model_id}  参数: vol={best_vol:.4f}"
        )
        self._main_bus.publish_param_update(update)

        return best_vol
