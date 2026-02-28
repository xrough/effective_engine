# ============================================================
# 文件：backtest_calibration_app.py
# 职责：回测与参数校准应用 — 在隔离的事件总线上重放历史行情，
#       比对"市场真实价格"与"模型预测价格"，优化隐含波动率 σ，
#       最终通过 ParamUpdateEvent 将校准结果发布到系统。
#
# 对应 Risk_Calibration.md §2（Backtest & Calibration Application）
#
# 参数反馈闭环（Risk_Calibration.md §2.7）：
#   BacktestCalibrationApp
#         ↓ 发布 ParamUpdateEvent（到主总线）
#   ParameterStore（参数仓库，存储带时间戳的校准结果）
#         ↓
#   实时定价引擎（查询 get_params() 获取最新 σ）
#
# 事件订阅：MarketDataEvent（在隔离的 backtest_bus 上）
# 事件发布：ParamUpdateEvent（发布到主总线 main_bus）
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime

from events.event_bus import EventBus
from events.events import MarketDataEvent, ParamUpdateEvent
from domain.instrument import Option
from domain.pricing_engine import BlackScholesPricingEngine
from domain.calibration_engine import CalibrationEngine


@dataclass
class _RawObs:
    """原始观测：保存每个 tick 的（标的价格、期权合约、市场价格），
    供 finalize() 在不同 σ 下重新计算模型价格，实现真正的参数扫描。
    """
    underlying_price: float   # 该 tick 的标的资产价格
    option:           Option  # 对应期权合约
    market_price:     float   # 市场（高精度 BS）报价


class BacktestCalibrationApp:
    """回测与参数校准应用。

    核心概念：
      market_engine — 代表真实市场的价格来源（vol=0.25，模拟交易所行情）
      model_engine  — 待校准的内部定价模型（初始 vol=0.15）
      校准目标：最小化 MSE(model_price, market_price)，拟合 σ → 0.25
    """

    def __init__(
        self,
        backtest_bus:   EventBus,
        main_bus:       EventBus,
        market_engine:  BlackScholesPricingEngine,  # "真实"市场价格来源
        model_engine:   BlackScholesPricingEngine,  # 待校准的内部模型
        options:        list[Option],
        calibrator:     CalibrationEngine,
        model_id:       str,
    ) -> None:
        self._backtest_bus  = backtest_bus   # 隔离回测总线（状态独立）
        self._main_bus      = main_bus       # 主系统总线（发布校准结果）
        self._market_engine = market_engine  # "市场真实"定价引擎
        self._model_engine  = model_engine   # 待校准的内部模型
        self._options       = options
        self._calibrator    = calibrator
        self._model_id      = model_id
        self._tick_count    = 0
        self._raw_obs: list[_RawObs] = []   # 原始观测缓存

    def register_handlers(self) -> None:
        """在回测专用总线上注册行情处理器（与主仿真完全隔离）。"""
        self._backtest_bus.subscribe(MarketDataEvent, self._on_market)

    def _on_market(self, event: MarketDataEvent) -> None:
        """行情事件处理器（回测重放时调用）。

        对每个期权：
          market_price = market_engine.price(opt, S).theo  （"真实"市场价）
          model_price  = model_engine.price(opt, S).theo   （模型当前预测价）
          calibrator.observe(market_price, model_price)    （记录偏差）
          raw_obs.append(...)                               （缓存原始数据）
        """
        self._tick_count += 1
        S = event.underlying_price

        for opt in self._options:
            # "市场"价格：高精度 BS（vol=0.25），模拟真实市场报价
            market_price = self._market_engine.price(opt, S).theo
            # 模型当前预测价格（vol=0.15，初始估计）
            model_price  = self._model_engine.price(opt, S).theo

            # 记录到 CalibrationEngine（用于初始 MSE 报告）
            self._calibrator.observe(market_price, model_price)
            # 缓存原始观测（供 finalize() 的 loss_fn 使用）
            self._raw_obs.append(_RawObs(S, opt, market_price))

            diff = model_price - market_price
            print(
                f"[回测|tick{self._tick_count:2d}] {opt.id}"
                f"  市场价=${market_price:.4f}"
                f"  模型价=${model_price:.4f}"
                f"  偏差=${diff:+.4f}"
            )

    def finalize(self) -> float:
        """运行参数校准，发布 ParamUpdateEvent，返回最优隐含波动率。

        流程：
          1. 构造 loss_fn(σ)：set_vol(σ) → 重算所有 raw_obs 的 MSE
          2. 调用 calibrator.solve(0.01, 1.0, loss_fn)
          3. 将最优 σ 应用到 model_engine
          4. 发布 ParamUpdateEvent 到主总线
        """
        print("\n[回测校准] ──── 开始参数优化 ────")
        print(
            f"[回测校准] 模型 ID: {self._model_id}"
            f"  原始观测数: {len(self._raw_obs)}"
            f"  初始 MSE (vol=0.15): {self._calibrator.mse():.6f}"
        )

        # ── 损失函数：给定 σ 重新计算所有观测的 MSE ─────
        # 每次调用都更新 model_engine.vol 并重新评估所有期权价格
        def loss_fn(sigma: float) -> float:
            self._model_engine.set_vol(sigma)  # 临时更新模型波动率
            if not self._raw_obs:
                return 0.0
            sum_sq = sum(
                (self._model_engine.price(obs.option, obs.underlying_price).theo
                 - obs.market_price) ** 2
                for obs in self._raw_obs
            )
            return sum_sq / len(self._raw_obs)

        # ── 黄金分割搜索：在 [0.01, 1.0] 寻找最优 σ ────
        best_vol = self._calibrator.solve(0.01, 1.0, loss_fn)

        # 将校准结果应用到模型引擎
        self._model_engine.set_vol(best_vol)
        error_pct = abs(best_vol - 0.25) / 0.25 * 100.0

        print(
            f"[回测校准] ✅ 校准完成！\n"
            f"             初始波动率: 0.1500\n"
            f"             校准波动率: {best_vol:.4f}\n"
            f"             目标波动率: 0.2500（市场真实值）\n"
            f"             校准误差:   {error_pct:.4f}%"
        )

        # ── 发布 ParamUpdateEvent 到主总线 ───────────────
        update = ParamUpdateEvent(
            model_id   = self._model_id,
            params     = {"vol": best_vol, "r": 0.05},
            updated_at = datetime.now(),
        )
        print(
            f"[回测校准] 📤 发布 ParamUpdateEvent → 模型: "
            f"{self._model_id}  参数: vol={best_vol:.4f}"
        )
        self._main_bus.publish(update)

        return best_vol
