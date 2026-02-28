# ============================================================
# 文件：risk_policy.py
# 职责：定义风险策略接口（IRiskPolicy）及默认实现（SimpleRiskPolicy）。
#
# 模式：策略模式（Strategy Pattern）
#   IRiskPolicy 抽象"如何根据风险指标做出决策"，
#   SimpleRiskPolicy 实现基本限额管控规则（来自 Risk_Calibration.md §1.8）。
# ============================================================

from __future__ import annotations
from abc import ABC, abstractmethod
import math

from events.events import RiskControlEvent, RiskAction
from .risk_metrics import RiskMetrics


class IRiskPolicy(ABC):
    """风险策略接口。

    输入：当前账户 ID + RiskMetrics 快照
    输出：需要执行的风控事件列表（可为空）
    """

    @abstractmethod
    def evaluate(
        self, account_id: str, metrics: RiskMetrics
    ) -> list[RiskControlEvent]:
        """评估风险指标，返回需要触发的风控事件列表。"""


class SimpleRiskPolicy(IRiskPolicy):
    """简单限额风险策略（MVP 实现）。

    规则（来自 Risk_Calibration.md §1.8）：
      1. realized_pnl < -loss_limit    → BlockOrders（止损冻结）
      2. |delta| > delta_limit         → ReduceOnly（Delta 超限）
      3. intraday_drawdown > drawdown_limit → 日志预警（不发出风控指令）
    """

    def __init__(
        self,
        loss_limit:     float = 1e6,
        delta_limit:    float = 10000.0,
        drawdown_limit: float = 5e5,
    ) -> None:
        self._loss_limit     = loss_limit      # 触发 BlockOrders 的亏损阈值
        self._delta_limit    = delta_limit     # 触发 ReduceOnly 的 Delta 绝对值上限
        self._drawdown_limit = drawdown_limit  # 触发风险预警的日内回撤上限

    def evaluate(
        self, account_id: str, metrics: RiskMetrics
    ) -> list[RiskControlEvent]:
        actions: list[RiskControlEvent] = []

        # ── 规则 1：止损触发 ─────────────────────────────
        if metrics.realized_pnl < -self._loss_limit:
            actions.append(RiskControlEvent(
                account_id = account_id,
                action     = RiskAction.BlockOrders,
                reason     = (
                    f"已实现亏损 ${int(metrics.realized_pnl)} "
                    f"超出止损限额 ${int(self._loss_limit)}"
                ),
            ))
            print(
                f"[风控策略|{account_id}] ⛔ 触发止损！"
                f" 已实现盈亏: ${metrics.realized_pnl:.2f}"
                f"  限额: -${self._loss_limit:.0f}  → BlockOrders"
            )

        # ── 规则 2：Delta 超限 ───────────────────────────
        if abs(metrics.delta) > self._delta_limit:
            actions.append(RiskControlEvent(
                account_id = account_id,
                action     = RiskAction.ReduceOnly,
                reason     = (
                    f"组合 Delta {metrics.delta:.1f} "
                    f"超出限额 ±{int(self._delta_limit)}"
                ),
            ))
            print(
                f"[风控策略|{account_id}] ⚠ Delta 超限！"
                f" Delta: {metrics.delta:.1f}"
                f"  限额: ±{self._delta_limit:.0f}  → ReduceOnly"
            )

        # ── 规则 3：日内回撤预警 ─────────────────────────
        if metrics.intraday_drawdown > self._drawdown_limit:
            print(
                f"[风控预警|{account_id}] 📉 日内回撤预警！"
                f" 回撤: ${metrics.intraday_drawdown:.2f}"
                f"  阈值: ${self._drawdown_limit:.0f}"
            )

        return actions
