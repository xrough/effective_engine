# ============================================================
# 文件：realtime_risk_app.py
# 职责：实时风险监控应用 — 订阅成交与行情事件，实时评估账户风险，
#       在触发限额时发出风控指令（RiskControlEvent / RiskAlertEvent）。
#
# 对应 Risk_Calibration.md §1（Realtime Risk Application）
#
# 工作流（与 Risk_Calibration.md §1.5 一致）：
#   1. TradeExecutedEvent  → 更新持仓 → 重新估值 → 计算 Greeks → 评估策略 → 发布风控
#   2. MarketDataEvent     → 更新标的价 → 重新估值 → 计算 Greeks → 评估策略 → 发布风控
#
# 事件订阅：TradeExecutedEvent, MarketDataEvent
# 事件发布：RiskControlEvent, RiskAlertEvent
# ============================================================

from __future__ import annotations

from events.event_bus import EventBus
from events.events import (
    TradeExecutedEvent, MarketDataEvent,
    RiskControlEvent, RiskAlertEvent,
)
from domain.instrument import Option
from domain.pricing_engine import IPricingEngine
from domain.portfolio_aggregate import PortfolioAggregate
from domain.risk_policy import IRiskPolicy


class RealtimeRiskApp:
    """实时风险监控应用。"""

    def __init__(
        self,
        bus:            EventBus,
        pricing_engine: IPricingEngine,
        options:        list[Option],
        underlying_id:  str,
        account_id:     str,
        risk_policy:    IRiskPolicy,
    ) -> None:
        self._bus            = bus
        self._pricing_engine = pricing_engine
        self._underlying_id  = underlying_id
        self._account_id     = account_id
        self._risk_policy    = risk_policy
        # PortfolioAggregate 是 RealtimeRiskApp 的私有状态，外部不可访问
        self._portfolio      = PortfolioAggregate(account_id, options)
        self._last_price     = 150.0  # 最近一次标的价格

    def register_handlers(self) -> None:
        """向 EventBus 注册成交与行情事件处理器。"""
        self._bus.subscribe(TradeExecutedEvent, self._on_trade)
        self._bus.subscribe(MarketDataEvent,    self._on_market)

    def _on_trade(self, event: TradeExecutedEvent) -> None:
        """成交事件处理器。

        流程：
          1. 更新持仓（apply_trade）
          2. 按最新市价重新估值（mark_to_market）
          3. 计算风险指标快照（compute_metrics）
          4. 评估风险策略（evaluate）
          5. 发布风控事件（publish_risk_actions）
        """
        # ── 步骤 1：更新持仓 ─────────────────────────────
        self._portfolio.apply_trade(event)

        # ── 步骤 2：重新估值 ──────────────────────────────
        self._portfolio.mark_to_market(self._pricing_engine, self._last_price)

        # ── 步骤 3：计算风险指标 ─────────────────────────
        metrics = self._portfolio.compute_metrics(self._pricing_engine, self._last_price)

        print(
            f"[实时风控|{self._account_id}] 成交后风险快照:"
            f"  已实现盈亏=${metrics.realized_pnl:.2f}"
            f"  未实现盈亏=${metrics.unrealized_pnl:.2f}"
            f"  组合Δ={metrics.delta:.3f}"
        )

        # ── 步骤 4-5：评估并发布风控事件 ─────────────────
        actions = self._risk_policy.evaluate(self._account_id, metrics)
        self._publish_risk_actions(actions)

        # ── 发布日内回撤预警 ──────────────────────────────
        if metrics.intraday_drawdown > 0.0:
            self._bus.publish(RiskAlertEvent(
                account_id  = self._account_id,
                metric_name = "intraday_drawdown",
                value       = metrics.intraday_drawdown,
                limit       = 5e5,  # 日内回撤预警阈值 $500,000
            ))

    def _on_market(self, event: MarketDataEvent) -> None:
        """行情事件处理器。

        流程：
          1. 更新标的价格
          2. 重新估值
          3. 计算风险指标
          4. 评估并发布（仅在有持仓时输出，避免噪音）
        """
        # ── 步骤 1：更新标的价格 ──────────────────────────
        self._last_price = event.underlying_price

        # ── 步骤 2-3：重新估值 + 计算指标 ────────────────
        self._portfolio.mark_to_market(self._pricing_engine, self._last_price)
        metrics = self._portfolio.compute_metrics(self._pricing_engine, self._last_price)

        # ── 步骤 4-5：仅在有 Delta 敞口时评估风险 ─────────
        if (self._portfolio.get_position(self._underlying_id) != 0
                or metrics.delta != 0.0):
            actions = self._risk_policy.evaluate(self._account_id, metrics)
            self._publish_risk_actions(actions)

    def _publish_risk_actions(self, actions: list[RiskControlEvent]) -> None:
        """将策略评估结果广播到 EventBus。"""
        for action in actions:
            self._bus.publish(action)
            action_name = action.action.name  # 枚举名称：BlockOrders / ReduceOnly / CancelOrders
            print(
                f"[实时风控|{self._account_id}] 🚨 发出风控指令: "
                f"{action_name}  原因: {action.reason}"
            )
