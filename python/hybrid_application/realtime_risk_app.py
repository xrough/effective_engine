# ============================================================
# 文件：hybrid_application/realtime_risk_app.py
# 职责：实时风险监控应用（混合模式）— 逻辑与 application/realtime_risk_app.py
#       完全相同，但使用 C++ omm_core 的所有核心对象。
#
# 对比纯 Python 版本的差异：
#   - 事件订阅：bus.subscribe_trade_executed / subscribe_market_data
#   - 事件发布：bus.publish_risk_control / publish_risk_alert
#   - PortfolioAggregate / SimpleRiskPolicy 均来自 omm_core
#
# 事件订阅：TradeExecutedEvent, MarketDataEvent
# 事件发布：RiskControlEvent, RiskAlertEvent
# ============================================================

from __future__ import annotations

import omm_core  # C++ 扩展模块（核心层）


class RealtimeRiskApp:
    """实时风险监控应用（混合模式 — Python 业务逻辑 + C++ 核心对象）。"""

    def __init__(
        self,
        bus:            omm_core.EventBus,
        pricing_engine: omm_core.IPricingEngine,
        options:        list,           # list[omm_core.Option]
        underlying_id:  str,
        account_id:     str,
        risk_policy:    omm_core.IRiskPolicy,
    ) -> None:
        self._bus            = bus
        self._pricing_engine = pricing_engine
        self._underlying_id  = underlying_id
        self._account_id     = account_id
        self._risk_policy    = risk_policy
        # C++ PortfolioAggregate — 与 C++ RealtimeRiskApp 结构完全对称
        self._portfolio      = omm_core.PortfolioAggregate(account_id, options)
        self._last_price     = 150.0

    def register_handlers(self) -> None:
        """向 C++ EventBus 注册成交与行情事件处理器。"""
        self._bus.subscribe_trade_executed(self._on_trade)
        self._bus.subscribe_market_data(self._on_market)

    def _on_trade(self, event: omm_core.TradeExecutedEvent) -> None:
        """成交事件处理器。

        流程：
          1. 更新 C++ PortfolioAggregate 持仓
          2. 按最新市价重新估值
          3. 计算风险指标快照（C++ RiskMetrics）
          4. 评估 C++ SimpleRiskPolicy，获取 RiskControlEvent 列表
          5. 发布风控事件到 C++ EventBus
        """
        # ── 步骤 1：更新持仓 ─────────────────────────────
        self._portfolio.applyTrade(event)

        # ── 步骤 2-3：估值 + 计算指标 ────────────────────
        self._portfolio.markToMarket(self._pricing_engine, self._last_price)
        metrics = self._portfolio.computeMetrics(self._pricing_engine, self._last_price)

        print(
            f"[实时风控|混合|{self._account_id}] 成交后风险快照:"
            f"  已实现盈亏=${metrics.realized_pnl:.2f}"
            f"  未实现盈亏=${metrics.unrealized_pnl:.2f}"
            f"  组合Δ={metrics.delta:.3f}"
        )

        # ── 步骤 4-5：评估并发布风控事件 ─────────────────
        actions = self._risk_policy.evaluate(self._account_id, metrics)
        self._publish_risk_actions(actions)

        # ── 发布日内回撤预警 ──────────────────────────────
        if metrics.intraday_drawdown > 0.0:
            alert = omm_core.RiskAlertEvent()
            alert.account_id   = self._account_id
            alert.metric_name  = "intraday_drawdown"
            alert.value        = metrics.intraday_drawdown
            alert.limit        = 5e5
            self._bus.publish_risk_alert(alert)

    def _on_market(self, event: omm_core.MarketDataEvent) -> None:
        """行情事件处理器。

        流程：
          1. 更新标的价格
          2. 重新估值 + 计算风险指标
          3. 仅在有敞口时评估风险策略并发布
        """
        # ── 步骤 1：更新标的价格 ──────────────────────────
        self._last_price = event.underlying_price

        # ── 步骤 2-3：估值 + 计算 ─────────────────────────
        self._portfolio.markToMarket(self._pricing_engine, self._last_price)
        metrics = self._portfolio.computeMetrics(self._pricing_engine, self._last_price)

        # 仅在有持仓时评估，避免无意义的噪音输出
        if (self._portfolio.get_position(self._underlying_id) != 0
                or metrics.delta != 0.0):
            actions = self._risk_policy.evaluate(self._account_id, metrics)
            self._publish_risk_actions(actions)

    def _publish_risk_actions(self, actions: list) -> None:
        """将策略评估结果广播到 C++ EventBus。"""
        for action in actions:
            self._bus.publish_risk_control(action)
            print(
                f"[实时风控|混合|{self._account_id}] 🚨 发出风控指令: "
                f"{action.action.name}  原因: {action.reason}"
            )
