# ============================================================
# 文件：hybrid_application/delta_hedger.py
# 职责：Delta 对冲器（混合模式）— 逻辑与 application/delta_hedger.py 完全相同，
#       但使用 C++ omm_core 的 EventBus / PositionManager / 事件对象。
#
# 对比纯 Python 版本的差异：
#   - 事件订阅：bus.subscribe_trade_executed(fn)
#   - 事件发布：bus.publish_order_submitted(event)
#   - PositionManager / TradeExecutedEvent 均来自 omm_core
#
# 注意：对冲成交直接调用 position_manager.on_trade_executed()，
#       绕过 EventBus 以防止无限递归（与 C++ 版本相同的处理方式）。
#
# 事件订阅：TradeExecutedEvent（来自 C++ ProbabilisticTaker）
# 事件发布：OrderSubmittedEvent（命令模式）
# ============================================================

from __future__ import annotations

import omm_core  # C++ 扩展模块（核心层）


class DeltaHedger:
    """基于 Delta 阈值的自动对冲器（混合模式）。"""

    def __init__(
        self,
        bus:              omm_core.EventBus,
        position_manager: omm_core.PositionManager,
        pricing_engine:   omm_core.IPricingEngine,
        options:          list,           # list[omm_core.Option]
        underlying_id:    str,
        delta_threshold:  float = 0.5,
    ) -> None:
        self._bus              = bus
        self._position_manager = position_manager
        self._pricing_engine   = pricing_engine
        self._options          = options
        self._underlying_id    = underlying_id
        self._delta_threshold  = delta_threshold
        self._last_price       = 150.0  # 最近一次标的价格（默认开盘估计值）

    def register_handlers(self) -> None:
        """向 C++ EventBus 注册成交事件处理器。"""
        self._bus.subscribe_trade_executed(self._on_trade_executed)

    def update_market_price(self, price: float) -> None:
        """更新标的资产最新价格（由行情事件驱动）。"""
        self._last_price = price

    def _on_trade_executed(self, event: omm_core.TradeExecutedEvent) -> None:
        """成交事件处理器（接收 C++ TradeExecutedEvent）。

        流程：
          1. 通过 C++ PositionManager 更新持仓
          2. 计算各合约 Delta（调用 C++ pricing_engine）
          3. 计算组合 Delta：Δ_portfolio = Σ(Δ_i × position_i)
          4. 若 |Δ| > 阈值：发出 OrderSubmittedEvent，直接记账对冲成交
        """
        # ── 步骤 1：更新持仓 ─────────────────────────────
        self._position_manager.on_trade_executed(event)

        # ── 步骤 2-3：计算组合 Delta ──────────────────────
        delta_map       = self._compute_delta_map(self._last_price)
        portfolio_delta = self._position_manager.compute_portfolio_delta(delta_map)

        print(
            f"[Delta对冲|混合] 组合 Delta = {portfolio_delta:.3f}"
            f"  (阈值 ±{self._delta_threshold})"
        )

        if abs(portfolio_delta) <= self._delta_threshold:
            print("[Delta对冲|混合] Delta 在阈值内，无需对冲")
            return

        # ── 步骤 4：触发对冲 ──────────────────────────────
        hedge_qty  = round(abs(portfolio_delta))
        hedge_side = omm_core.Side.Sell if portfolio_delta > 0.0 else omm_core.Side.Buy

        order = omm_core.OrderSubmittedEvent()
        order.instrument_id = self._underlying_id
        order.side          = hedge_side
        order.quantity      = hedge_qty
        order.order_type    = omm_core.OrderType.Market

        side_str = "卖出" if hedge_side == omm_core.Side.Sell else "买入"
        print(
            f"[Delta对冲|混合] *** 触发对冲！发送市价单: "
            f"{side_str} {hedge_qty} 股 {self._underlying_id}"
            f"  (当前 Delta={portfolio_delta:.3f})"
        )

        self._bus.publish_order_submitted(order)

        # ── 对冲成交直接记账（绕过事件总线，防止递归）───────
        fill_side = omm_core.Side.Sell if hedge_side == omm_core.Side.Buy else omm_core.Side.Buy
        hedge_fill = omm_core.TradeExecutedEvent()
        hedge_fill.instrument_id = self._underlying_id
        hedge_fill.side          = fill_side
        hedge_fill.price         = self._last_price
        hedge_fill.quantity      = hedge_qty
        hedge_fill.timestamp     = event.timestamp
        self._position_manager.on_trade_executed(hedge_fill)
        print("[Delta对冲|混合] 对冲成交已直接记账（绕过事件总线，防止递归）")

    def _compute_delta_map(self, current_price: float) -> dict:
        """计算所有合约的 Delta 映射（传入 C++ PositionManager）。

        标的资产 Delta 恒为 1.0；各期权 Delta 由 C++ pricing_engine 计算。
        """
        deltas: dict = {self._underlying_id: 1.0}
        for opt in self._options:
            result        = self._pricing_engine.price(opt, current_price)
            deltas[opt.id] = result.delta
        return deltas
