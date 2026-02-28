# ============================================================
# 文件：delta_hedger.py
# 职责：Delta 对冲器 — 订阅成交事件，监控组合 Delta 敞口，
#       当 |Δ| 超过阈值时提交市价对冲单，并直接记账对冲成交。
#
# 模式：策略模式（Strategy Pattern）+ 命令模式（Command Pattern）
#   - IDeltaHedgeStrategy 可替换对冲逻辑（Gamma 对冲、Vega 对冲等）
#   - OrderSubmittedEvent 将对冲意图封装为命令对象
#
# 注意：对冲成交直接绕过事件总线记入 PositionManager，
#       以避免触发新的 TradeExecutedEvent 导致无限递归。
#
# 事件订阅：TradeExecutedEvent（订阅后同步更新持仓与 Delta）
# 事件发布：OrderSubmittedEvent（当 |Δ| 超阈值时发出对冲单）
# ============================================================

from __future__ import annotations
import math

from events.event_bus import EventBus
from events.events import (
    TradeExecutedEvent, OrderSubmittedEvent,
    MarketDataEvent, Side, OrderType,
)
from domain.instrument import Option
from domain.pricing_engine import IPricingEngine
from domain.position_manager import PositionManager


class DeltaHedger:
    """基于 Delta 阈值的自动对冲器。"""

    def __init__(
        self,
        bus:              EventBus,
        position_manager: PositionManager,
        pricing_engine:   IPricingEngine,
        options:          list[Option],
        underlying_id:    str,
        delta_threshold:  float = 0.5,  # |Δ| 超过此值时触发对冲
    ) -> None:
        self._bus              = bus
        self._position_manager = position_manager
        self._pricing_engine   = pricing_engine
        self._options          = options
        self._underlying_id    = underlying_id
        self._delta_threshold  = delta_threshold
        self._last_price       = 150.0  # 最近一次标的价格（默认开盘估计值）

    def register_handlers(self) -> None:
        """向 EventBus 注册成交事件处理器。"""
        self._bus.subscribe(TradeExecutedEvent, self._on_trade_executed)

    def update_market_price(self, price: float) -> None:
        """更新标的资产最新价格（由行情事件驱动）。"""
        self._last_price = price

    def _on_trade_executed(self, event: TradeExecutedEvent) -> None:
        """成交事件处理器。

        流程：
          1. 更新持仓（PositionManager）
          2. 计算各合约 Delta
          3. 计算组合 Delta：Δ_portfolio = Σ(Δ_i × position_i)
          4. 若 |Δ| > 阈值：发出 OrderSubmittedEvent，直接记账对冲成交
        """
        # ── 步骤 1：更新持仓 ─────────────────────────────
        self._position_manager.on_trade_executed(event)

        # ── 步骤 2-3：计算组合 Delta ──────────────────────
        delta_map       = self._compute_delta_map(self._last_price)
        portfolio_delta = self._position_manager.compute_portfolio_delta(delta_map)

        print(
            f"[Delta对冲] 组合 Delta = {portfolio_delta:.3f}"
            f"  (阈值 ±{self._delta_threshold})"
        )

        if abs(portfolio_delta) <= self._delta_threshold:
            print("[Delta对冲] Delta 在阈值内，无需对冲")
            return

        # ── 步骤 4：触发对冲 ──────────────────────────────
        hedge_qty  = round(abs(portfolio_delta))
        hedge_side = Side.Sell if portfolio_delta > 0.0 else Side.Buy

        order = OrderSubmittedEvent(
            instrument_id = self._underlying_id,
            side          = hedge_side,
            quantity      = hedge_qty,
            order_type    = OrderType.Market,
        )

        side_str = "卖出" if hedge_side == Side.Sell else "买入"
        print(
            f"[Delta对冲] *** 触发对冲！发送市价单: "
            f"{side_str} {hedge_qty} 股 {self._underlying_id}"
            f"  (当前 Delta={portfolio_delta:.3f})"
        )

        self._bus.publish(order)

        # ── 对冲成交直接记账（绕过事件总线，防止递归）───────
        fill_side = Side.Sell if hedge_side == Side.Buy else Side.Buy
        hedge_fill = TradeExecutedEvent(
            instrument_id = self._underlying_id,
            side          = fill_side,
            price         = self._last_price,
            quantity      = hedge_qty,
            timestamp     = event.timestamp,
        )
        self._position_manager.on_trade_executed(hedge_fill)
        print("[Delta对冲] 对冲成交已直接记账（绕过事件总线，防止递归）")

    def _compute_delta_map(self, current_price: float) -> dict[str, float]:
        """计算所有合约的 Delta 映射。

        标的资产 Delta 恒为 1.0；
        各期权 Delta 由 pricing_engine 计算。
        """
        deltas: dict[str, float] = {self._underlying_id: 1.0}
        for opt in self._options:
            result             = self._pricing_engine.price(opt, current_price)
            deltas[opt.id]     = result.delta
        return deltas
