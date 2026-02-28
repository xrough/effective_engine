# ============================================================
# 文件：probabilistic_taker.py
# 职责：概率成交模拟器 — 订阅报价事件，以固定概率模拟客户成交行为。
#
# 设计：
#   - 使用固定随机种子（42）确保仿真结果可重现
#   - 30% 概率触发成交；成交方向随机（Hit Bid / Lift Ask 各 50%）
#
# 事件订阅：QuoteGeneratedEvent
# 事件发布：TradeExecutedEvent
# ============================================================

from __future__ import annotations
import random

from events.event_bus import EventBus
from events.events import QuoteGeneratedEvent, TradeExecutedEvent, Side


class ProbabilisticTaker:
    """模拟市场参与者（随机概率成交）。"""

    def __init__(
        self,
        bus:              EventBus,
        trade_probability: float = 0.10,  # 默认 10% 成交概率
        seed:             int   = 42,     # 固定种子，确保可重现
    ) -> None:
        self._bus              = bus
        self._trade_probability = trade_probability
        self._rng              = random.Random(seed)  # 隔离随机状态

    def register_handlers(self) -> None:
        """向 EventBus 注册报价事件处理器。"""
        self._bus.subscribe(QuoteGeneratedEvent, self._on_quote_generated)

    def _on_quote_generated(self, event: QuoteGeneratedEvent) -> None:
        """报价事件处理器：以概率 trade_probability 模拟客户成交。

        流程：
          1. 掷随机数，若 roll < trade_probability：触发成交
          2. 随机选择方向：Hit Bid（客户卖出）或 Lift Ask（客户买入）
          3. 发布 TradeExecutedEvent
        """
        roll = self._rng.random()
        if roll >= self._trade_probability:
            return  # 未触发成交

        # 随机选择成交方向
        hit_bid = self._rng.randint(0, 1) == 0
        side    = Side.Sell if hit_bid else Side.Buy
        price   = event.bid_price if hit_bid else event.ask_price

        direction = "Hit Bid" if hit_bid else "Lift Ask"
        print(
            f"[成交模拟] {event.instrument_id}  {direction} @ ${price:.4f}"
            f"  (掷骰结果: {roll:.2f} < {self._trade_probability})"
        )

        self._bus.publish(TradeExecutedEvent(
            instrument_id = event.instrument_id,
            side          = side,
            price         = price,
            quantity      = 1,
            timestamp     = event.timestamp,
        ))
