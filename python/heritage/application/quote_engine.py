# ============================================================
# 文件：quote_engine.py
# 职责：报价引擎 — 订阅行情事件，为每个期权生成 bid/ask 报价，
#       并将报价通过 EventBus 广播。
#
# 模式：策略模式（Strategy Pattern）
#   IQuoteStrategy 抽象"如何报价"，QuoteEngine 实现固定价差策略。
#   替换报价逻辑（如库存倾斜报价、波动率曲面报价）只需新建子类。
#
# 事件订阅：MarketDataEvent
# 事件发布：QuoteGeneratedEvent（每个 tick × 每个期权）
# ============================================================

from __future__ import annotations
from abc import ABC, abstractmethod

from events.event_bus import EventBus
from events.events import MarketDataEvent, QuoteGeneratedEvent
from domain.instrument import Option
from domain.pricing_engine import IPricingEngine


class IQuoteStrategy(ABC):
    """报价策略接口（抽象"如何生成 bid/ask"）。"""

    @abstractmethod
    def on_market_data(self, event: MarketDataEvent) -> None:
        """接收行情事件并生成报价。"""


class QuoteEngine(IQuoteStrategy):
    """固定价差报价引擎（MVP 具体策略）。

    报价逻辑：
      theo = pricing_engine.price(option, S).theo
      bid  = max(0, theo - half_spread)
      ask  = max(0, theo + half_spread)
    """

    def __init__(
        self,
        bus:            EventBus,
        pricing_engine: IPricingEngine,
        options:        list[Option],
        half_spread:    float = 0.05,  # 半价差（美元）
    ) -> None:
        self._bus            = bus             # 事件总线（用于发布报价）
        self._pricing_engine = pricing_engine  # 定价引擎（策略模式注入）
        self._options        = options         # 需要报价的期权合约列表
        self._half_spread    = half_spread     # 半价差：bid = theo - hs, ask = theo + hs

    def register_handlers(self) -> None:
        """向 EventBus 注册行情事件处理器。"""
        self._bus.subscribe(MarketDataEvent, self.on_market_data)

    def on_market_data(self, event: MarketDataEvent) -> None:
        """行情事件处理器 — 为每个期权生成并发布报价。

        流程：
          1. 调用 pricing_engine.price() 获取理论价和 Delta
          2. 计算 bid = theo - half_spread，ask = theo + half_spread
          3. 发布 QuoteGeneratedEvent
        """
        for option in self._options:
            result = self._pricing_engine.price(option, event.underlying_price)

            bid = max(0.0, result.theo - self._half_spread)  # bid 不得为负
            ask = max(0.0, result.theo + self._half_spread)  # ask 不得为负

            print(
                f"[报价引擎] {option.id}"
                f"  Bid=${bid:.4f}  Ask=${ask:.4f}"
                f"  (theo=${result.theo:.4f}  Δ={result.delta:.4f})"
            )

            # 发布报价事件（由 ProbabilisticTaker 订阅）
            self._bus.publish(QuoteGeneratedEvent(
                instrument_id = option.id,
                bid_price     = bid,
                ask_price     = ask,
                timestamp     = event.timestamp,
            ))
