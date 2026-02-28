# ============================================================
# 文件：hybrid_application/quote_engine.py
# 职责：报价引擎（混合模式）— 逻辑与 application/quote_engine.py 完全相同，
#       但直接使用 C++ omm_core 模块的 EventBus / 事件 / 期权合约对象。
#
# 对比纯 Python 版本的差异：
#   - 事件订阅：bus.subscribe_market_data(fn)     （非 bus.subscribe(MarketDataEvent, fn)）
#   - 事件发布：bus.publish_quote_generated(event) （非 bus.publish(event)）
#   - 事件构造：omm_core.QuoteGeneratedEvent()     （非 Python dataclass）
#
# 事件订阅：MarketDataEvent（来自 C++ MarketDataAdapter）
# 事件发布：QuoteGeneratedEvent（由 C++ ProbabilisticTaker 消费）
# ============================================================

from __future__ import annotations

import omm_core  # C++ 扩展模块（核心层）


class QuoteEngine:
    """固定价差报价引擎（混合模式 — Python 逻辑 + C++ 核心对象）。

    报价逻辑：
      theo = pricing_engine.price(option, S).theo
      bid  = max(0, theo - half_spread)
      ask  = max(0, theo + half_spread)
    """

    def __init__(
        self,
        bus:            omm_core.EventBus,
        pricing_engine: omm_core.IPricingEngine,  # 可为 SimplePricingEngine / BlackScholes
        options:        list,                      # list[omm_core.Option]
        half_spread:    float = 0.05,              # 单边价差（美元）
    ) -> None:
        self._bus            = bus
        self._pricing_engine = pricing_engine
        self._options        = options
        self._half_spread    = half_spread

    def register_handlers(self) -> None:
        """向 C++ EventBus 注册行情事件处理器。"""
        self._bus.subscribe_market_data(self._on_market_data)

    def _on_market_data(self, event: omm_core.MarketDataEvent) -> None:
        """行情事件处理器 — 为每个期权生成并发布报价（C++ 事件对象）。

        流程：
          1. 调用 C++ pricing_engine.price() 获取理论价和 Delta
          2. 计算 bid / ask（双边固定价差）
          3. 构造 omm_core.QuoteGeneratedEvent 并发布到 C++ EventBus
        """
        for option in self._options:
            result = self._pricing_engine.price(option, event.underlying_price)

            bid = max(0.0, result.theo - self._half_spread)
            ask = max(0.0, result.theo + self._half_spread)

            print(
                f"[报价引擎|混合] {option.id}"
                f"  Bid=${bid:.4f}  Ask=${ask:.4f}"
                f"  (theo=${result.theo:.4f}  Δ={result.delta:.4f})"
            )

            # 构造 C++ 事件对象并发布到 C++ EventBus
            q = omm_core.QuoteGeneratedEvent()
            q.instrument_id = option.id
            q.bid_price     = bid
            q.ask_price     = ask
            q.timestamp     = event.timestamp  # datetime.datetime（chrono 转换）
            self._bus.publish_quote_generated(q)
