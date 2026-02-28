# ============================================================
# 文件：pricing_engine.py
# 职责：定义期权定价引擎的抽象接口及两种具体实现。
#
# 模式：策略模式（Strategy Pattern）
#   IPricingEngine 抽象"如何定价"，替换定价模型只需：
#     1. 新建实现了 IPricingEngine 的子类
#     2. 在 main.py 中替换注入的实例
#     3. 其他所有代码零改动
# ============================================================

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import math

from .instrument import Option, OptionType


def _norm_cdf(x: float) -> float:
    """标准正态累积分布函数 N(x)，使用 math.erf 近似计算。
    公式：N(x) = 0.5 * (1 + erf(x / √2))
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class PriceResult:
    """定价结果值对象。
    theo  — 理论公允价值（做市商基于模型估算的中间价）
    delta — 期权价格对标的资产价格的一阶敏感度（dV/dS）
    """
    theo:  float  # 理论价格（美元）
    delta: float  # Delta 值（无量纲）


class IPricingEngine(ABC):
    """定价引擎策略接口（所有定价模型必须实现此接口）。"""

    @abstractmethod
    def price(self, option: Option, underlying_price: float) -> PriceResult:
        """核心定价方法。

        Args:
            option:           待定价的期权合约
            underlying_price: 标的资产当前市场价格

        Returns:
            PriceResult（theo + delta）
        """


class SimplePricingEngine(IPricingEngine):
    """简化定价引擎（MVP 具体策略）。

    定价逻辑（内在价值，无时间价值）：
      看涨 theo  = max(0, S - K)
      看跌 theo  = max(0, K - S)
      Delta      = +0.5（看涨） / -0.5（看跌）（固定桩值，真实应为 N(d₁)）

    注意：此实现有意简化，目的是展示策略模式的接口结构。
    """

    def price(self, option: Option, underlying_price: float) -> PriceResult:
        if option.option_type == OptionType.Call:
            # 看涨期权内在价值：max(0, S - K)
            theo  = max(0.0, underlying_price - option.strike)
            delta = +0.5  # 固定 Delta 桩值
        else:
            # 看跌期权内在价值：max(0, K - S)
            theo  = max(0.0, option.strike - underlying_price)
            delta = -0.5  # 固定 Delta 桩值
        return PriceResult(theo=theo, delta=delta)


class BlackScholesPricingEngine(IPricingEngine):
    """标准 Black-Scholes 定价引擎。

    实现欧式香草期权的完整 Black-Scholes 公式：
      d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
      d₂ = d₁ - σ·√T
      Call = S·N(d₁) - K·e^(-rT)·N(d₂)
      Put  = K·e^(-rT)·N(-d₂) - S·N(-d₁)
      Call Delta = N(d₁),  Put Delta = N(d₁) - 1

    可调参数：
      vol — 隐含波动率（年化，如 0.20 = 20%），用于回测校准
      r   — 无风险利率（年化，默认 0.05 = 5%）
    """

    def __init__(self, vol: float = 0.20, r: float = 0.05) -> None:
        self._vol = vol  # 年化隐含波动率
        self._r   = r    # 年化无风险利率

    def set_vol(self, vol: float) -> None:
        """运行时更新波动率（由 ParameterStore 注入校准结果）。"""
        self._vol = vol

    def get_vol(self) -> float:
        """查询当前波动率。"""
        return self._vol

    def price(self, option: Option, underlying_price: float) -> PriceResult:
        S     = underlying_price   # 标的资产当前价格
        K     = option.strike      # 行权价
        sigma = self._vol          # 年化波动率

        # 计算到期年限 T（从当前时刻到期权到期日）
        now = datetime.now()
        seconds_to_expiry = (option.expiry - now).total_seconds()
        T = max(seconds_to_expiry / (365.0 * 86400.0), 1e-6)  # 转换为年，防止除零

        # 计算 d₁ 和 d₂（Black-Scholes 核心公式）
        d1 = (math.log(S / K) + (self._r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option.option_type == OptionType.Call:
            # 看涨期权定价：Call = S·N(d₁) - K·e^(-rT)·N(d₂)
            theo  = S * _norm_cdf(d1) - K * math.exp(-self._r * T) * _norm_cdf(d2)
            delta = _norm_cdf(d1)          # 看涨 Delta = N(d₁)，范围 [0, 1]
        else:
            # 看跌期权定价：Put = K·e^(-rT)·N(-d₂) - S·N(-d₁)
            theo  = K * math.exp(-self._r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
            delta = _norm_cdf(d1) - 1.0    # 看跌 Delta = N(d₁) - 1，范围 [-1, 0]

        theo = max(0.0, theo)  # 期权理论价格不得为负
        return PriceResult(theo=theo, delta=delta)
