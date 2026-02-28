# ============================================================
# 文件：instrument.py
# 职责：定义金融工具的领域模型：标的资产与期权合约。
#
# 类层次结构：
#   Instrument（抽象基类）
#     ├── Underlying（标的资产，如 AAPL 股票）
#     └── Option（欧式香草期权，含看涨/看跌）
#
# 工厂模式（Factory Pattern）：
#   InstrumentFactory 集中管理合约对象的创建，
#   调用方无需知晓 ID 生成规则或构造细节。
# ============================================================

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto


class OptionType(Enum):
    """期权类型枚举。"""
    Call = auto()  # 看涨期权
    Put  = auto()  # 看跌期权


class Instrument(ABC):
    """金融工具抽象基类（所有可交易合约的公共接口）。"""

    def __init__(self, id: str) -> None:
        self._id = id  # 合约唯一标识符

    @property
    def id(self) -> str:
        """合约标识符（只读）。"""
        return self._id

    @abstractmethod
    def type_name(self) -> str:
        """合约类型名称（供日志/调试使用）。"""


class Underlying(Instrument):
    """标的资产（如 AAPL 股票）。

    Delta 恒为 1.0：标的资产价格每变动 $1，自身价值变动 $1。
    """

    def type_name(self) -> str:
        return "Underlying"


class Option(Instrument):
    """欧式香草期权合约。

    包含定价所需的全部合约要素：
      underlying_id — 标的合约 ID
      strike        — 行权价（美元）
      expiry        — 到期日（datetime）
      option_type   — Call / Put
    """

    def __init__(
        self,
        id:           str,
        underlying_id: str,
        strike:       float,
        expiry:       datetime,
        option_type:  OptionType,
    ) -> None:
        super().__init__(id)
        self._underlying_id = underlying_id  # 标的合约 ID
        self._strike        = strike         # 行权价
        self._expiry        = expiry         # 到期日
        self._option_type   = option_type    # 看涨 / 看跌

    @property
    def underlying_id(self) -> str:
        return self._underlying_id

    @property
    def strike(self) -> float:
        return self._strike

    @property
    def expiry(self) -> datetime:
        return self._expiry

    @property
    def option_type(self) -> OptionType:
        return self._option_type

    def type_name(self) -> str:
        return "Call" if self._option_type == OptionType.Call else "Put"


class InstrumentFactory:
    """工厂模式：集中创建金融工具对象。"""

    @staticmethod
    def make_underlying(id: str) -> Underlying:
        """创建标的资产。"""
        return Underlying(id)

    @staticmethod
    def make_call(underlying_id: str, strike: float, expiry: datetime) -> Option:
        """创建看涨期权。"""
        id = InstrumentFactory._make_option_id(underlying_id, strike, OptionType.Call, expiry)
        return Option(id, underlying_id, strike, expiry, OptionType.Call)

    @staticmethod
    def make_put(underlying_id: str, strike: float, expiry: datetime) -> Option:
        """创建看跌期权。"""
        id = InstrumentFactory._make_option_id(underlying_id, strike, OptionType.Put, expiry)
        return Option(id, underlying_id, strike, expiry, OptionType.Put)

    @staticmethod
    def _make_option_id(
        underlying_id: str,
        strike: float,
        option_type: OptionType,
        expiry: datetime,
    ) -> str:
        """生成期权合约 ID，格式：{标的}_{行权价}_{C/P}_{到期日YYYYMMDD}。"""
        type_char = "C" if option_type == OptionType.Call else "P"
        return f"{underlying_id}_{int(strike)}_{type_char}_{expiry.strftime('%Y%m%d')}"
