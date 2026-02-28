# ============================================================
# 文件：events.py
# 职责：定义系统中所有领域事件（Domain Events）的数据结构。
#
# 设计原则：
#   - 事件是"已发生的事实"，使用 @dataclass 表示，无行为方法。
#   - 事件是值对象（Value Object），可安全复制传递。
#   - 所有事件均为纯数据载体，不持有任何 I/O 或业务逻辑。
#
# 典型流程：
#   MarketDataEvent → QuoteGeneratedEvent → TradeExecutedEvent
#     → OrderSubmittedEvent / RiskControlEvent / ParamUpdateEvent
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto


# ── 枚举类型 ─────────────────────────────────────────────────

class Side(Enum):
    """成交方向（客户视角）。"""
    Buy  = auto()  # 客户买入（lift ask）
    Sell = auto()  # 客户卖出（hit bid）


class OrderType(Enum):
    """订单类型。"""
    Market = auto()  # 市价单：立即以市场最优价成交
    Limit  = auto()  # 限价单：仅在指定价格或更优价格成交


class RiskAction(Enum):
    """风控动作枚举。"""
    BlockOrders  = auto()  # 冻结账户下单权限
    CancelOrders = auto()  # 批量撤单
    ReduceOnly   = auto()  # 限制为减仓模式


# ── 领域事件（Domain Events）────────────────────────────────

@dataclass
class MarketDataEvent:
    """行情数据事件。
    发布者：MarketDataAdapter  订阅者：QuoteEngine, DeltaHedger, RealtimeRiskApp
    """
    timestamp:         datetime  # 行情时间戳
    underlying_price:  float     # 标的资产当前价格（如 AAPL 股价）


@dataclass
class QuoteGeneratedEvent:
    """报价生成事件。
    发布者：QuoteEngine  订阅者：ProbabilisticTaker
    """
    instrument_id: str       # 期权合约标识符
    bid_price:     float     # 做市商买价（客户可卖出的价格）
    ask_price:     float     # 做市商卖价（客户可买入的价格）
    timestamp:     datetime  # 报价时间戳


@dataclass
class TradeExecutedEvent:
    """成交事件。
    发布者：ProbabilisticTaker  订阅者：DeltaHedger, RealtimeRiskApp
    注意：Side 表示客户方向，非做市商方向。
    """
    instrument_id: str       # 成交合约标识符
    side:          Side      # 客户方向（Buy / Sell）
    price:         float     # 成交价格
    quantity:      int       # 成交数量
    timestamp:     datetime  # 成交时间戳


@dataclass
class OrderSubmittedEvent:
    """订单提交事件（Command 模式）。
    发布者：DeltaHedger  订阅者：OrderRouter 存根
    """
    instrument_id: str        # 订单标的合约
    side:          Side       # 订单方向
    quantity:      int        # 订单数量
    order_type:    OrderType  # 订单类型


@dataclass
class RiskControlEvent:
    """风控指令事件。
    发布者：RealtimeRiskApp  订阅者：OrderRouter, 日志处理器
    """
    account_id: str         # 受影响账户 ID
    action:     RiskAction  # 风控动作
    reason:     str         # 触发原因描述


@dataclass
class RiskAlertEvent:
    """风险预警事件。
    发布者：RealtimeRiskApp  订阅者：日志处理器
    """
    account_id:   str    # 告警账户 ID
    metric_name:  str    # 触发告警的指标名称
    value:        float  # 当前指标值
    limit:        float  # 对应风险限额


@dataclass
class ParamUpdateEvent:
    """模型参数更新事件。
    发布者：BacktestCalibrationApp  订阅者：ParameterStore
    校准引擎完成优化后，通过此事件将新参数广播到系统。
    """
    model_id:   str              # 模型标识符（如 "bs_model"）
    params:     dict[str, float] # 参数键值对（如 {"vol": 0.22}）
    updated_at: datetime         # 参数更新时间戳
