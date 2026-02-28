# ============================================================
# 文件：portfolio_aggregate.py
# 职责：维护单账户的完整投资组合状态，并计算风险指标快照。
#
# 领域聚合根（Domain Aggregate Root）：
#   封装账户内所有持仓、盈亏状态，
#   保证内部数据一致性（持仓变更必须通过 apply_trade() 进行）。
#
# 持仓更新策略：加权平均成本法（Average Cost Basis）
#   - 增仓：加权平均更新成本
#   - 减仓：按平均成本计算已实现盈亏
# ============================================================

from __future__ import annotations
import math
from events.events import TradeExecutedEvent, Side
from .instrument import Option
from .pricing_engine import IPricingEngine
from .risk_metrics import RiskMetrics


class PortfolioAggregate:
    """单账户投资组合聚合根。"""

    def __init__(self, account_id: str, options: list[Option]) -> None:
        self._account_id      = account_id        # 账户标识符
        self._options         = options           # 期权合约列表（用于 Greeks 计算）
        self._positions:  dict[str, int]   = {}   # 净持仓
        self._avg_cost:   dict[str, float] = {}   # 平均成本
        self._realized_pnl    = 0.0               # 累计已实现盈亏
        self._unrealized_pnl  = 0.0               # 当前未实现盈亏
        self._total_pnl_high  = 0.0               # 日内总盈亏高水位

    def apply_trade(self, event: TradeExecutedEvent) -> None:
        """应用成交事件，更新净持仓与已实现盈亏。

        客户视角转换：
          客户 Buy  → 做市商持仓减少（做市商卖出）
          客户 Sell → 做市商持仓增加（做市商买入）
        """
        id_      = event.instrument_id
        # 做市商持仓变化方向
        pos_chg  = +event.quantity if event.side == Side.Sell else -event.quantity
        old_pos  = self._positions.get(id_, 0)
        new_pos  = old_pos + pos_chg
        price    = event.price

        if old_pos == 0 or (old_pos > 0 and pos_chg > 0) or (old_pos < 0 and pos_chg < 0):
            # 同向加仓：加权平均更新成本
            old_notional = self._avg_cost.get(id_, price) * abs(old_pos)
            new_notional = price * abs(pos_chg)
            total_qty    = abs(old_pos) + abs(pos_chg)
            self._avg_cost[id_] = (old_notional + new_notional) / total_qty if total_qty else price
        else:
            # 反向减仓：计算已实现盈亏
            close_qty = min(abs(old_pos), abs(pos_chg))
            cost      = self._avg_cost.get(id_, price)
            pnl = (price - cost) * close_qty if old_pos > 0 else (cost - price) * close_qty
            self._realized_pnl += pnl
            if abs(pos_chg) > abs(old_pos):
                # 反手建仓，更新成本为新价格
                self._avg_cost[id_] = price

        self._positions[id_] = new_pos
        print(
            f"[持仓聚合|{self._account_id}] {id_} 持仓更新: "
            f"{old_pos:+d} → {new_pos:+d}  已实现盈亏累计: ${self._realized_pnl:.2f}"
        )

    def mark_to_market(self, engine: IPricingEngine, underlying_price: float) -> None:
        """按当前市价重新估值，更新未实现盈亏与高水位。"""
        unrealized = 0.0
        for opt in self._options:
            pos = self.get_position(opt.id)
            if pos == 0:
                continue
            result = engine.price(opt, underlying_price)
            cost   = self._avg_cost.get(opt.id, result.theo)
            # 未实现盈亏 = (当前价 - 成本价) × 净持仓
            unrealized += (result.theo - cost) * float(pos)

        self._unrealized_pnl = unrealized
        # 更新日内总盈亏高水位（用于回撤计算）
        total_pnl            = self._realized_pnl + self._unrealized_pnl
        self._total_pnl_high = max(self._total_pnl_high, total_pnl)

    def compute_metrics(
        self, engine: IPricingEngine, underlying_price: float
    ) -> RiskMetrics:
        """生成当前时刻的风险指标快照。"""
        m = RiskMetrics(
            realized_pnl   = self._realized_pnl,
            unrealized_pnl = self._unrealized_pnl,
        )

        # 计算组合 Delta：Σ(delta_i × position_i)
        for opt in self._options:
            pos = self.get_position(opt.id)
            if pos == 0:
                continue
            result  = engine.price(opt, underlying_price)
            m.delta += result.delta * float(pos)

        # 日内最大回撤 = 高水位 - 当前总盈亏
        total_pnl          = self._realized_pnl + self._unrealized_pnl
        m.intraday_drawdown = self._total_pnl_high - total_pnl

        # VaR 简化估算：1日 VaR ≈ |Delta| × S × σ × √(1/252)（95% 置信度，σ=20% 代理值）
        m.var_1d = abs(m.delta) * underlying_price * 0.20 / math.sqrt(252.0)

        return m

    def get_position(self, instrument_id: str) -> int:
        """查询指定合约净持仓。"""
        return self._positions.get(instrument_id, 0)
