# ============================================================
# 文件：position_manager.py
# 职责：持仓管理器 — 跟踪做市商对每个合约的净持仓，
#       并计算整个组合的 Delta 敞口。
# ============================================================

from __future__ import annotations
from events.events import TradeExecutedEvent, Side


class PositionManager:
    """持仓管理器（原始 MVP 持仓追踪，供 DeltaHedger 使用）。"""

    def __init__(self) -> None:
        # 净持仓：instrument_id → 净头寸（正 = 多头，负 = 空头）
        self._positions: dict[str, int] = {}

    def on_trade_executed(self, event: TradeExecutedEvent) -> None:
        """应用成交事件，更新净持仓。

        客户视角转换为做市商视角：
          客户 Buy  → 做市商持仓减少（做市商是卖方，delta_qty = -qty）
          客户 Sell → 做市商持仓增加（做市商是买方，delta_qty = +qty）
        """
        delta_qty = (
            -event.quantity if event.side == Side.Buy else +event.quantity
        )
        self._positions[event.instrument_id] = (
            self._positions.get(event.instrument_id, 0) + delta_qty
        )
        pos = self._positions[event.instrument_id]
        print(f"[PositionManager] 持仓更新: {event.instrument_id} → {pos:+d}")

    def get_position(self, instrument_id: str) -> int:
        """查询指定合约的净持仓。"""
        return self._positions.get(instrument_id, 0)

    def compute_portfolio_delta(self, deltas: dict[str, float]) -> float:
        """计算组合总 Delta：Σ(delta_i × position_i)。

        Args:
            deltas: instrument_id → delta 值的映射
        """
        total = 0.0
        for instr_id, pos in self._positions.items():
            if instr_id in deltas:
                total += deltas[instr_id] * float(pos)
        return total

    def print_positions(self) -> None:
        """打印当前所有持仓汇总（仿真结束时调用）。"""
        if not self._positions:
            print("[PositionManager] 当前无持仓")
            return
        print("[PositionManager] 最终持仓汇总:")
        for instr_id, pos in self._positions.items():
            print(f"  {instr_id:<30} {pos:+d}")
