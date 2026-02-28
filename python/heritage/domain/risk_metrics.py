# ============================================================
# 文件：risk_metrics.py
# 职责：定义单账户的风险指标快照（纯数据值对象）。
#
# RiskMetrics 由 PortfolioAggregate.compute_metrics() 生成，
# 传递给 IRiskPolicy.evaluate() 进行风险策略评估。
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RiskMetrics:
    """单账户风险指标快照（Value Object，生成后不可变）。"""

    # ── 盈亏指标 ──────────────────────────────────────────
    realized_pnl:      float = 0.0  # 已实现盈亏（美元），平仓后计入
    unrealized_pnl:    float = 0.0  # 未实现盈亏（美元），按当前市价估值

    # ── 希腊字母（Greeks）────────────────────────────────
    delta:             float = 0.0  # Δ — 组合价值对标的价格的一阶导数
    gamma:             float = 0.0  # Γ — Delta 对标的价格的变化率
    vega:              float = 0.0  # ν — 组合价值对隐含波动率的敏感度
    theta:             float = 0.0  # Θ — 时间衰减：每日组合价值减少量

    # ── 下行风险指标 ─────────────────────────────────────
    var_1d:            float = 0.0  # 1日 VaR（在险价值），置信度 95%
    intraday_drawdown: float = 0.0  # 日内最大回撤（从日内高水位计算）
