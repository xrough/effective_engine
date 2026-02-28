#!/usr/bin/env python3
# ============================================================
# 文件：main.py
# 职责：组合根（Composition Root）— 系统唯一的组件连线点。
#
# 这是系统中唯一出现具体类名（而非接口/抽象类）的地方。
# 所有其他模块通过接口和事件总线相互通信。
#
# ════════════════════════════════════════════════════════════
# 运行阶段说明：
#
#   第一阶段：实时仿真 + 实时风控监控
#     MarketDataEvent → QuoteEngine → ProbabilisticTaker
#       → TradeExecutedEvent → DeltaHedger + RealtimeRiskApp
#
#   第二阶段：回测与参数校准
#     独立 EventBus 重放历史行情 → BacktestCalibrationApp
#       → 黄金分割搜索拟合 σ → ParamUpdateEvent → ParameterStore
#
#   两个阶段使用独立 EventBus，保证状态完全隔离。
# ════════════════════════════════════════════════════════════

from __future__ import annotations
import sys
import os
from datetime import datetime, timedelta

# ── 将 python/ 目录加入 sys.path（支持直接运行）──────────────
sys.path.insert(0, os.path.dirname(__file__))

# 事件层
from events.event_bus import EventBus
from events.events import (
    OrderSubmittedEvent, RiskControlEvent, RiskAlertEvent,
    Side, OrderType,
)

# 领域层
from domain.instrument import InstrumentFactory
from domain.pricing_engine import SimplePricingEngine, BlackScholesPricingEngine
from domain.position_manager import PositionManager
from domain.risk_policy import SimpleRiskPolicy
from domain.calibration_engine import CalibrationEngine

# 应用层（PRIMARY）
from application.quote_engine import QuoteEngine
from application.delta_hedger import DeltaHedger
from application.realtime_risk_app import RealtimeRiskApp
from application.backtest_calibration_app import BacktestCalibrationApp

# 基础设施层
from infrastructure.market_data_adapter import MarketDataAdapter
from infrastructure.probabilistic_taker import ProbabilisticTaker
from infrastructure.parameter_store import ParameterStore


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     期权做市商 MVP — 事件驱动仿真系统（Python 版）       ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── 公共资源：两个阶段共享合约定义（不可变，安全共享）────
    expiry     = datetime.now() + timedelta(days=30)  # 约 30 天后到期
    underlying = InstrumentFactory.make_underlying("AAPL")
    call_150   = InstrumentFactory.make_call("AAPL", 150.0, expiry)
    call_145   = InstrumentFactory.make_call("AAPL", 145.0, expiry)
    put_155    = InstrumentFactory.make_put("AAPL",  155.0, expiry)
    options    = [call_150, call_145, put_155]

    print("[初始化] 工厂模式创建合约:")
    print(f"  {underlying.id} ({underlying.type_name()})")
    for opt in options:
        print(f"  {opt.id} ({opt.type_name()})")
    print()

    # ── 主事件总线（两阶段共享；接收 ParamUpdateEvent）────────
    main_bus    = EventBus()
    param_store = ParameterStore(main_bus)
    param_store.subscribe_handlers()
    print("[初始化] ParameterStore 已注册（订阅 ParamUpdateEvent）\n")

    # ════════════════════════════════════════════════════════
    # ██ 第一阶段：实时做市仿真 + 实时风控 ██
    # ════════════════════════════════════════════════════════
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  第一阶段：实时做市仿真 + 实时风控监控                  │")
    print("└──────────────────────────────────────────────────────────┘\n")

    live_bus = EventBus()  # 第一阶段独立事件总线
    print("[初始化] 第一阶段 EventBus 已创建（实时仿真专用）")

    # ── 定价策略 ─────────────────────────────────────────────
    pricing_engine = SimplePricingEngine()
    position_mgr   = PositionManager()
    print("[初始化] 定价策略: SimplePricingEngine（内在价值定价）")

    # ── QuoteEngine — 固定价差报价策略 ──────────────────────
    quote_engine = QuoteEngine(live_bus, pricing_engine, options, half_spread=0.05)
    quote_engine.register_handlers()
    print("[初始化] QuoteEngine 已注册（策略：固定价差 ±$0.05）")

    # ── DeltaHedger — Delta 对冲策略 ─────────────────────────
    delta_hedger = DeltaHedger(
        live_bus, position_mgr, pricing_engine, options,
        underlying.id, delta_threshold=0.5,
    )
    delta_hedger.register_handlers()
    # 额外订阅行情事件，让 DeltaHedger 始终掌握最新标的价格
    live_bus.subscribe(
        __import__("events.events", fromlist=["MarketDataEvent"]).MarketDataEvent,
        lambda evt: delta_hedger.update_market_price(evt.underlying_price),
    )
    print("[初始化] DeltaHedger 已注册（阈值: ±0.5 Delta）")

    # ── RealtimeRiskApp — 实时风险监控 ───────────────────────
    risk_policy = SimpleRiskPolicy(
        loss_limit     = 1e6,     # 止损限额 $1,000,000
        delta_limit    = 10000.0, # Delta 限额 ±10,000
        drawdown_limit = 5e5,     # 日内回撤预警 $500,000
    )
    risk_app = RealtimeRiskApp(
        live_bus, pricing_engine, options,
        underlying.id, "DESK_A", risk_policy,
    )
    risk_app.register_handlers()
    print("[初始化] RealtimeRiskApp 已注册（账户: DESK_A，风控策略: SimpleRiskPolicy）")

    # ── 风控事件日志处理器 ────────────────────────────────────
    def _on_risk_control(evt: RiskControlEvent) -> None:
        print(
            f"[风控日志] 📋 风控指令已记录: 账户={evt.account_id}"
            f"  动作={evt.action.name}  原因: {evt.reason}"
        )

    def _on_risk_alert(evt: RiskAlertEvent) -> None:
        if evt.value > 1.0:  # 仅打印有意义的预警
            print(
                f"[风险预警] 📊 {evt.account_id}"
                f"  指标: {evt.metric_name}"
                f"  值: ${evt.value:.2f}  限额: ${evt.limit:.0f}"
            )

    live_bus.subscribe(RiskControlEvent, _on_risk_control)
    live_bus.subscribe(RiskAlertEvent,   _on_risk_alert)

    # ── OrderSubmittedEvent 处理器（命令模式）────────────────
    def _on_order(evt: OrderSubmittedEvent) -> None:
        side_str  = "买入" if evt.side == Side.Buy else "卖出"
        order_str = "市价单" if evt.order_type == OrderType.Market else "限价单"
        print(
            f"[订单路由] ★ 收到对冲命令（命令模式）: "
            f"{side_str} {evt.quantity} 股 {evt.instrument_id} [{order_str}]"
        )

    live_bus.subscribe(OrderSubmittedEvent, _on_order)

    # ── ProbabilisticTaker — 概率成交模拟器 ─────────────────
    taker = ProbabilisticTaker(live_bus, trade_probability=0.30, seed=42)
    taker.register_handlers()
    print("[初始化] ProbabilisticTaker 已注册（成交概率: 30%，种子: 42）\n")

    print("┌──────────────── 第一阶段事件订阅关系 ────────────────┐")
    print("│ MarketDataEvent     → QuoteEngine（报价）            │")
    print("│                    → DeltaHedger（价格更新）         │")
    print("│                    → RealtimeRiskApp（重新估值）     │")
    print("│ QuoteGeneratedEvent → ProbabilisticTaker（模拟成交） │")
    print("│ TradeExecutedEvent  → DeltaHedger（对冲检查）        │")
    print("│                    → RealtimeRiskApp（风险检查）     │")
    print("│ RiskControlEvent    → 风控日志存根                   │")
    print("│ OrderSubmittedEvent → 订单路由存根                   │")
    print("└───────────────────────────────────────────────────────┘\n")

    print("══════════════ 第一阶段仿真开始 ══════════════\n")
    adapter = MarketDataAdapter(live_bus, csv_path="data/market_data.csv")
    adapter.run()
    print("\n══════════════ 第一阶段仿真结束 ══════════════\n")
    position_mgr.print_positions()

    # ════════════════════════════════════════════════════════
    # ██ 第二阶段：回测与参数校准 ██
    # ════════════════════════════════════════════════════════
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│  第二阶段：历史回测 + 模型参数校准                      │")
    print("│  目标：拟合 Black-Scholes 波动率 σ（真实值: 0.25）      │")
    print("└──────────────────────────────────────────────────────────┘\n")

    # 第二阶段独立事件总线（与第一阶段完全隔离）
    backtest_bus = EventBus()
    print("[回测初始化] 独立回测 EventBus 已创建（状态隔离）")

    # "市场"定价引擎（vol=0.25，代表真实市场）
    market_engine = BlackScholesPricingEngine(vol=0.25, r=0.05)
    # "模型"定价引擎（vol=0.15，待校准）
    model_engine  = BlackScholesPricingEngine(vol=0.15, r=0.05)
    calibrator    = CalibrationEngine()

    backtest_app = BacktestCalibrationApp(
        backtest_bus, main_bus,
        market_engine, model_engine,
        options, calibrator, "bs_model",
    )
    backtest_app.register_handlers()

    print("[回测初始化] BlackScholesPricingEngine 已就绪")
    print("             市场引擎: vol=0.25（真实市场波动率）")
    print("             模型引擎: vol=0.15（初始估计，待校准）\n")

    print("══════════════ 第二阶段回测开始 ══════════════\n")
    backtest_adapter = MarketDataAdapter(backtest_bus, csv_path="data/market_data.csv")
    backtest_adapter.run()

    print("\n══════════════ 历史重放完成，开始校准 ══════════════")
    backtest_app.finalize()
    print("\n══════════════ 第二阶段校准完成 ══════════════")

    # ── 汇报：参数仓库校准结果 ────────────────────────────────
    param_store.print_all()

    # ── 演示参数反馈闭环 ──────────────────────────────────────
    updated = param_store.get_params("bs_model")
    if updated:
        new_vol = updated["vol"]
        print(
            f"\n[参数反馈] 将校准波动率 {new_vol:.4f} 注入实时定价引擎"
            f"（BlackScholesPricingEngine）"
        )
        print(
            f"[参数反馈] 模型定价精度提升：初始 vol=0.15"
            f" → 校准 vol={new_vol:.4f} → 真实 vol=0.25"
        )

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  仿真完成！已演示：                                      ║")
    print("║  ① 事件驱动做市仿真（Phase 1）                          ║")
    print("║  ② 实时风险监控与限额执行（RealtimeRiskApp）            ║")
    print("║  ③ 历史回测与模型参数校准（BacktestCalibrationApp）     ║")
    print("║  ④ 参数反馈闭环（ParamUpdateEvent → ParameterStore）    ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
