#!/usr/bin/env python3
# ============================================================
# 文件：hybrid_main.py
# 职责：混合模式组合根（Composition Root）
#
# 架构：
#   C++ omm_core（核心层）：EventBus、领域对象、基础设施适配器
#   Python hybrid_application/（应用层）：QuoteEngine、DeltaHedger、
#         RealtimeRiskApp、BacktestCalibrationApp
#
# 与 main.py 的对比：
#   main.py          — 纯 Python（events / domain / infra 均用 Python 实现）
#   hybrid_main.py   — C++ 核心 + Python 应用层（通过 pybind11 桥接）
#
# 运行：
#   export PYTHONPATH=/path/to/build:$PYTHONPATH  # build/ 内含 omm_core.so
#   python3 python/hybrid_main.py
# ============================================================

from __future__ import annotations
import sys
import os
import datetime

# ── 将 python/ 目录加入 sys.path（支持直接运行）──────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── C++ 核心层（via pybind11）────────────────────────────────
import omm_core  # EventBus、领域对象、基础设施适配器

# ── Python 应用层（hybrid_application/）─────────────────────
from hybrid_application.quote_engine              import QuoteEngine
from hybrid_application.delta_hedger              import DeltaHedger
from hybrid_application.realtime_risk_app         import RealtimeRiskApp
from hybrid_application.backtest_calibration_app  import BacktestCalibrationApp


def main() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  期权做市商 MVP — 混合模式（C++ 核心 + Python 应用层）  ║")
    print("║  核心层：omm_core（pybind11 扩展）                      ║")
    print("║  应用层：python/hybrid_application/                     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── 公共资源：两阶段共享，通过 C++ InstrumentFactory 创建 ────
    expiry     = datetime.datetime.now() + datetime.timedelta(days=30)
    underlying = omm_core.InstrumentFactory.make_underlying("AAPL")
    call_150   = omm_core.InstrumentFactory.make_call("AAPL", 150.0, expiry)
    call_145   = omm_core.InstrumentFactory.make_call("AAPL", 145.0, expiry)
    put_155    = omm_core.InstrumentFactory.make_put("AAPL",  155.0, expiry)
    options    = [call_150, call_145, put_155]

    print("[初始化] C++ InstrumentFactory 创建合约:")
    print(f"  {underlying.id} ({underlying.type_name()})")
    for opt in options:
        print(f"  {opt.id} ({opt.type_name()})")
    print()

    # ── 主事件总线（两阶段共享；接收 ParamUpdateEvent）────────
    main_bus    = omm_core.EventBus()
    param_store = omm_core.ParameterStore(main_bus)
    param_store.subscribe_handlers()
    print("[初始化] C++ ParameterStore 已注册（订阅 ParamUpdateEvent）\n")

    # ════════════════════════════════════════════════════════
    # ██ 第一阶段：实时做市仿真 + 实时风控（混合模式）██
    # ════════════════════════════════════════════════════════
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  第一阶段：实时做市仿真 + 实时风控监控（混合模式）      │")
    print("│  EventBus / Infra = C++    Application Layer = Python    │")
    print("└──────────────────────────────────────────────────────────┘\n")

    live_bus = omm_core.EventBus()  # C++ EventBus（第一阶段专用）
    print("[初始化] C++ EventBus 已创建（实时仿真专用）")

    # ── 定价策略（C++ SimplePricingEngine）──────────────────
    pricing_engine = omm_core.SimplePricingEngine()
    position_mgr   = omm_core.PositionManager()
    print("[初始化] C++ 定价策略: SimplePricingEngine（内在价值定价）")

    # ── Python QuoteEngine — 固定价差报价策略 ───────────────
    quote_engine = QuoteEngine(live_bus, pricing_engine, options, half_spread=0.05)
    quote_engine.register_handlers()
    print("[初始化] Python QuoteEngine 已注册（策略：固定价差 ±$0.05）")

    # ── Python DeltaHedger — Delta 对冲策略 ─────────────────
    delta_hedger = DeltaHedger(
        live_bus, position_mgr, pricing_engine, options,
        underlying.id, delta_threshold=0.5,
    )
    delta_hedger.register_handlers()
    # 订阅行情事件，让 DeltaHedger 始终掌握最新标的价格
    live_bus.subscribe_market_data(
        lambda evt: delta_hedger.update_market_price(evt.underlying_price)
    )
    print("[初始化] Python DeltaHedger 已注册（阈值: ±0.5 Delta）")

    # ── Python RealtimeRiskApp — 实时风险监控 ────────────────
    risk_policy = omm_core.SimpleRiskPolicy(
        loss_limit     = 1e6,
        delta_limit    = 10000.0,
        drawdown_limit = 5e5,
    )
    risk_app = RealtimeRiskApp(
        live_bus, pricing_engine, options,
        underlying.id, "DESK_A", risk_policy,
    )
    risk_app.register_handlers()
    print("[初始化] Python RealtimeRiskApp 已注册（账户: DESK_A）")

    # ── 风控事件日志处理器（C++ EventBus 订阅）───────────────
    def _on_risk_control(evt: omm_core.RiskControlEvent) -> None:
        print(
            f"[风控日志] 📋 风控指令已记录: 账户={evt.account_id}"
            f"  动作={evt.action.name}  原因: {evt.reason}"
        )

    def _on_risk_alert(evt: omm_core.RiskAlertEvent) -> None:
        if evt.value > 1.0:
            print(
                f"[风险预警] 📊 {evt.account_id}"
                f"  指标: {evt.metric_name}"
                f"  值: ${evt.value:.2f}  限额: ${evt.limit:.0f}"
            )

    live_bus.subscribe_risk_control(_on_risk_control)
    live_bus.subscribe_risk_alert(_on_risk_alert)

    # ── OrderSubmittedEvent 处理器（命令模式）────────────────
    def _on_order(evt: omm_core.OrderSubmittedEvent) -> None:
        side_str  = "买入" if evt.side == omm_core.Side.Buy else "卖出"
        order_str = "市价单" if evt.order_type == omm_core.OrderType.Market else "限价单"
        print(
            f"[订单路由] ★ 收到对冲命令（命令模式）: "
            f"{side_str} {evt.quantity} 股 {evt.instrument_id} [{order_str}]"
        )

    live_bus.subscribe_order_submitted(_on_order)

    # ── C++ ProbabilisticTaker — 概率成交模拟器 ──────────────
    taker = omm_core.ProbabilisticTaker(live_bus, trade_probability=0.30, seed=42)
    taker.register_handlers()
    print("[初始化] C++ ProbabilisticTaker 已注册（成交概率: 30%，种子: 42）\n")

    print("┌──────────────── 第一阶段事件订阅关系（混合模式）────────┐")
    print("│ [C++] MarketDataAdapter → MarketDataEvent                │")
    print("│       → [Python] QuoteEngine（报价）                    │")
    print("│       → [Python] DeltaHedger（价格更新）                │")
    print("│       → [Python] RealtimeRiskApp（重新估值）            │")
    print("│ [Python] QuoteGeneratedEvent → [C++] ProbabilisticTaker │")
    print("│ [C++] TradeExecutedEvent → [Python] DeltaHedger         │")
    print("│                          → [Python] RealtimeRiskApp     │")
    print("└──────────────────────────────────────────────────────────┘\n")

    print("══════════════ 第一阶段仿真开始 ══════════════\n")
    adapter = omm_core.MarketDataAdapter(live_bus, "data/market_data.csv")
    adapter.run()
    print("\n══════════════ 第一阶段仿真结束 ══════════════\n")
    position_mgr.print_positions()

    # ════════════════════════════════════════════════════════
    # ██ 第二阶段：历史回测 + 参数校准（混合模式）██
    # ════════════════════════════════════════════════════════
    print("\n┌──────────────────────────────────────────────────────────┐")
    print("│  第二阶段：历史回测 + 模型参数校准（混合模式）          │")
    print("│  目标：拟合 Black-Scholes 波动率 σ（真实值: 0.25）      │")
    print("│  搜索：C++ 黄金分割 + Python loss_fn 闭包（跨语言优化）│")
    print("└──────────────────────────────────────────────────────────┘\n")

    backtest_bus = omm_core.EventBus()  # C++ EventBus（第二阶段独立）
    print("[回测初始化] 独立 C++ EventBus 已创建（状态隔离）")

    market_engine = omm_core.BlackScholesPricingEngine(vol=0.25, r=0.05)
    model_engine  = omm_core.BlackScholesPricingEngine(vol=0.15, r=0.05)
    calibrator    = omm_core.CalibrationEngine()

    backtest_app = BacktestCalibrationApp(
        backtest_bus, main_bus,
        market_engine, model_engine,
        options, calibrator, "bs_model",
    )
    backtest_app.register_handlers()

    print("[回测初始化] C++ BlackScholesPricingEngine 已就绪")
    print("             市场引擎: vol=0.25（真实市场波动率）")
    print("             模型引擎: vol=0.15（初始估计，待校准）\n")

    print("══════════════ 第二阶段回测开始 ══════════════\n")
    backtest_adapter = omm_core.MarketDataAdapter(backtest_bus, "data/market_data.csv")
    backtest_adapter.run()

    print("\n══════════════ 历史重放完成，开始校准 ══════════════")
    backtest_app.finalize()
    print("\n══════════════ 第二阶段校准完成 ══════════════")

    # ── 汇报：C++ ParameterStore 校准结果 ─────────────────────
    param_store.print_all()

    # ── 演示参数反馈闭环 ──────────────────────────────────────
    updated = param_store.get_params("bs_model")
    if updated:
        new_vol = updated["vol"]
        print(
            f"\n[参数反馈] 将校准波动率 {new_vol:.4f} 注入实时定价引擎"
            f"（C++ BlackScholesPricingEngine）"
        )
        print(
            f"[参数反馈] 模型定价精度提升：初始 vol=0.15"
            f" → 校准 vol={new_vol:.4f} → 真实 vol=0.25"
        )

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  混合模式仿真完成！已演示：                              ║")
    print("║  ① C++ EventBus 驱动 Python 应用层（跨语言事件分发）   ║")
    print("║  ② Python 应用类使用 C++ PortfolioAggregate / Policy   ║")
    print("║  ③ Python loss_fn 闭包传入 C++ CalibrationEngine::solve║")
    print("║  ④ C++ ParameterStore 接收 Python 发布的 ParamUpdate   ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
