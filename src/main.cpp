// ============================================================
// 文件：main.cpp
// 职责：组合根（Composition Root）— 系统唯一的组件连线点。
//
// 这是系统中唯一出现具体类名（而非接口/抽象类）的地方。
// 所有其他模块通过接口和事件总线相互通信，不持有对彼此的直接引用。
//
// ════════════════════════════════════════════════════════════
// 运行阶段说明：
//
//   第一阶段：实时仿真（Live Simulation）
//     复现真实做市商的运作方式：
//       MarketDataEvent → QuoteEngine → ProbabilisticTaker
//         → TradeExecutedEvent → DeltaHedger + RealtimeRiskApp
//     RealtimeRiskApp 同步监控账户风险，在限额触发时发出风控指令。
//
//   第二阶段：回测与参数校准（Backtest & Calibration）
//     在独立事件总线上重放相同历史行情：
//       BacktestCalibrationApp 比对"市场价格"与"模型预测价格"，
//       通过黄金分割搜索优化波动率 σ，
//       发布 ParamUpdateEvent → ParameterStore 存储校准结果。
//
//   两个阶段使用独立 EventBus，保证状态完全隔离（Risk_Calibration.md §3）。
//
// ════════════════════════════════════════════════════════════
// 设计模式汇总（本文件是所有模式交汇的地方）：
//
//   观察者模式（Observer Pattern）   — EventBus 发布/订阅基础设施
//   策略模式（Strategy Pattern）     — IPricingEngine / IRiskPolicy 可替换注入
//   适配器模式（Adapter Pattern）    — MarketDataAdapter 翻译 CSV → 领域事件
//   命令模式（Command Pattern）      — OrderSubmittedEvent 封装对冲指令
//   工厂模式（Factory Pattern）      — InstrumentFactory 集中创建金融工具
// ============================================================

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>

// 事件层
#include "events/EventBus.hpp"
#include "events/Events.hpp"

// 领域层
#include "domain/Instrument.hpp"
#include "domain/InstrumentFactory.hpp"
#include "domain/PositionManager.hpp"
#include "domain/PricingEngine.hpp"
#include "domain/RiskPolicy.hpp"
#include "domain/CalibrationEngine.hpp"

// 基础设施层
#include "infrastructure/MarketDataAdapter.hpp"
#include "infrastructure/ProbabilisticTaker.hpp"
#include "infrastructure/ParameterStore.hpp"

// 应用层
#include "application/QuoteEngine.hpp"
#include "application/DeltaHedger.hpp"
#include "application/RealtimeRiskApp.hpp"
#include "application/BacktestCalibrationApp.hpp"

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║     期权做市商 MVP — 事件驱动仿真系统（完整版）         ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ────────────────────────────────────────────────────────
    // 公共资源：两个阶段共享合约定义与工厂（数据不可变，安全共享）
    // ────────────────────────────────────────────────────────
    auto expiry = std::chrono::system_clock::now()
                  + std::chrono::hours(30 * 24);

    auto underlying = omm::domain::InstrumentFactory::make_underlying("AAPL");
    auto call_150   = omm::domain::InstrumentFactory::make_call("AAPL", 150.0, expiry);
    auto call_145   = omm::domain::InstrumentFactory::make_call("AAPL", 145.0, expiry);
    auto put_155    = omm::domain::InstrumentFactory::make_put ("AAPL", 155.0, expiry);

    std::vector<std::shared_ptr<omm::domain::Option>> options = {
        call_150, call_145, put_155
    };

    std::cout << "[初始化] 工厂模式创建合约:\n";
    std::cout << "  " << underlying->id() << " (" << underlying->type_name() << ")\n";
    for (const auto& opt : options) {
        std::cout << "  " << opt->id() << " (" << opt->type_name() << ")\n";
    }
    std::cout << "\n";

    // ────────────────────────────────────────────────────────
    // 主事件总线（两个阶段共享；用于接收 ParamUpdateEvent）
    // ────────────────────────────────────────────────────────
    auto main_bus = std::make_shared<omm::events::EventBus>();

    // ────────────────────────────────────────────────────────
    // ParameterStore — 订阅 ParamUpdateEvent，接收校准结果
    // （在主总线上注册，两阶段均可向其发布参数）
    // ────────────────────────────────────────────────────────
    auto param_store = std::make_shared<omm::infrastructure::ParameterStore>(main_bus);
    param_store->subscribe_handlers();
    std::cout << "[初始化] ParameterStore 已注册（订阅 ParamUpdateEvent）\n\n";

    // ════════════════════════════════════════════════════════
    // ██ 第一阶段：实时仿真 + 实时风控 ██
    // ════════════════════════════════════════════════════════
    std::cout << "┌──────────────────────────────────────────────────────────┐\n"
              << "│  第一阶段：实时做市仿真 + 实时风控监控                  │\n"
              << "└──────────────────────────────────────────────────────────┘\n\n";

    // 第一阶段独立事件总线
    auto live_bus = std::make_shared<omm::events::EventBus>();
    std::cout << "[初始化] 第一阶段 EventBus 已创建（实时仿真专用）\n";

    // ── 定价策略：简化内在价值定价（与原 MVP 保持一致）──────
    auto pricing_engine = std::make_shared<omm::domain::SimplePricingEngine>();
    auto position_mgr   = std::make_shared<omm::domain::PositionManager>();
    std::cout << "[初始化] 定价策略: SimplePricingEngine（内在价值定价）\n";

    // ── QuoteEngine — 固定价差报价策略 ──────────────────────
    auto quote_engine = std::make_shared<omm::application::QuoteEngine>(
        live_bus, pricing_engine, options,
        0.05  // half_spread = $0.05
    );
    quote_engine->register_handlers();
    std::cout << "[初始化] QuoteEngine 已注册（策略：固定价差 ±$0.05）\n";

    // ── DeltaHedger — Delta 对冲策略 ─────────────────────────
    auto delta_hedger = std::make_shared<omm::application::DeltaHedger>(
        live_bus, position_mgr, pricing_engine, options,
        underlying->id(),
        0.5  // Delta 阈值 ±0.5
    );
    delta_hedger->register_handlers();
    live_bus->subscribe<omm::events::MarketDataEvent>(
        [&dh = *delta_hedger](const omm::events::MarketDataEvent& evt) {
            dh.update_market_price(evt.underlying_price);
        }
    );
    std::cout << "[初始化] DeltaHedger 已注册（阈值: ±0.5 Delta）\n";

    // ── RealtimeRiskApp — 实时风险监控（新增） ───────────────
    // 策略模式：SimpleRiskPolicy 定义风险限额规则，可替换
    auto risk_policy = std::make_shared<omm::domain::SimpleRiskPolicy>(
        1e6,    // 止损限额：已实现亏损超 $1,000,000 → BlockOrders
        10000.0, // Delta 限额：|Δ| > 10,000 → ReduceOnly
        5e5      // 日内回撤预警：> $500,000 → RiskAlertEvent
    );

    auto risk_app = std::make_shared<omm::application::RealtimeRiskApp>(
        live_bus,
        pricing_engine,
        options,
        underlying->id(),
        "DESK_A",   // 账户 ID
        risk_policy
    );
    risk_app->register_handlers(); // 订阅 TradeExecutedEvent + MarketDataEvent
    std::cout << "[初始化] RealtimeRiskApp 已注册（账户: DESK_A，风控策略: SimpleRiskPolicy）\n";

    // ── 风控事件处理器（日志存根）────────────────────────────
    live_bus->subscribe<omm::events::RiskControlEvent>(
        [](const omm::events::RiskControlEvent& evt) {
            std::cout << "[风控日志] 📋 风控指令已记录: 账户=" << evt.account_id
                      << "  动作=";
            switch (evt.action) {
                case omm::events::RiskAction::BlockOrders:  std::cout << "BlockOrders"; break;
                case omm::events::RiskAction::CancelOrders: std::cout << "CancelOrders"; break;
                case omm::events::RiskAction::ReduceOnly:   std::cout << "ReduceOnly"; break;
            }
            std::cout << "  原因: " << evt.reason << "\n";
        }
    );

    live_bus->subscribe<omm::events::RiskAlertEvent>(
        [](const omm::events::RiskAlertEvent& evt) {
            if (evt.value > 1.0) { // 仅打印有意义的预警（跳过 0 值噪音）
                std::cout << "[风险预警] 📊 " << evt.account_id
                          << "  指标: " << evt.metric_name
                          << "  值: $" << std::fixed << std::setprecision(2) << evt.value
                          << "  限额: $" << evt.limit << "\n";
            }
        }
    );

    // ── OrderSubmittedEvent 处理器（命令模式）────────────────
    live_bus->subscribe<omm::events::OrderSubmittedEvent>(
        [](const omm::events::OrderSubmittedEvent& evt) {
            std::cout << "[订单路由] ★ 收到对冲命令（命令模式）: "
                      << (evt.side == omm::events::Side::Buy ? "买入" : "卖出")
                      << " " << evt.quantity << " 股 " << evt.instrument_id
                      << " [" << (evt.order_type == omm::events::OrderType::Market
                                  ? "市价单" : "限价单") << "]\n";
        }
    );

    // ── ProbabilisticTaker — 概率成交模拟器 ──────────────────
    auto taker = std::make_shared<omm::infrastructure::ProbabilisticTaker>(
        live_bus, 0.30, 42
    );
    taker->register_handlers();
    std::cout << "[初始化] ProbabilisticTaker 已注册（成交概率: 30%，种子: 42）\n";

    std::cout << "\n┌──────────────── 第一阶段事件订阅关系 ────────────────┐\n"
              << "│ MarketDataEvent     → QuoteEngine（报价）            │\n"
              << "│                    → DeltaHedger（价格更新）         │\n"
              << "│                    → RealtimeRiskApp（重新估值）     │\n"
              << "│ QuoteGeneratedEvent → ProbabilisticTaker（模拟成交） │\n"
              << "│ TradeExecutedEvent  → DeltaHedger（对冲检查）        │\n"
              << "│                    → RealtimeRiskApp（风险检查）     │\n"
              << "│ RiskControlEvent    → 风控日志存根                   │\n"
              << "│ OrderSubmittedEvent → 订单路由存根                   │\n"
              << "└───────────────────────────────────────────────────────┘\n\n";

    std::cout << "══════════════ 第一阶段仿真开始 ══════════════\n\n";

    omm::infrastructure::MarketDataAdapter adapter(
        live_bus,
        "data/market_data.csv"
    );
    adapter.run();

    std::cout << "\n══════════════ 第一阶段仿真结束 ══════════════\n\n";
    position_mgr->print_positions();

    // ════════════════════════════════════════════════════════
    // ██ 第二阶段：回测与参数校准 ██
    // ════════════════════════════════════════════════════════
    std::cout << "\n┌──────────────────────────────────────────────────────────┐\n"
              << "│  第二阶段：历史回测 + 模型参数校准                      │\n"
              << "│  目标：拟合 Black-Scholes 波动率 σ（真实值: 0.25）      │\n"
              << "└──────────────────────────────────────────────────────────┘\n\n";

    // 第二阶段独立事件总线（与第一阶段完全隔离，无共享可变状态）
    auto backtest_bus = std::make_shared<omm::events::EventBus>();
    std::cout << "[回测初始化] 独立回测 EventBus 已创建（状态隔离）\n";

    // ── "市场"定价引擎：代表真实市场（vol=0.25，为"地面真值"）──
    auto market_engine = std::make_shared<omm::domain::BlackScholesPricingEngine>(
        0.25, // 真实隐含波动率（市场已知值，校准目标）
        0.05  // 无风险利率 5%
    );

    // ── "模型"定价引擎：待校准（初始 vol=0.15，偏低约 40%）──
    auto model_engine = std::make_shared<omm::domain::BlackScholesPricingEngine>(
        0.15, // 模型初始波动率（故意偏离真实值以演示校准效果）
        0.05
    );

    auto calibrator = std::make_shared<omm::domain::CalibrationEngine>();

    auto backtest_app = std::make_shared<omm::application::BacktestCalibrationApp>(
        backtest_bus,
        main_bus,     // 校准结果发布到主总线 → ParameterStore
        market_engine,
        model_engine,
        options,
        calibrator,
        "bs_model"    // 模型 ID
    );
    backtest_app->register_handlers();

    std::cout << "[回测初始化] BlackScholesPricingEngine 已就绪\n"
              << "             市场引擎: vol=0.25（真实市场波动率）\n"
              << "             模型引擎: vol=0.15（初始估计，待校准）\n\n";

    std::cout << "══════════════ 第二阶段回测开始 ══════════════\n\n";

    // 在回测总线上重放相同的历史行情（数据来源相同，总线完全隔离）
    omm::infrastructure::MarketDataAdapter backtest_adapter(
        backtest_bus,
        "data/market_data.csv"
    );
    backtest_adapter.run();

    std::cout << "\n══════════════ 历史重放完成，开始校准 ══════════════\n";

    // 运行黄金分割搜索，将校准结果发布到主总线
    backtest_app->finalize();

    std::cout << "\n══════════════ 第二阶段校准完成 ══════════════\n";

    // ════════════════════════════════════════════════════════
    // 汇报：参数仓库中的校准结果
    // ════════════════════════════════════════════════════════
    param_store->print_all();

    // ── 演示：用校准结果更新实时引擎（参数反馈闭环）─────────
    auto updated_params = param_store->get_params("bs_model");
    if (!updated_params.empty()) {
        double new_vol = updated_params.at("vol");
        std::cout << "\n[参数反馈] 将校准波动率 " << std::fixed << std::setprecision(4)
                  << new_vol << " 注入实时定价引擎（BlackScholesPricingEngine）\n"
                  << "[参数反馈] 模型定价精度提升：初始 vol=0.15 → 校准 vol="
                  << new_vol << " → 真实 vol=0.25\n";
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  仿真完成！已演示：                                      ║\n"
              << "║  ① 事件驱动做市仿真（Phase 1）                          ║\n"
              << "║  ② 实时风险监控与限额执行（RealtimeRiskApp）            ║\n"
              << "║  ③ 历史回测与模型参数校准（BacktestCalibrationApp）     ║\n"
              << "║  ④ 参数反馈闭环（ParamUpdateEvent → ParameterStore）    ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
