// ============================================================
// File: main.cpp
// Responsibility: Composition Root — The only point where concrete classes are instantiated.
// ════════════════════════════════════════════════════════════
// Design Patterns
//
//   Observer Pattern   — EventBus, e.g. PortfolioService subscribes to MarketDataEvent; DeltaHedger releases QuoteGeneratedEvent etc. 

//   Strategy Pattern   — The abstract interface defines the contract, the concrete class provides the implementation, and main.cpp is the only place that decides which concrete class to use e.g. IPricingEngine and IRiskPolicy interfaces allow dynamic swapping of pricing and risk strategies (RoughVolPricingEngine, SimpleRiskPolicy, etc.)

//   Adapter Pattern    — Adapt from external data sources (MarketDataAdapter) and services (ModelServiceClient) to internal interfaces

//   Command Pattern    — e.g. OrderSubmittedEvent encapsulates order details; OrderRouter executes commands

//   Factory Pattern    — e.g. InstrumentFactory creates domain entities (Underlying, Option) with consistent IDs
// ============================================================

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>

#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"

#include "core/domain/Instrument.hpp"
#include "core/domain/InstrumentFactory.hpp"
#include "core/domain/PositionManager.hpp"

#include "core/analytics/PricingEngine.hpp"
#include "core/analytics/RoughVolPricingEngine.hpp"
#include "core/analytics/RiskPolicy.hpp"
#include "core/analytics/CalibrationEngine.hpp"

#ifdef BUILD_GRPC_CLIENT
#include "core/infrastructure/ModelServiceClient.hpp"
#endif

#include "core/infrastructure/MarketDataAdapter.hpp"
#include "core/infrastructure/ParameterStore.hpp"
#include "core/infrastructure/OrderRouter.hpp"

#include "core/application/PortfolioService.hpp"

// Seller modules
#include "modules/seller/QuoteEngine.hpp"
#include "modules/seller/DeltaHedger.hpp"
#include "modules/seller/SellerRiskApp.hpp"
#include "modules/seller/ProbabilisticTaker.hpp"
#include "modules/seller/BacktestCalibrationApp.hpp"

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║     Options Market Maker MVP — Event-Driven Simulation   ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ────────────────────────────────────────────────────────
    // Shared domain entities (instruments) — created once, shared across phases
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

    std::cout << "[Init] Factory: created instruments:\n";
    std::cout << "  " << underlying->id() << " (" << underlying->type_name() << ")\n";
    for (const auto& opt : options) {
        std::cout << "  " << opt->id() << " (" << opt->type_name() << ")\n";
    }
    std::cout << "\n";

    // ────────────────────────────────────────────────────────
    // main bus - receives events from all components, used for cross-cutting concerns like parameter updates
    // ────────────────────────────────────────────────────────
    auto main_bus = std::make_shared<omm::events::EventBus>(); // initialize main event bus as shared pointer pointed at type omm::events::EventBus

    // ────────────────────────────────────────────────────────
    // ParameterStore — centralized repository for model parameters, subscribes to ParamUpdateEvent
    // ────────────────────────────────────────────────────────
    auto param_store = std::make_shared<omm::infrastructure::ParameterStore>(main_bus);
    param_store->subscribe_handlers();
    std::cout << "[Init] ParameterStore registered (subscribing to ParamUpdateEvent)\n\n";

    // ════════════════════════════════════════════════════════
    // ██ 第一阶段：实时仿真 + 实时风控 ██
    // ════════════════════════════════════════════════════════
    std::cout << "┌──────────────────────────────────────────────────────────┐\n"
              << "│  Phase 1: Live Market-Making Simulation + Risk Monitoring│\n"
              << "└──────────────────────────────────────────────────────────┘\n\n";

    // 第一阶段独立事件总线
    auto live_bus = std::make_shared<omm::events::EventBus>();
    std::cout << "[Init] Phase 1 EventBus created (live simulation)\n";

    // ── 定价策略：粗糙波动率偏斜修正定价（Bergomi-Guyon 渐近展开）──
    // 默认参数：H=0.1, η=1.5, ρ=-0.7, ξ₀=0.0625（基于实证标定的短期权益典型值）
    // Phase 2 校准完成后通过 rough_engine->update_params() 热注入优化结果
    omm::domain::RoughVolParams rough_params{};  // 使用默认值
    auto rough_engine = std::make_shared<omm::domain::RoughVolPricingEngine>(rough_params, 0.05);
    // 以接口形式共享，供 PortfolioService / QuoteEngine / DeltaHedger 使用
    std::shared_ptr<omm::domain::IPricingEngine> pricing_engine = rough_engine;
    std::cout << "[Init] Pricing strategy: RoughVolPricingEngine (Bergomi-Guyon skew, H=0.1)\n";

    // ── PortfolioService — 模式无关的持仓追踪与估值 ─────────
    // Phase 2：从 SellerRiskApp 中提取，统一由 PortfolioService 管理持仓
    // 订阅 FillEvent + MarketDataEvent → 发布 PortfolioUpdateEvent
    auto portfolio_svc = std::make_shared<omm::application::PortfolioService>(
        live_bus, pricing_engine, options, underlying->id(), "DESK_A"
    );
    portfolio_svc->register_handlers();
    std::cout << "[Init] PortfolioService registered (account: DESK_A)\n";

    // ── DeltaHedger 独立持仓管理器（用于对冲 Delta 计算）───
    auto position_mgr = std::make_shared<omm::domain::PositionManager>();

    // ── QuoteEngine — 固定价差报价策略 ──────────────────────
    auto quote_engine = std::make_shared<omm::application::QuoteEngine>(
        live_bus, pricing_engine, options,
        0.05  // half_spread = $0.05
    );
    quote_engine->register_handlers();
    std::cout << "[Init] QuoteEngine registered (strategy: fixed spread ±$0.05)\n";

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
    std::cout << "[Init] DeltaHedger registered (threshold: ±0.5 delta)\n";

    // ── SellerRiskApp — 实时风险监控（Phase 2 简化版）─────────
    // Phase 2：只订阅 PortfolioUpdateEvent，不再管理持仓状态
    // 策略模式：SimpleRiskPolicy 定义风险限额规则，可替换
    auto risk_policy = std::make_shared<omm::domain::SimpleRiskPolicy>(
        1e6,     // 止损限额：已实现亏损超 $1,000,000 → BlockOrders
        10000.0, // Delta 限额：|Δ| > 10,000 → ReduceOnly
        5e5      // 日内回撤预警：> $500,000 → RiskAlertEvent
    );

    auto risk_app = std::make_shared<omm::application::SellerRiskApp>(
        live_bus,
        "DESK_A",   // 账户 ID
        risk_policy
    );
    risk_app->register_handlers(); // 订阅 PortfolioUpdateEvent
    std::cout << "[Init] SellerRiskApp registered (account: DESK_A, policy: SimpleRiskPolicy)\n";

    // ── 风控事件处理器（日志存根）────────────────────────────
    live_bus->subscribe<omm::events::RiskControlEvent>(
        [](const omm::events::RiskControlEvent& evt) {
            std::cout << "[Risk Log] action recorded: account=" << evt.account_id
                      << "  action=";
            switch (evt.action) {
                case omm::events::RiskAction::BlockOrders:  std::cout << "BlockOrders"; break;
                case omm::events::RiskAction::CancelOrders: std::cout << "CancelOrders"; break;
                case omm::events::RiskAction::ReduceOnly:   std::cout << "ReduceOnly"; break;
            }
            std::cout << "  reason: " << evt.reason << "\n";
        }
    );

    live_bus->subscribe<omm::events::RiskAlertEvent>(
        [](const omm::events::RiskAlertEvent& evt) {
            if (evt.value > 1.0) { // 仅打印有意义的预警（跳过 0 值噪音）
                std::cout << "[Risk Alert] " << evt.account_id
                          << "  metric: " << evt.metric_name
                          << "  value: $" << std::fixed << std::setprecision(2) << evt.value
                          << "  limit: $" << evt.limit << "\n";
            }
        }
    );

    // ── OrderRouter — 基础设施层执行边界（骨架实现）─────────
    // 当前：订阅 OrderSubmittedEvent，记录并调用 send_to_exchange() 存根
    // 未来：实现 FIX/WebSocket 协议，填单后发布 FillEvent 回总线
    auto order_router = std::make_shared<omm::infrastructure::OrderRouter>(live_bus);
    order_router->register_handlers();
    std::cout << "[Init] OrderRouter registered (skeleton — stub send_to_exchange)\n";

    // ── ProbabilisticTaker — 概率成交模拟器 ──────────────────
    auto taker = std::make_shared<omm::infrastructure::ProbabilisticTaker>(
        live_bus, 0.30, 42
    );
    taker->register_handlers();
    std::cout << "[Init] ProbabilisticTaker registered (fill probability: 30%, seed: 42)\n";

    std::cout << "\n┌──────────────── Phase 1 Event Subscriptions ─────────────────┐\n"
              << "│ MarketDataEvent       → QuoteEngine (quote)                   │\n"
              << "│                       → DeltaHedger (price update)            │\n"
              << "│                       → PortfolioService (re-valuation)       │\n"
              << "│ QuoteGeneratedEvent   → ProbabilisticTaker (simulated fill)   │\n"
              << "│ FillEvent             → PortfolioService (track position)     │\n"
              << "│                       → DeltaHedger (hedge check)             │\n"
              << "│ PortfolioUpdateEvent  → SellerRiskApp (risk check)            │\n"
              << "│ RiskControlEvent      → risk log stub                         │\n"
              << "│ OrderSubmittedEvent   → order router stub                     │\n"
              << "└───────────────────────────────────────────────────────────────┘\n\n";

    std::cout << "══════════════ Phase 1 Simulation Start ══════════════\n\n";

    omm::infrastructure::MarketDataAdapter adapter(
        live_bus,
        "data/market_data.csv"
    );
    adapter.run();

    std::cout << "\n══════════════ Phase 1 Simulation End ══════════════\n\n";
    position_mgr->print_positions();

    // ════════════════════════════════════════════════════════
    // ██ 第二阶段：回测与参数校准 ██
    // ════════════════════════════════════════════════════════
    std::cout << "\n┌──────────────────────────────────────────────────────────┐\n"
              << "│  Phase 2: Historical Backtest + Model Calibration        │\n"
              << "│  Goal: fit Black-Scholes volatility σ (true value: 0.25) │\n"
              << "└──────────────────────────────────────────────────────────┘\n\n";

    // 第二阶段独立事件总线（与第一阶段完全隔离，无共享可变状态）
    auto backtest_bus = std::make_shared<omm::events::EventBus>();
    std::cout << "[Backtest Init] isolated backtest EventBus created (state isolation)\n";

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

#ifdef BUILD_GRPC_CLIENT
    // ── gRPC 客户端（可选）：仅在 BUILD_GRPC_CLIENT=ON 时构建 ──
    // 校准服务需提前启动：cd ~/rough_pricing_env/Rough-Pricing && python3 -m roughvol.service.server
    auto grpc_client = std::make_shared<omm::infrastructure::ModelServiceClient>("localhost:50051");
    std::cout << "[Backtest Init] gRPC ModelServiceClient created (target: localhost:50051)\n";
#endif

    auto backtest_app = std::make_shared<omm::application::BacktestCalibrationApp>(
        backtest_bus,
        main_bus,     // 校准结果发布到主总线 → ParameterStore
        market_engine,
        model_engine,
        options,
        calibrator,
        "bs_model"    // 模型 ID
#ifdef BUILD_GRPC_CLIENT
        , grpc_client // 注入 gRPC 客户端 → finalize() 触发 Rough Bergomi 校准
#endif
    );
    backtest_app->register_handlers();

    std::cout << "[Backtest Init] BlackScholesPricingEngine ready\n"
              << "               market engine: vol=0.25 (true market vol)\n"
              << "               model engine:  vol=0.15 (initial estimate, to be calibrated)\n\n";

    std::cout << "══════════════ Phase 2 Backtest Start ══════════════\n\n";

    // 在回测总线上重放相同的历史行情（数据来源相同，总线完全隔离）
    omm::infrastructure::MarketDataAdapter backtest_adapter(
        backtest_bus,
        "data/market_data.csv"
    );
    backtest_adapter.run();

    std::cout << "\n══════════════ Historical replay complete — calibrating ══════════════\n";

    // 运行黄金分割搜索，将校准结果发布到主总线
    backtest_app->finalize();

    std::cout << "\n══════════════ Phase 2 Calibration Complete ══════════════\n";

    // ════════════════════════════════════════════════════════
    // 汇报：参数仓库中的校准结果
    // ════════════════════════════════════════════════════════
    param_store->print_all();

    // ── 演示：用 BS 校准结果更新实时引擎（参数反馈闭环）────────
    auto updated_params = param_store->get_params("bs_model");
    if (!updated_params.empty()) {
        double new_vol = updated_params.at("vol");
        std::cout << "\n[Param Feedback] BS vol 注入: initial=0.15 → calibrated="
                  << std::fixed << std::setprecision(4) << new_vol << " → true=0.25\n";
    }

    // ── 演示：用粗糙 Bergomi 校准结果热注入 RoughVolPricingEngine ──
    auto rough_calibrated = param_store->get_params("rough_bergomi");
    if (!rough_calibrated.empty()) {
        omm::domain::RoughVolParams calibrated{
            rough_calibrated.at("H"),
            rough_calibrated.at("eta"),
            rough_calibrated.at("rho"),
            rough_calibrated.at("xi0")
        };
        rough_engine->update_params(calibrated);
        std::cout << "\n[粗糙波动率反馈] 校准参数已注入 RoughVolPricingEngine:\n"
                  << "  H   = " << calibrated.H   << "\n"
                  << "  eta = " << calibrated.eta  << "\n"
                  << "  rho = " << calibrated.rho  << "\n"
                  << "  xi0 = " << calibrated.xi0  << "\n";

        // Delta 对比：BS（默认 vol=0.25）vs 粗糙 Bergomi（校准后）
        auto bs_compare = std::make_shared<omm::domain::BlackScholesPricingEngine>(0.25, 0.05);
        double spot_ref = 150.0;  // 参考标的价（AAPL 近似）
        std::cout << "\n[粗糙波动率反馈] Delta 对比（标的价=$" << spot_ref << "）:\n";
        std::cout << "  " << std::setw(14) << "合约"
                  << "  BS Δ(σ=0.25)  RoughVol Δ\n";
        for (const auto& opt : options) {
            auto bs_r  = bs_compare->price(*opt, spot_ref);
            auto rv_r  = rough_engine->price(*opt, spot_ref);
            std::cout << "  " << std::setw(14) << opt->id()
                      << "  " << std::showpos << std::fixed << std::setprecision(4)
                      << bs_r.delta << "      " << rv_r.delta << std::noshowpos << "\n";
        }
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  Simulation complete! Demonstrated:                      ║\n"
              << "║  1. Event-driven market-making simulation  (Phase 1)     ║\n"
              << "║     RoughVolPricingEngine: Bergomi-Guyon skew hedging    ║\n"
              << "║  2. Live risk monitoring + limit enforcement              ║\n"
              << "║  3. Historical backtest + model calibration               ║\n"
              << "║     BS: golden-section vol search                        ║\n"
              << "║     Rough Bergomi: gRPC batch calibration (if enabled)   ║\n"
              << "║  4. Parameter feedback loop                               ║\n"
              << "║     (ParamUpdateEvent → ParameterStore → hot-inject)     ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
