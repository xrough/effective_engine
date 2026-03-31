// ============================================================
// File: main.cpp
// Responsibility: Composition Root — The only point where concrete classes are instantiated.
// ════════════════════════════════════════════════════════════
// Design Patterns
//
//   Observer Pattern   — EventBus, e.g. PortfolioService subscribes to MarketDataEvent; DeltaHedger releases QuoteGeneratedEvent etc. 

//   Strategy Pattern   — The abstract interface defines the contract, the concrete class provides the implementation, and main.cpp is the only place that decides which concrete class to use e.g. IPricingEngine and IRiskPolicy interfaces allow dynamic swapping of pricing and risk strategies (RoughVolPricingEngine, SimpleRiskPolicy, etc.)

//   Adapter Pattern    — Adapt from external data sources (MarketDataAdapter) to internal interfaces

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

#include "core/infrastructure/MarketDataAdapter.hpp"
#include "core/infrastructure/ParameterStore.hpp"

// Seller module (composition entry point — wires all seller components)
#include "modules/seller/SellerModule.hpp"
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

    // instantiate options (ptrs) vector for easy passing to modules.

    std::vector<std::shared_ptr<omm::domain::Option>> options = {
        call_150, call_145, put_155
    };

    std::cout << "[Init] Factory: created instruments:\n";
    std::cout << "  " << underlying->id() << " (" << underlying->type_name() << ")\n";
    for (const auto& opt : options) { // const:只读 &: 引用避免拷贝
        std::cout << "  " << opt->id() << " (" << opt->type_name() << ")\n";
    }
    std::cout << "\n";

    // ────────────────────────────────────────────────────────
    // main bus - communicates between channels (Phase 1 and 2 in this case), used for cross-cutting concerns like parameter updates
    // ────────────────────────────────────────────────────────
    auto main_bus = std::make_shared<omm::events::EventBus>(); // initialize main event bus as shared pointer pointed at type EventBus.

    // ────────────────────────────────────────────────────────
    // ParameterStore — 
    // BacktestCalibrationApp → ParamUpdateEvent → ParameterStore

    // ────────────────────────────────────────────────────────
    auto param_store = std::make_shared<omm::infrastructure::ParameterStore>(main_bus); // initialize ParameterStore with main_bus.
    param_store->subscribe_handlers();
    std::cout << "[Init] ParameterStore registered (subscribing to ParamUpdateEvent)\n\n";

    // ════════════════════════════════════════════════════════
    // Stage One: Live Market-Making Simulation + Risk Monitoring
    // ════════════════════════════════════════════════════════
    std::cout << "┌──────────────────────────────────────────────────────────┐\n"
              << "│  Phase 1: Live Market-Making Simulation + Risk Monitoring│\n"
              << "└──────────────────────────────────────────────────────────┘\n\n";

    // 第一阶段独立事件总线
    auto live_bus = std::make_shared<omm::events::EventBus>();
    std::cout << "[Init] Phase 1 EventBus created (live simulation)\n";

    // ── 定价策略：粗糙波动率定价引擎（Bergomi-Guyon 渐近展开）──
    // Phase 2 校准完成后通过 rough_engine->update_params() 热注入优化结果
    omm::domain::RoughVolParams rough_params{};
    auto rough_engine = std::make_shared<omm::domain::RoughVolPricingEngine>(rough_params, 0.05);
    std::cout << "[Init] Pricing strategy: RoughVolPricingEngine (H=0.1, Bergomi-Guyon skew)\n";

    // ── 卖方模块安装（组合入口）──────────────────────────────
    // SellerModule::install() 连线所有卖方组件并注册事件订阅
    omm::seller::SellerConfig seller_cfg;
    seller_cfg.underlying_id = underlying->id();

    auto seller_ctx = omm::seller::SellerModule::install(
        live_bus, seller_cfg, options, rough_engine
    );

    std::cout << "\n┌──────────────── Phase 1 Event Subscriptions ─────────────────┐\n"
              << "│ MarketDataEvent       → QuoteEngine (quote)                   │\n"
              << "│                       → DeltaHedger (price update)            │\n"
              << "│                       → PortfolioService (re-valuation)       │\n"
              << "│ QuoteGeneratedEvent   → ProbabilisticTaker (simulated fill)   │\n"
              << "│ FillEvent             → PortfolioService (track position)     │\n"
              << "│                       → DeltaHedger (hedge check)             │\n"
              << "│ PortfolioUpdateEvent  → SellerRiskApp (risk check)            │\n"
              << "│ RiskControlEvent      → risk log                              │\n"
              << "│ OrderSubmittedEvent   → OrderRouter (stub)                    │\n"
              << "└───────────────────────────────────────────────────────────────┘\n\n";

    std::cout << "══════════════ Phase 1 Simulation Start ══════════════\n\n";

    omm::infrastructure::MarketDataAdapter adapter(
        live_bus,
        "data/market_data.csv"
    );
    adapter.run();

    std::cout << "\n══════════════ Phase 1 Simulation End ══════════════\n\n";
    seller_ctx.position_mgr->print_positions();

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
              << "║  4. Parameter feedback loop                               ║\n"
              << "║     (ParamUpdateEvent → ParameterStore → hot-inject)     ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
