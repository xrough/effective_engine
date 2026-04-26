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

// Seller module (market making + risk + backtest calibration)
#include "modules/seller/SellerModule.hpp"
#include "modules/seller/BacktestCalibrationApp.hpp"

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║        Effective Engine MVP — Event-Driven Simulation    ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ────────────────────────────────────────────────────────
    // 共享领域对象（合约）— 创建一次，跨模块共享只读引用
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

    std::cout << "[Init] 创建合约:\n";
    std::cout << "  " << underlying->id() << " (" << underlying->type_name() << ")\n";
    for (const auto& opt : options) {
        std::cout << "  " << opt->id() << " (" << opt->type_name() << ")\n";
    }
    std::cout << "\n";

    // ────────────────────────────────────────────────────────
    // 主总线 — 跨模块关切（参数更新）
    // ────────────────────────────────────────────────────────
    auto main_bus   = std::make_shared<omm::events::EventBus>();
    auto param_store = std::make_shared<omm::infrastructure::ParameterStore>(main_bus);
    param_store->subscribe_handlers();
    std::cout << "[Init] ParameterStore 已注册 (主总线)\n\n";

    // ════════════════════════════════════════════════════════
    // 卖方（做市 + 风控 + 回测校准）
    //   独立总线 + 独立定价引擎
    //   校准结果通过主总线 → ParameterStore → 热注入
    // ════════════════════════════════════════════════════════
    std::cout << "┌──────────────────────────────────────────────────────────┐\n"
              << "│  Module 2: Seller — Market Making + Risk + Calibration   │\n"
              << "└──────────────────────────────────────────────────────────┘\n\n";

    auto seller_bus          = std::make_shared<omm::events::EventBus>();
    auto seller_rough_engine = std::make_shared<omm::domain::RoughVolPricingEngine>(
        omm::domain::RoughVolParams{}, 0.05
    );
    std::cout << "[Init] Pricing strategy: RoughVolPricingEngine (H=0.1, Bergomi-Guyon skew)\n";

    omm::seller::SellerConfig seller_cfg;
    seller_cfg.underlying_id = underlying->id();

    auto seller_ctx = omm::seller::SellerModule::install(
        seller_bus, seller_cfg, options, seller_rough_engine
    );

    std::cout << "\n  [Seller Bus] 事件流:\n"
              << "  MarketDataEvent      → QuoteEngine / DeltaHedger / PortfolioService\n"
              << "  QuoteGeneratedEvent  → ProbabilisticTaker\n"
              << "  FillEvent            → PortfolioService / DeltaHedger\n"
              << "  PortfolioUpdateEvent → SellerRiskApp\n"
              << "  OrderSubmittedEvent  → OrderRouter\n\n";

    // ── 实时仿真 ─────────────────────────────────────────────
    std::cout << "══════════════ Seller Simulation Start ══════════════\n\n";

    omm::infrastructure::MarketDataAdapter adapter(seller_bus, "data/market_data.csv");
    adapter.run();
    seller_ctx.order_router->flush_all();

    std::cout << "\n══════════════ Seller Simulation End ══════════════\n\n";
    seller_ctx.position_mgr->print_positions();

    // ── 回测与校准 ─────────────────────────────────────────────
    std::cout << "\n┌──────────────────────────────────────────────────────────┐\n"
              << "│  Phase: Historical Backtest + Model Calibration          │\n"
              << "│  Goal: fit Black-Scholes volatility σ (true value: 0.25) │\n"
              << "└──────────────────────────────────────────────────────────┘\n\n";

    auto backtest_bus = std::make_shared<omm::events::EventBus>();
    std::cout << "[Backtest Init] isolated backtest EventBus created\n";

    auto market_engine = std::make_shared<omm::domain::BlackScholesPricingEngine>(0.25, 0.05);
    auto model_engine  = std::make_shared<omm::domain::BlackScholesPricingEngine>(0.15, 0.05);
    auto calibrator    = std::make_shared<omm::domain::CalibrationEngine>();

    auto backtest_app = std::make_shared<omm::application::BacktestCalibrationApp>(
        backtest_bus, main_bus, market_engine, model_engine, options, calibrator, "bs_model"
    );
    backtest_app->register_handlers();

    std::cout << "[Backtest Init] market engine: vol=0.25 | model engine: vol=0.15\n\n"
              << "══════════════ Backtest Start ══════════════\n\n";

    omm::infrastructure::MarketDataAdapter backtest_adapter(backtest_bus, "data/market_data.csv");
    backtest_adapter.run();

    std::cout << "\n══════════════ Calibrating ══════════════\n";
    backtest_app->finalize();
    std::cout << "\n══════════════ Calibration Complete ══════════════\n";

    param_store->print_all();

    // ── 参数反馈闭环 ─────────────────────────────────────────
    auto updated_params = param_store->get_params("bs_model");
    if (!updated_params.empty()) {
        double new_vol = updated_params.at("vol");
        std::cout << "\n[Param Feedback] BS vol: initial=0.15 → calibrated="
                  << std::fixed << std::setprecision(4) << new_vol << " → true=0.25\n";
    }

    auto rough_calibrated = param_store->get_params("rough_bergomi");
    if (!rough_calibrated.empty()) {
        omm::domain::RoughVolParams calibrated{
            rough_calibrated.at("H"),
            rough_calibrated.at("eta"),
            rough_calibrated.at("rho"),
            rough_calibrated.at("xi0")
        };
        seller_rough_engine->update_params(calibrated);
        std::cout << "\n[粗糙波动率反馈] 校准参数已注入 seller RoughVolPricingEngine:\n"
                  << "  H=" << calibrated.H << "  eta=" << calibrated.eta
                  << "  rho=" << calibrated.rho << "  xi0=" << calibrated.xi0 << "\n";

        auto bs_compare = std::make_shared<omm::domain::BlackScholesPricingEngine>(0.25, 0.05);
        double spot_ref = 150.0;
        std::cout << "\n[粗糙波动率反馈] Delta 对比（标的价=$" << spot_ref << "）:\n";
        std::cout << "  " << std::setw(14) << "合约"
                  << "  BS Δ(σ=0.25)  RoughVol Δ\n";
        for (const auto& opt : options) {
            auto bs_r = bs_compare->price(*opt, spot_ref);
            auto rv_r = seller_rough_engine->price(*opt, spot_ref);
            std::cout << "  " << std::setw(14) << opt->id()
                      << "  " << std::showpos << std::fixed << std::setprecision(4)
                      << bs_r.delta << "      " << rv_r.delta << std::noshowpos << "\n";
        }
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  Complete.                                               ║\n"
              << "║  Seller: market making + risk + calibration              ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
