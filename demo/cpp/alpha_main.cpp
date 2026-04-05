// ============================================================
// File: alpha_main.cpp  (demo/cpp/)
// Role: buyer-side demo — variance alpha pipeline end-to-end.
//
// Pipeline:
//   MarketDataAdapter (CSV)
//     → SyntheticOptionFeed (adapter)   → OptionMidQuoteEvent
//     → ImpliedVarianceExtractor        → ImpliedVariancePoint
//     → VarianceAlphaSignal             → SignalSnapshotEvent
//     → StrategyController              → OrderSubmittedEvent
//     → SimpleExecSim (adapter)         → FillEvent
//     → DeltaHedger                     → OrderSubmittedEvent (hedge)
//     → AlphaPnLTracker                 → StrategyPnLBreakdown
// ============================================================

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <chrono>

#include "core/events/EventBus.hpp"
#include "core/domain/Instrument.hpp"
#include "core/domain/InstrumentFactory.hpp"
#include "core/domain/PositionManager.hpp"
#include "core/analytics/RoughVolPricingEngine.hpp"
#include "core/analytics/ImpliedVarianceExtractor.hpp"
#include "core/infrastructure/MarketDataAdapter.hpp"
#include "modules/buyer/BuyerModule.hpp"
#include "modules/seller/DeltaHedger.hpp"

// Demo-local files (same directory)
#include "SyntheticOptionFeed.hpp"
#include "SimpleExecSim.hpp"
#include "AlphaPnLTracker.hpp"
#include "VarianceAlphaSignal.hpp"
#include "StrategyController.hpp"

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║   Variance Alpha Pipeline Demo — Buyer Side              ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ── 事件总线 ─────────────────────────────────────────────
    auto bus = std::make_shared<omm::events::EventBus>();

    // ── 定价引擎（粗糙模型） ──────────────────────────────────
    auto rough_engine = std::make_shared<omm::domain::RoughVolPricingEngine>(
        omm::domain::RoughVolParams{}, 0.05
    );

    // ── 模拟适配器（demo infrastructure） ────────────────────
    auto feed     = std::make_shared<omm::demo::SyntheticOptionFeed>(bus);
    auto exec_sim = std::make_shared<omm::demo::SimpleExecSim>(bus);
    feed->register_handlers();
    exec_sim->register_handlers();
    std::cout << "[Demo] SyntheticOptionFeed + SimpleExecSim 已注册\n";

    // ── 买方核心组件（注入到 BuyerModule） ────────────────────
    auto extractor  = std::make_shared<omm::analytics::ImpliedVarianceExtractor>(bus);

    omm::buyer::AlphaSignalConfig signal_cfg;
    signal_cfg.window = 10;   // demo: 缩小窗口适配 20 条 tick 数据（生产默认 50）
    omm::buyer::StrategyControllerConfig ctrl_cfg;  // 默认: max_holding=100

    auto signal     = std::make_shared<omm::buyer::VarianceAlphaSignal>(
        bus, extractor, rough_engine, signal_cfg
    );
    auto controller = std::make_shared<omm::buyer::StrategyController>(
        bus, signal_cfg, ctrl_cfg
    );

    // 调用方在 install() 前注册 signal + controller
    signal->register_handlers();
    controller->register_handlers();

    omm::buyer::BuyerModule::install(bus, extractor, signal, controller);
    std::cout << "\n";

    // ── Delta 对冲 ────────────────────────────────────────────
    auto expiry   = std::chrono::system_clock::now() + std::chrono::hours(30 * 24);
    auto call_atm = omm::domain::InstrumentFactory::make_call("AAPL", 150.0, expiry);
    auto put_atm  = omm::domain::InstrumentFactory::make_put ("AAPL", 150.0, expiry);
    std::vector<std::shared_ptr<omm::domain::Option>> hedge_options = {call_atm, put_atm};

    auto position_mgr = std::make_shared<omm::domain::PositionManager>();
    auto delta_hedger = std::make_shared<omm::application::DeltaHedger>(
        bus, position_mgr, rough_engine, hedge_options, "AAPL", /*threshold=*/0.3
    );
    delta_hedger->register_handlers();

    // DeltaHedger 需要跟踪最新标的价格
    bus->subscribe<omm::events::MarketDataEvent>(
        [&delta_hedger](const omm::events::MarketDataEvent& e) {
            delta_hedger->update_market_price(e.underlying_price);
        }
    );
    std::cout << "[Demo] DeltaHedger 已注册 (阈值=0.3)\n";

    // ── PnL 追踪（行为组件，不依赖 demo 适配器） ─────────────
    auto pnl_tracker = std::make_shared<omm::demo::AlphaPnLTracker>(bus);
    pnl_tracker->register_handlers();
    std::cout << "[Demo] AlphaPnLTracker 已注册\n\n";

    // ── 事件流说明 ────────────────────────────────────────────
    std::cout << "  [事件流]\n"
              << "  MarketDataEvent\n"
              << "    → SyntheticOptionFeed  → OptionMidQuoteEvent\n"
              << "    → ImpliedVarianceExtractor (σ²_atm)\n"
              << "    → VarianceAlphaSignal  → SignalSnapshotEvent (zscore)\n"
              << "    → StrategyController   → OrderSubmittedEvent (straddle)\n"
              << "    → SimpleExecSim        → FillEvent\n"
              << "    → DeltaHedger          → OrderSubmittedEvent (hedge)\n"
              << "    → AlphaPnLTracker      (option_mtm + delta_hedge_pnl)\n\n";

    // ── 运行仿真 ──────────────────────────────────────────────
    std::cout << "══════════════ Alpha Pipeline Start ══════════════\n\n";
    omm::infrastructure::MarketDataAdapter adapter(bus, "../data/market_data.csv");
    adapter.run();
    std::cout << "\n══════════════ Alpha Pipeline End ══════════════\n";

    // ── 最终持仓 + PnL 归因 ───────────────────────────────────
    position_mgr->print_positions();
    pnl_tracker->print_summary();

    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║  Complete.                                               ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
