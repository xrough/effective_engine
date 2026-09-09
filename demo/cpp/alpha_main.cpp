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
//     → DeltaHedger / NeuralBSDEHedger  → OrderSubmittedEvent (hedge)
//     → AlphaPnLTracker                 → StrategyPnLBreakdown
//
// 对冲引擎选择（运行时）：
//   BUILD_ONNX_DEMO=ON 且模型文件存在 → NeuralBSDEHedger（BSDE 神经网络）
//   否则                              → DeltaHedger（解析 BS Delta）
// ============================================================

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <chrono>
#include <filesystem>
#include <string>

#include "core/events/EventBus.hpp"
#include "core/domain/Instrument.hpp"
#include "core/domain/InstrumentFactory.hpp"
#include "core/domain/PositionManager.hpp"
#include "core/analytics/RoughVolPricingEngine.hpp"
#include "core/analytics/ImpliedVarianceExtractor.hpp"
#include "core/infrastructure/MarketDataAdapter.hpp"
#include "modules/buyer/BuyerModule.hpp"
#include "modules/seller/DeltaHedger.hpp"

// Demo-local files 
#include "feed/SyntheticOptionFeed.hpp"
#include "execution/SimpleExecSim.hpp"
#include "pnl/AlphaPnLTracker.hpp"
#include "signal/VarianceAlphaSignal.hpp"
#include "signal/VRPAlphaSignal.hpp"
#include "signal/HARRealizedVolSignal.hpp"
#include "signal/SkewCurvatureAlphaSignal.hpp"
#include "signal/CompositeAlphaSignal.hpp"
#include "strategy/StrategyController.hpp"
#include "analytics/RoughVolCalibrator.hpp"
#include "analytics/LiftedHestonStateEstimator.hpp"
#include "analytics/SmileVarianceExtractor.hpp"
#include "strategy/NeuralBSDEHedger.hpp"
#include "feed/HistoricalChainAdapter.hpp"

#ifdef BUILD_ONNX_DEMO
#include "execution/OnnxInference.hpp"
#endif

namespace {

struct AlphaRunnerConfig {
    std::string csv;
    std::string start_date;
    std::string end_date;
    std::string artifacts = "artifacts";
    std::string results_csv;
    std::string hedger = "auto";  // auto preserves historical no-arg behavior.
    bool show_help = false;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --csv <path>          Historical SPY chain CSV. Default: demo data panel.\n"
        << "  --start-date <date>   Inclusive replay start date (YYYY-MM-DD).\n"
        << "  --end-date <date>     Inclusive replay end date (YYYY-MM-DD).\n"
        << "  --artifacts <dir>     Model artifact directory. Default: artifacts.\n"
        << "  --results-csv <path>  Write daily PnL CSV for pipeline gating.\n"
        << "  --hedger <mode>       auto, neural, or bs. Default: auto.\n"
        << "  --help                Show this help.\n";
}

bool parse_args(int argc, char** argv, AlphaRunnerConfig& cfg) {
    auto need_value = [&](int i, const std::string& flag) -> bool {
        if (i + 1 >= argc) {
            std::cerr << "[alpha_runner] Missing value for " << flag << "\n";
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            cfg.show_help = true;
            return true;
        } else if (arg == "--csv") {
            if (!need_value(i, arg)) return false;
            cfg.csv = argv[++i];
        } else if (arg == "--start-date") {
            if (!need_value(i, arg)) return false;
            cfg.start_date = argv[++i];
        } else if (arg == "--end-date") {
            if (!need_value(i, arg)) return false;
            cfg.end_date = argv[++i];
        } else if (arg == "--artifacts") {
            if (!need_value(i, arg)) return false;
            cfg.artifacts = argv[++i];
        } else if (arg == "--results-csv") {
            if (!need_value(i, arg)) return false;
            cfg.results_csv = argv[++i];
        } else if (arg == "--hedger") {
            if (!need_value(i, arg)) return false;
            cfg.hedger = argv[++i];
            if (cfg.hedger != "auto" && cfg.hedger != "neural" && cfg.hedger != "bs") {
                std::cerr << "[alpha_runner] --hedger must be auto, neural, or bs\n";
                return false;
            }
        } else {
            std::cerr << "[alpha_runner] Unknown option: " << arg << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    AlphaRunnerConfig cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 2;
    }
    if (cfg.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    std::cout <<  "Variance Alpha Pipeline Demo\n"
              <<  "============================\n\n";

    // ── 数据模式检测 ──────────────────────────────────────────
    // Prefer the multi-day panel (127 days, includes atm_iv + 25Δ columns);
    // fall back to the legacy single-day ATM CSV for backward compat.
    // Paths relative to the demo/ working directory (where alpha_runner is invoked)
    const std::string panel_csv  = "data/spy_chain_panel.csv";
    const std::string legacy_csv = "data/spy_atm_chain.csv";
    const bool explicit_csv = !cfg.csv.empty();
    const std::string real_data_csv = explicit_csv
        ? cfg.csv
        : (std::filesystem::exists(panel_csv) ? panel_csv : legacy_csv);
    if (explicit_csv && !std::filesystem::exists(real_data_csv)) {
        std::cerr << "[alpha_runner] CSV not found: " << real_data_csv << "\n";
        return 2;
    }
    const bool use_real_data = std::filesystem::exists(real_data_csv);
    std::cout << "[数据模式] " << (use_real_data
        ? "真实 OPRA 数据 (" + real_data_csv + ")"
        : "合成数据 (market_data.csv)") << "\n\n";
    if (!cfg.start_date.empty() || !cfg.end_date.empty()) {
        std::cout << "[Replay Window] "
                  << (cfg.start_date.empty() ? "<first>" : cfg.start_date)
                  << " → "
                  << (cfg.end_date.empty() ? "<last>" : cfg.end_date)
                  << "\n\n";
    }

    // ── 事件总线 ─────────────────────────────────────────────
    auto bus = std::make_shared<omm::events::EventBus>();

    // ── 定价引擎（粗糙模型） ──────────────────────────────────
    auto rough_engine = std::make_shared<omm::domain::RoughVolPricingEngine>(
        omm::domain::RoughVolParams{}, 0.05
    );

    // ── 执行模拟器（两种模式都需要） ──────────────────────────
    auto exec_sim = std::make_shared<omm::demo::SimpleExecSim>(bus);
    exec_sim->register_handlers();

    // ── 合成模式：SyntheticOptionFeed 将 MarketDataEvent → OptionMidQuoteEvent
    // ── 真实模式：HistoricalChainAdapter 直接发布两类事件，无需此适配器 ──
    std::shared_ptr<omm::demo::SyntheticOptionFeed> feed;
    std::shared_ptr<omm::demo::HistoricalChainAdapter> chain_adapter;

    if (use_real_data) {
        chain_adapter = std::make_shared<omm::demo::HistoricalChainAdapter>(
            bus, real_data_csv, cfg.start_date, cfg.end_date);
    } else {
        feed = std::make_shared<omm::demo::SyntheticOptionFeed>(bus);
        feed->register_handlers();
        std::cout << "[Demo] SyntheticOptionFeed 已注册\n";
    }
    std::cout << "[Demo] SimpleExecSim 已注册\n";

    // ── 买方核心组件（注入到 BuyerModule） ────────────────────
    auto extractor  = std::make_shared<omm::analytics::ImpliedVarianceExtractor>(bus);

    int win = use_real_data ? 50 : 10;
    omm::buyer::StrategyControllerConfig ctrl_cfg;  // 默认: max_holding=100

    // ── 在线 xi0 校准器（在信号之前注册，确保先更新引擎参数） ─
    auto calibrator = std::make_shared<omm::demo::RoughVolCalibrator>(
        bus, extractor, rough_engine, /*lambda=*/0.15
    );
    calibrator->register_handlers();
    std::cout << "[Demo] RoughVolCalibrator 已注册 (λ=0.15)\n";

    // ── SmileVarianceExtractor（真实数据模式：VIX 方差互换 + SSVI 参数）──
    auto smile_extractor = std::make_shared<omm::demo::SmileVarianceExtractor>(bus);
    smile_extractor->register_handlers();

    // ── Composite signal: VRP (0.50) + HAR-RV (0.30) + SkewCurvature (0.20) ─
    // Inner bus isolates sub-signal events from StrategyController.
    auto signal    = std::make_shared<omm::buyer::CompositeAlphaSignal>(bus);
    auto inner_bus = signal->inner_bus();

    omm::buyer::VRPSignalConfig vrp_cfg;
    vrp_cfg.base.window  = win;  vrp_cfg.base.z_entry = 1.5;  vrp_cfg.base.z_exit = 0.5;
    auto vrp_signal = std::make_shared<omm::buyer::VRPAlphaSignal>(
        inner_bus, extractor, rough_engine, vrp_cfg
    );

    omm::buyer::HARSignalConfig har_cfg;
    har_cfg.base.window  = win;  har_cfg.base.z_entry = 1.5;  har_cfg.base.z_exit = 0.5;
    auto har_signal = std::make_shared<omm::buyer::HARRealizedVolSignal>(
        inner_bus, extractor, rough_engine, har_cfg
    );

    omm::buyer::SkewCurvatureSignalConfig skew_cfg;
    skew_cfg.base.window       = win;  skew_cfg.base.z_entry = 1.5;  skew_cfg.base.z_exit = 0.5;
    skew_cfg.var_weight        = 0.70;
    skew_cfg.curvature_weight  = 0.30;
    auto skew_signal = std::make_shared<omm::buyer::SkewCurvatureAlphaSignal>(
        inner_bus, extractor, rough_engine, skew_cfg
    );

    // 注入 SmileVarianceExtractor：真实数据模式使用 VIX / SSVI；合成模式自动降级
    vrp_signal->set_smile_extractor(smile_extractor);
    skew_signal->set_smile_extractor(smile_extractor);

    signal->add(vrp_signal,  0.50);
    signal->add(har_signal,  0.30);
    signal->add(skew_signal, 0.20);

    auto controller = std::make_shared<omm::buyer::StrategyController>(
        bus, vrp_cfg.base, ctrl_cfg
    );
    controller->register_handlers();

    // Install BuyerModule first (registers extractor), then composite forwarder
    omm::buyer::BuyerModule::install(bus, extractor, signal, controller);
    signal->register_handlers();
    std::cout << "[Demo] CompositeAlphaSignal 已注册 (VRP 50% + HAR 30% + SkewCurv 20%)\n\n";

    // ── 对冲引擎（DeltaHedger 或 NeuralBSDEHedger） ─────────
    // 真实数据：从 CSV 首行读取行权价和到期日；合成数据：使用固定值
    auto expiry = use_real_data
        ? chain_adapter->initial_expiry()
        : std::chrono::system_clock::now() + std::chrono::hours(30 * 24);
    double STRIKE_ENTRY = use_real_data ? chain_adapter->initial_strike() : 150.0;
    const std::string  CALL_ID      = "ATM_CALL";
    const std::string  PUT_ID       = "ATM_PUT";

    auto position_mgr = std::make_shared<omm::domain::PositionManager>();

    // 将所有对冲组件声明在外层作用域，确保生命周期覆盖 adapter.run()
#ifdef BUILD_ONNX_DEMO
    const std::filesystem::path artifacts_dir = cfg.artifacts;
    const std::string onnx_path = (artifacts_dir / "neural_bsde.onnx").string();
    const std::string norm_path = (artifacts_dir / "normalization.json").string();
    const bool neural_artifacts_available = std::filesystem::exists(onnx_path) &&
                                            std::filesystem::exists(norm_path);
    if (cfg.hedger == "neural" && !neural_artifacts_available) {
        std::cerr << "[alpha_runner] Neural hedger requested but artifacts are missing:\n"
                  << "  " << onnx_path << "\n"
                  << "  " << norm_path << "\n";
        return 2;
    }
    const bool use_neural = (cfg.hedger != "bs") && neural_artifacts_available;
    std::shared_ptr<::demo::OnnxInference>          onnx_model;
    std::shared_ptr<omm::demo::NeuralBSDEHedger>    neural_hedger;
#else
    if (cfg.hedger == "neural") {
        std::cerr << "[alpha_runner] Neural hedger requested, but this binary was "
                  << "built without BUILD_ONNX_DEMO.\n";
        return 2;
    }
    const bool use_neural = false;
#endif
    std::shared_ptr<omm::application::DeltaHedger>  delta_hedger;

    if (use_neural) {
#ifdef BUILD_ONNX_DEMO
        // LRH 状态估计器（normalization.json 中的 model_params 提供默认值）
        // SPY-calibrated params from normalization.json (retrained on spy_chain_panel.csv)
        auto state_est = std::make_shared<omm::demo::LiftedHestonStateEstimator>(
            /*kappa=*/0.3, /*theta=*/0.01386, /*xi=*/0.5, /*V0=*/0.01386
        );
        onnx_model    = std::make_shared<::demo::OnnxInference>(onnx_path, norm_path);
        neural_hedger = std::make_shared<omm::demo::NeuralBSDEHedger>(
            bus, onnx_model, rough_engine, position_mgr, state_est,
            "AAPL", CALL_ID, PUT_ID,
            STRIKE_ENTRY, /*threshold=*/10.0
        );
        neural_hedger->register_handlers();
        std::cout << "[Hedger] NeuralBSDEHedger 已激活（BSDE + 在线 OU 状态，阈值=10.0 股）\n";
#endif
    } else {
        auto call_atm = omm::domain::InstrumentFactory::make_call("AAPL", STRIKE_ENTRY, expiry);
        auto put_atm  = omm::domain::InstrumentFactory::make_put ("AAPL", STRIKE_ENTRY, expiry);
        // Use OptionEntry pairs so delta_map keys match PositionManager fill IDs
        std::vector<omm::application::DeltaHedger::OptionEntry> hedge_options = {
            {"ATM_CALL", call_atm},
            {"ATM_PUT",  put_atm}
        };

        delta_hedger = std::make_shared<omm::application::DeltaHedger>(
            bus, position_mgr, rough_engine, hedge_options, "AAPL", /*threshold=*/10.0
        );
        delta_hedger->register_handlers();
        std::cout << "[Hedger] DeltaHedger 已激活（解析 BS Delta，阈值=10.0 股）\n";
        std::cout << "[Hedger] DeltaHedger 将在真实数据回放时通过 set_market_state() 使用市场 IV 和 T_sim\n";
    }

    // ── PnL 追踪（行为组件，注入引擎 + 期权列表 + extractor） ─
    // 使用与 StrategyController 相同的 fill ID ("ATM_CALL" / "ATM_PUT")
    // 以确保 on_market_data Greeks 循环能正确找到对应持仓
    auto call_pnl = omm::domain::InstrumentFactory::make_call("AAPL", STRIKE_ENTRY, expiry);
    auto put_pnl  = omm::domain::InstrumentFactory::make_put ("AAPL", STRIKE_ENTRY, expiry);
    std::vector<omm::demo::AlphaPnLTracker::OptionEntry> pnl_options = {
        {CALL_ID, call_pnl},   // "ATM_CALL" → call option for repricing
        {PUT_ID,  put_pnl}     // "ATM_PUT"  → put option for repricing
    };

    auto pnl_tracker = std::make_shared<omm::demo::AlphaPnLTracker>(
        bus, rough_engine, pnl_options, extractor
    );
    pnl_tracker->register_handlers();
    std::cout << "[Demo] AlphaPnLTracker 已注册（Greeks 归因）\n\n";

    // ── 事件流说明 ────────────────────────────────────────────
    std::cout << "  [事件流]\n"
              << "  MarketDataEvent\n"
              << "    → SyntheticOptionFeed      → OptionMidQuoteEvent\n"
              << "    → ImpliedVarianceExtractor (σ²_atm)\n"
              << "    → CompositeAlphaSignal     → SignalSnapshotEvent (zscore)\n"
              << "       (VRP 50% + HAR-RV 30% + SkewCurvature 20%)\n"
              << "    → StrategyController       → OrderSubmittedEvent (straddle)\n"
              << "    → SimpleExecSim            → FillEvent\n"
              << "    → " << (use_neural ? "NeuralBSDEHedger" : "DeltaHedger      ")
              <<                            " → OrderSubmittedEvent (hedge)\n"
              << "    → AlphaPnLTracker          (option_mtm + delta_hedge_pnl)\n\n";

    // ── 运行仿真 ──────────────────────────────────────────────
    std::cout << "══════════════ Alpha Pipeline Start ══════════════\n\n";
    if (use_real_data) {
        // on_row callback: fires after each bar's events are published.
        // 1. Inject market IV + T_sim into DeltaHedger for correct historical delta.
        // 2. Detect session (date) boundaries for per-day PnL accounting.
        std::string prev_date;
        chain_adapter->run([&](double atm_iv, double T_sim, const std::string& date) {
            if (delta_hedger && atm_iv > 0.0 && T_sim > 0.0)
                delta_hedger->set_market_state(atm_iv, T_sim);
#ifdef BUILD_ONNX_DEMO
            if (neural_hedger && T_sim > 0.0)
                neural_hedger->set_market_state(atm_iv, T_sim);
#endif
            if (!prev_date.empty() && date != prev_date)
                pnl_tracker->on_session_end(prev_date);
            prev_date = date;
        });
        exec_sim->flush_all();
        // Flush the last day
        if (!prev_date.empty())
            pnl_tracker->on_session_end(prev_date);
    } else {
        omm::infrastructure::MarketDataAdapter adapter(bus, "../data/market_data.csv");
        adapter.run();
        exec_sim->flush_all();
    }
    std::cout << "\n══════════════ Alpha Pipeline End ══════════════\n";

    // ── 最终持仓 + PnL 归因 ───────────────────────────────────
    position_mgr->print_positions();
    pnl_tracker->print_summary();
    if (!cfg.results_csv.empty())
        pnl_tracker->write_daily_csv(cfg.results_csv);

    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║  Complete.                                               ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
