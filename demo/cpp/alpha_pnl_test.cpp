// ============================================================
// File: alpha_pnl_test.cpp  (demo/cpp/)
// Role: Alpha PnL test with rough-vol smile signals and
//       dual-hedger comparison (DeltaHedger vs NeuralBSDEHedger).
//
// Pipeline (per pass):
//   MarketDataAdapter / HistoricalChainAdapter
//     → SyntheticOptionFeed / chain replay  → OptionMidQuoteEvent
//     → ImpliedVarianceExtractor            → ImpliedVariancePoint
//     → RoughVolCalibrator (EWMA xi0)
//     → SkewCurvatureAlphaSignal            → SignalSnapshotEvent
//         (variance z-score 0.7 + curvature z-score 0.3)
//     → StrategyController                  → OrderSubmittedEvent
//     → SimpleExecSim                       → FillEvent
//     → DeltaHedger | NeuralBSDEHedger      → OrderSubmittedEvent (hedge)
//     → ExtendedPnLTracker                  → ExtendedPnLBreakdown
//
// Two sequential passes over the same data — identical signal,
// different hedger — then a side-by-side comparison table.
//
// Rough Heston calibration params (AAPL 2026-04-08, best fit):
//   H=0.01, ν=0.838, ρ=−0.507, v0=0.077
//   Skew scales T^{−0.49}, curvature T^{−0.98} for short-dated
// ============================================================

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <chrono>
#include <filesystem>
#include <functional>

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
#include "signal/SkewCurvatureAlphaSignal.hpp"
#include "strategy/StrategyController.hpp"
#include "analytics/RoughVolCalibrator.hpp"
#include "pnl/ExtendedPnLTracker.hpp"
#include "feed/HistoricalChainAdapter.hpp"
#include "analytics/LiftedHestonStateEstimator.hpp"
#include "strategy/NeuralBSDEHedger.hpp"

#ifdef BUILD_ONNX_DEMO
#include "execution/OnnxInference.hpp"
#endif

// ── Calibrated Rough Heston params (AAPL gate result 2026-04-08) ──
static omm::domain::RoughVolParams ROUGH_HESTON_PARAMS = {
    .H   = 0.01,    // Hurst exponent — extremely rough, T^{−0.49} skew scaling
    .eta = 0.838,   // ν in Rough Heston ≈ η in Rough Bergomi (vol-of-vol)
    .rho = -0.507,  // spot-vol correlation (leverage effect)
    .xi0 = 0.077    // initial forward variance (v0 from calibration, EWMA-updated online)
};

static const std::string REAL_DATA_CSV  = "../demo/data/spy_atm_chain.csv";
static const std::string SYNTH_DATA_CSV = "../data/market_data.csv";
static const std::string ONNX_PATH      = "artifacts/neural_bsde.onnx";
static const std::string NORM_PATH      = "artifacts/normalization.json";

// ── Single simulation pass ────────────────────────────────────
// Returns ExtendedPnLBreakdown for the given hedger configuration.
// All state is fresh per call — safe to call twice with same CSV.
static omm::demo::ExtendedPnLBreakdown run_simulation(
    bool        use_neural,
    const std::string& label,
    bool        use_real_data
) {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  Pass: " << std::left << std::setw(50) << label << "║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ── Event bus ─────────────────────────────────────────────
    auto bus = std::make_shared<omm::events::EventBus>();

    // ── Pricing engine (calibrated Rough Heston params) ──────
    auto rough_engine = std::make_shared<omm::domain::RoughVolPricingEngine>(
        ROUGH_HESTON_PARAMS, 0.05
    );

    // ── Exec sim ──────────────────────────────────────────────
    auto exec_sim = std::make_shared<omm::demo::SimpleExecSim>(bus);
    exec_sim->register_handlers();

    // ── Data feed ─────────────────────────────────────────────
    std::shared_ptr<omm::demo::SyntheticOptionFeed>    feed;
    std::shared_ptr<omm::demo::HistoricalChainAdapter> chain_adapter;

    if (use_real_data) {
        chain_adapter = std::make_shared<omm::demo::HistoricalChainAdapter>(bus, REAL_DATA_CSV);
    } else {
        feed = std::make_shared<omm::demo::SyntheticOptionFeed>(bus);
        feed->register_handlers();
    }

    // ── IV extractor ──────────────────────────────────────────
    auto extractor = std::make_shared<omm::analytics::ImpliedVarianceExtractor>(bus);

    // ── Online xi0 calibrator (registered before signal) ─────
    auto calibrator = std::make_shared<omm::demo::RoughVolCalibrator>(
        bus, extractor, rough_engine, /*lambda=*/0.15
    );
    calibrator->register_handlers();

    // ── Skew+Curvature alpha signal ───────────────────────────
    omm::buyer::SkewCurvatureSignalConfig signal_cfg;
    signal_cfg.base.window   = use_real_data ? 50 : 10;
    signal_cfg.base.z_entry  = 1.5;
    signal_cfg.base.z_exit   = 0.5;
    signal_cfg.var_weight    = 0.70;
    signal_cfg.curvature_weight = 0.30;

    auto signal = std::make_shared<omm::buyer::SkewCurvatureAlphaSignal>(
        bus, extractor, rough_engine, signal_cfg
    );
    signal->register_handlers();

    // ── Strategy controller ───────────────────────────────────
    omm::buyer::StrategyControllerConfig ctrl_cfg;
    auto controller = std::make_shared<omm::buyer::StrategyController>(
        bus, signal_cfg.base, ctrl_cfg
    );
    controller->register_handlers();

    omm::buyer::BuyerModule::install(bus, extractor, signal, controller);

    // ── Determine entry strike/expiry ────────────────────────
    auto expiry = use_real_data
        ? chain_adapter->initial_expiry()
        : std::chrono::system_clock::now() + std::chrono::hours(30 * 24);
    double strike = use_real_data ? chain_adapter->initial_strike() : 150.0;

    const std::string CALL_ID = "ATM_CALL";
    const std::string PUT_ID  = "ATM_PUT";

    auto position_mgr = std::make_shared<omm::domain::PositionManager>();

    // ── Hedger ────────────────────────────────────────────────
#ifdef BUILD_ONNX_DEMO
    const bool neural_available = use_neural
        && std::filesystem::exists(ONNX_PATH)
        && std::filesystem::exists(NORM_PATH);
    std::shared_ptr<::demo::OnnxInference>       onnx_model;
    std::shared_ptr<omm::demo::NeuralBSDEHedger> neural_hedger;
#else
    const bool neural_available = false;
#endif
    std::shared_ptr<omm::application::DeltaHedger> delta_hedger;

    if (neural_available) {
#ifdef BUILD_ONNX_DEMO
        auto state_est = std::make_shared<omm::demo::LiftedHestonStateEstimator>(
            /*kappa=*/0.3, /*theta=*/0.04, /*xi=*/0.5, /*V0=*/ROUGH_HESTON_PARAMS.xi0
        );
        onnx_model    = std::make_shared<::demo::OnnxInference>(ONNX_PATH, NORM_PATH);
        neural_hedger = std::make_shared<omm::demo::NeuralBSDEHedger>(
            bus, onnx_model, rough_engine, position_mgr, state_est,
            "AAPL", CALL_ID, PUT_ID, strike, expiry, /*threshold=*/0.3
        );
        neural_hedger->register_handlers();
        std::cout << "[Hedger] NeuralBSDEHedger active (BSDE + Markovian lift state)\n";
#endif
    } else {
        auto call_atm = omm::domain::InstrumentFactory::make_call("AAPL", strike, expiry);
        auto put_atm  = omm::domain::InstrumentFactory::make_put ("AAPL", strike, expiry);
        std::vector<std::shared_ptr<omm::domain::Option>> opts = {call_atm, put_atm};
        delta_hedger = std::make_shared<omm::application::DeltaHedger>(
            bus, position_mgr, rough_engine, opts, "AAPL", /*threshold=*/0.3
        );
        delta_hedger->register_handlers();
        bus->subscribe<omm::events::MarketDataEvent>(
            [delta_hedger](const omm::events::MarketDataEvent& e) {
                delta_hedger->update_market_price(e.underlying_price);
            }
        );
        if (use_neural)
            std::cout << "[Hedger] NeuralBSDEHedger unavailable (no ONNX model), "
                         "falling back to DeltaHedger\n";
        else
            std::cout << "[Hedger] DeltaHedger active (analytical BS delta)\n";
    }

    // ── Extended PnL tracker ──────────────────────────────────
    auto call_pnl = omm::domain::InstrumentFactory::make_call("AAPL", strike, expiry);
    auto put_pnl  = omm::domain::InstrumentFactory::make_put ("AAPL", strike, expiry);
    std::vector<omm::demo::ExtendedPnLTracker::OptionEntry> pnl_opts = {
        {CALL_ID, call_pnl},
        {PUT_ID,  put_pnl}
    };

    auto tracker = std::make_shared<omm::demo::ExtendedPnLTracker>(
        bus, rough_engine, pnl_opts, extractor,
        /*hedger_label=*/neural_available ? "NeuralBSDEHedger" : "DeltaHedger",
        /*rate=*/0.05
    );
    tracker->register_handlers();

    // ── Print smile predictions at current params ─────────────
    {
        omm::demo::RoughVolSmilePredictor pred;
        for (double T : {7.0/365.0, 30.0/365.0, 60.0/365.0}) {
            auto s = pred.predict(ROUGH_HESTON_PARAMS, T);
            std::cout << std::fixed << std::setprecision(4)
                      << "  [SmilePredict] T=" << std::setw(6) << T
                      << "  σ₀=" << s.atm_vol
                      << "  skew=" << s.skew_slope
                      << "  curv=" << s.curvature << "\n";
        }
        std::cout << "\n";
    }

    // ── Run simulation ────────────────────────────────────────
    std::cout << "── Simulation Start ─────────────────────────────────────\n";
    if (use_real_data) {
        chain_adapter->run();
    } else {
        omm::infrastructure::MarketDataAdapter adapter(bus, SYNTH_DATA_CSV);
        adapter.run();
    }
    std::cout << "── Simulation End ───────────────────────────────────────\n";

    position_mgr->print_positions();
    return tracker->breakdown();
}

// ─────────────────────────────────────────────────────────────
int main() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║  Alpha PnL Test — Rough Vol Smile + Dual Hedger          ║\n"
              << "╠══════════════════════════════════════════════════════════╣\n"
              << "║  Signal: variance z-score (0.70) + curvature z-score     ║\n"
              << "║          (0.30) from Rough Heston calibration            ║\n"
              << "║  Params: H=0.01  η=0.838  ρ=−0.507  v0=0.077            ║\n"
              << "║  Skew scaling: T^{H−0.5} = T^{−0.49}                    ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    const bool use_real_data = std::filesystem::exists(REAL_DATA_CSV);
    std::cout << "[Data] " << (use_real_data
        ? "Real OPRA chain: " + REAL_DATA_CSV
        : "Synthetic data:  " + SYNTH_DATA_CSV) << "\n\n";

    // ── Pass 1: DeltaHedger ───────────────────────────────────
    auto delta_result = run_simulation(/*use_neural=*/false, "DeltaHedger", use_real_data);

    // ── Pass 2: NeuralBSDEHedger (or fallback) ────────────────
    auto bsde_result  = run_simulation(/*use_neural=*/true,  "NeuralBSDEHedger", use_real_data);

    // ── Comparison table ──────────────────────────────────────
    omm::demo::ExtendedPnLTracker::print_comparison(delta_result, bsde_result);

    // ── Signal decomposition summary ─────────────────────────
    std::cout << "  Signal Decomposition (Rough Heston gate results)\n"
              << "  ─────────────────────────────────────────────────────────\n"
              << std::fixed << std::setprecision(4)
              << "  Variance signal weight:   70%\n"
              << "  Curvature signal weight:  30%\n"
              << "  Model H (roughness):      0.01  → T^{−0.49} skew scaling\n"
              << "  Model η (vol-of-vol):     0.838 → high curvature\n"
              << "  Model ρ (leverage):       −0.507→ negative skew\n\n";

    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║  Complete. Build with -DBUILD_ONNX_DEMO=ON to activate   ║\n"
              << "║  NeuralBSDEHedger for the path-dependent BSDE pass.      ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
