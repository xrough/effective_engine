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
#include "signal/VRPAlphaSignal.hpp"
#include "signal/HARRealizedVolSignal.hpp"
#include "signal/CompositeAlphaSignal.hpp"
#include "signal/DownsideVRPSignal.hpp"
#include "signal/LogIVRVRatioSignal.hpp"
#include "signal/RealizedSkewnessSignal.hpp"
#include "signal/IVAutocorrSignal.hpp"
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

// Prefer multi-day panel; fall back to legacy single-day ATM CSV
static const std::string PANEL_CSV      = "../data/spy_chain_panel.csv";
static const std::string LEGACY_CSV     = "../data/spy_atm_chain.csv";
static const std::string REAL_DATA_CSV  =
    std::filesystem::exists(PANEL_CSV) ? PANEL_CSV : LEGACY_CSV;
static const std::string SYNTH_DATA_CSV = "../data/market_data.csv";
static const std::string ONNX_PATH      = "artifacts/neural_bsde.onnx";
static const std::string NORM_PATH      = "artifacts/normalization.json";
// IS-trained model artifacts (produced by extract_real_paths.py + generate_is_synthetic.py + trainer.py)
static const std::string ONNX_IS_PATH       = "artifacts/neural_bsde_is.onnx";
static const std::string NORM_IS_PATH       = "artifacts/normalization_is.json";
// Delta-only BSDE (partial hedge, preserves VRP alpha)
static const std::string ONNX_IS_DELTA_PATH = "artifacts/neural_bsde_is_delta.onnx";
// NORM_IS_PATH is reused for the delta model (same normalization stats)
// Calendar split: IS = "" → SPLIT_DATE (exclusive),  OOS = SPLIT_DATE → ""
static const std::string SPLIT_DATE         = "2026-01-01";

// HedgerMode — which hedge strategy to use in this simulation pass.
// NEURAL_BSDE_IS       uses the IS-trained full-replication model.
// NEURAL_BSDE_IS_DELTA uses the IS-trained delta-only model (preserves VRP).
enum class HedgerMode { BS_DELTA, ROUGH_DELTA, NEURAL_BSDE, NEURAL_BSDE_IS, NEURAL_BSDE_IS_DELTA };

// ── Single simulation pass ────────────────────────────────────
// Returns ExtendedPnLBreakdown for the given hedger configuration.
// All state is fresh per call — safe to call three times with same CSV.
static omm::demo::ExtendedPnLBreakdown run_simulation(
    HedgerMode         mode,
    const std::string& label,
    bool               use_real_data,
    const std::string& start_date = "",  // "YYYY-MM-DD" inclusive, "" = no filter
    const std::string& end_date   = ""   // "YYYY-MM-DD" inclusive, "" = no filter
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
        chain_adapter = std::make_shared<omm::demo::HistoricalChainAdapter>(
            bus, REAL_DATA_CSV, start_date, end_date);
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

    // ── Alpha signals: VRP + HAR-RV + SkewCurvature composite ──
    int win = use_real_data ? 50 : 10;

    // Composite: inner bus isolates sub-signal events from StrategyController
    auto composite = std::make_shared<omm::buyer::CompositeAlphaSignal>(bus);
    auto inner_bus = composite->inner_bus();

    // Sub-signal 1: VRP (IV² − RV5²), weight 0.50 — constructed with inner bus
    omm::buyer::VRPSignalConfig vrp_cfg;
    vrp_cfg.base.window  = win;
    vrp_cfg.base.z_entry = 1.5;
    vrp_cfg.base.z_exit  = 0.5;
    auto vrp_signal = std::make_shared<omm::buyer::VRPAlphaSignal>(
        inner_bus, extractor, rough_engine, vrp_cfg
    );

    // Sub-signal 2: HAR-RV forecast vs IV, weight 0.30 — constructed with inner bus
    omm::buyer::HARSignalConfig har_cfg;
    har_cfg.base.window  = win;
    har_cfg.base.z_entry = 1.5;
    har_cfg.base.z_exit  = 0.5;
    auto har_signal = std::make_shared<omm::buyer::HARRealizedVolSignal>(
        inner_bus, extractor, rough_engine, har_cfg
    );

    // Sub-signal 3: Rough-model variance + curvature z-score, weight 0.20
    omm::buyer::SkewCurvatureSignalConfig skew_cfg;
    skew_cfg.base.window        = win;
    skew_cfg.base.z_entry       = 1.5;
    skew_cfg.base.z_exit        = 0.5;
    skew_cfg.var_weight         = 0.70;
    skew_cfg.curvature_weight   = 0.30;
    auto skew_signal = std::make_shared<omm::buyer::SkewCurvatureAlphaSignal>(
        inner_bus, extractor, rough_engine, skew_cfg
    );

    composite->add(vrp_signal,  0.50);
    composite->add(har_signal,  0.30);
    composite->add(skew_signal, 0.20);

    // ── Strategy controller ───────────────────────────────────
    omm::buyer::StrategyControllerConfig ctrl_cfg;
    auto controller = std::make_shared<omm::buyer::StrategyController>(
        bus, vrp_cfg.base, ctrl_cfg
    );
    controller->register_handlers();

    // Install BuyerModule FIRST so extractor subscribes to OptionMidQuoteEvent
    // before composite's inner-bus forwarder does — guarantees extractor.last_point()
    // is fresh when sub-signals call it.
    omm::buyer::BuyerModule::install(bus, extractor, composite, controller);

    // Register composite AFTER extractor is wired (inner-bus forwarding order correct)
    composite->register_handlers();

    // ── Determine entry strike/expiry ────────────────────────
    auto expiry = use_real_data
        ? chain_adapter->initial_expiry()
        : std::chrono::system_clock::now() + std::chrono::hours(30 * 24);
    double strike = use_real_data ? chain_adapter->initial_strike() : 150.0;

    const std::string CALL_ID = "ATM_CALL";
    const std::string PUT_ID  = "ATM_PUT";

    auto position_mgr = std::make_shared<omm::domain::PositionManager>();

    // ── Option instruments (needed by both delta hedger and PnL tracker) ──
    auto call_atm = omm::domain::InstrumentFactory::make_call("AAPL", strike, expiry);
    auto put_atm  = omm::domain::InstrumentFactory::make_put ("AAPL", strike, expiry);

    // ── Hedger ────────────────────────────────────────────────
    // NEURAL_BSDE uses the synthetic-trained model; NEURAL_BSDE_IS uses the IS full-replication model;
    // NEURAL_BSDE_IS_DELTA uses the IS delta-only model (preserves VRP alpha).
#ifdef BUILD_ONNX_DEMO
    const bool is_neural_mode = (mode == HedgerMode::NEURAL_BSDE ||
                                 mode == HedgerMode::NEURAL_BSDE_IS ||
                                 mode == HedgerMode::NEURAL_BSDE_IS_DELTA);
    const std::string& sel_onnx =
        (mode == HedgerMode::NEURAL_BSDE_IS_DELTA) ? ONNX_IS_DELTA_PATH :
        (mode == HedgerMode::NEURAL_BSDE_IS)        ? ONNX_IS_PATH       :
        ONNX_PATH;
    const std::string& sel_norm =
        (mode == HedgerMode::NEURAL_BSDE_IS || mode == HedgerMode::NEURAL_BSDE_IS_DELTA)
        ? NORM_IS_PATH : NORM_PATH;
    const bool neural_available = is_neural_mode
        && std::filesystem::exists(sel_onnx)
        && std::filesystem::exists(sel_norm);
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
        onnx_model    = std::make_shared<::demo::OnnxInference>(sel_onnx, sel_norm);
        neural_hedger = std::make_shared<omm::demo::NeuralBSDEHedger>(
            bus, onnx_model, rough_engine, position_mgr, state_est,
            "AAPL", CALL_ID, PUT_ID, strike, /*threshold=*/0.3
        );
        neural_hedger->register_handlers();
        std::cout << "[Hedger] NeuralBSDEHedger active ("
                  << (mode == HedgerMode::NEURAL_BSDE_IS_DELTA ? "IS-delta, partial hedge"
                    : mode == HedgerMode::NEURAL_BSDE_IS       ? "IS-trained, full replication"
                    : "synth-trained")
                  << " BSDE + Markovian lift state)\n";
#endif
    } else if (mode == HedgerMode::ROUGH_DELTA) {
        std::vector<omm::application::DeltaHedger::OptionEntry> opts = {
            {"ATM_CALL", call_atm},
            {"ATM_PUT",  put_atm}
        };
        delta_hedger = std::make_shared<omm::application::DeltaHedger>(
            bus, position_mgr, rough_engine, opts, "AAPL", /*threshold=*/0.3,
            omm::application::HedgeMode::ROUGH_DELTA
        );
        delta_hedger->register_handlers();
        std::cout << "[Hedger] RoughVolDelta active (Bergomi-Guyon ∂σ/∂S correction)\n";
    } else {
        std::vector<omm::application::DeltaHedger::OptionEntry> opts = {
            {"ATM_CALL", call_atm},
            {"ATM_PUT",  put_atm}
        };
        delta_hedger = std::make_shared<omm::application::DeltaHedger>(
            bus, position_mgr, rough_engine, opts, "AAPL", /*threshold=*/0.3,
            omm::application::HedgeMode::BS_DELTA
        );
        delta_hedger->register_handlers();
        if (mode == HedgerMode::NEURAL_BSDE ||
            mode == HedgerMode::NEURAL_BSDE_IS ||
            mode == HedgerMode::NEURAL_BSDE_IS_DELTA)
            std::cout << "[Hedger] NeuralBSDEHedger unavailable (no ONNX model — "
                      << "build with -DBUILD_ONNX_DEMO=ON and run training pipeline), "
                      << "falling back to BSDelta\n";
        else
            std::cout << "[Hedger] BSDelta active (plain N(d1) at market IV)\n";
    }

    // ── Extended PnL tracker ──────────────────────────────────
    auto call_pnl = omm::domain::InstrumentFactory::make_call("AAPL", strike, expiry);
    auto put_pnl  = omm::domain::InstrumentFactory::make_put ("AAPL", strike, expiry);
    std::vector<omm::demo::ExtendedPnLTracker::OptionEntry> pnl_opts = {
        {CALL_ID, call_pnl},
        {PUT_ID,  put_pnl}
    };

    std::string tracker_label;
    if      (neural_available)             tracker_label = "NeuralBSDEHedger";
    else if (mode == HedgerMode::ROUGH_DELTA) tracker_label = "RoughVolDelta";
    else                                   tracker_label = "BSDelta";

    auto tracker = std::make_shared<omm::demo::ExtendedPnLTracker>(
        bus, rough_engine, pnl_opts, extractor,
        tracker_label, /*rate=*/0.05
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
        chain_adapter->run([&](double atm_iv, double T_sim, const std::string& /*date*/) {
            if (delta_hedger && atm_iv > 0.0 && T_sim > 0.0)
                delta_hedger->set_market_state(atm_iv, T_sim);
#ifdef BUILD_ONNX_DEMO
            if (neural_hedger && T_sim > 0.0)
                neural_hedger->set_market_state(atm_iv, T_sim);
#endif
        });
    } else {
        omm::infrastructure::MarketDataAdapter adapter(bus, SYNTH_DATA_CSV);
        adapter.run();
    }
    std::cout << "── Simulation End ───────────────────────────────────────\n";

    position_mgr->print_positions();
    return tracker->breakdown();
}

// ─────────────────────────────────────────────────────────────
// Pass 4: 2020s vol alpha composite (equal-weight, BS Delta only).
// Signals: DownsideVRP + LogIVRVRatio + RealizedSkewness + IVAutocorr
// This function is structurally identical to run_simulation() — only
// the signal block differs. No changes to hedger or PnL tracker.
// ─────────────────────────────────────────────────────────────
static omm::demo::ExtendedPnLBreakdown run_simulation_modern(
    HedgerMode         mode,
    const std::string& label,
    bool               use_real_data,
    const std::string& start_date = "",  // "YYYY-MM-DD" inclusive, "" = no filter
    const std::string& end_date   = ""   // "YYYY-MM-DD" inclusive, "" = no filter
) {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  Pass: " << std::left << std::setw(50) << label << "║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    auto bus = std::make_shared<omm::events::EventBus>();
    auto rough_engine = std::make_shared<omm::domain::RoughVolPricingEngine>(
        ROUGH_HESTON_PARAMS, 0.05
    );
    auto exec_sim = std::make_shared<omm::demo::SimpleExecSim>(bus);
    exec_sim->register_handlers();

    std::shared_ptr<omm::demo::SyntheticOptionFeed>    feed;
    std::shared_ptr<omm::demo::HistoricalChainAdapter> chain_adapter;
    if (use_real_data) {
        chain_adapter = std::make_shared<omm::demo::HistoricalChainAdapter>(
            bus, REAL_DATA_CSV, start_date, end_date);
    } else {
        feed = std::make_shared<omm::demo::SyntheticOptionFeed>(bus);
        feed->register_handlers();
    }

    auto extractor = std::make_shared<omm::analytics::ImpliedVarianceExtractor>(bus);
    auto calibrator = std::make_shared<omm::demo::RoughVolCalibrator>(
        bus, extractor, rough_engine, /*lambda=*/0.15
    );
    calibrator->register_handlers();

    int win = use_real_data ? 50 : 10;

    // ── 2020s composite: inner bus isolates sub-signal events ────
    auto signal    = std::make_shared<omm::buyer::CompositeAlphaSignal>(bus);
    auto inner_bus = signal->inner_bus();

    omm::buyer::DownsideVRPSignalConfig dvp_cfg;
    dvp_cfg.base.window = win;  dvp_cfg.base.z_entry = 1.5;  dvp_cfg.base.z_exit = 0.5;
    auto dvp_signal = std::make_shared<omm::buyer::DownsideVRPSignal>(
        inner_bus, extractor, rough_engine, dvp_cfg
    );

    omm::buyer::LogIVRVRatioSignalConfig livrv_cfg;
    livrv_cfg.base.window = win;  livrv_cfg.base.z_entry = 1.5;  livrv_cfg.base.z_exit = 0.5;
    auto livrv_signal = std::make_shared<omm::buyer::LogIVRVRatioSignal>(
        inner_bus, extractor, rough_engine, livrv_cfg
    );

    omm::buyer::RealizedSkewnessSignalConfig rsk_cfg;
    rsk_cfg.base.window = win;  rsk_cfg.base.z_entry = 1.5;  rsk_cfg.base.z_exit = 0.5;
    auto rsk_signal = std::make_shared<omm::buyer::RealizedSkewnessSignal>(
        inner_bus, extractor, rough_engine, rsk_cfg
    );

    omm::buyer::IVAutocorrSignalConfig ivac_cfg;
    ivac_cfg.base.window = win;  ivac_cfg.base.z_entry = 1.5;  ivac_cfg.base.z_exit = 0.5;
    auto ivac_signal = std::make_shared<omm::buyer::IVAutocorrSignal>(
        inner_bus, extractor, rough_engine, ivac_cfg
    );

    signal->add(dvp_signal,   0.25);
    signal->add(livrv_signal, 0.25);
    signal->add(rsk_signal,   0.25);
    signal->add(ivac_signal,  0.25);
    // ──────────────────────────────────────────────────────────

    omm::buyer::StrategyControllerConfig ctrl_cfg;
    auto controller = std::make_shared<omm::buyer::StrategyController>(
        bus, dvp_cfg.base, ctrl_cfg
    );
    controller->register_handlers();

    // Install extractor FIRST, then register composite forwarder (ordering matters)
    omm::buyer::BuyerModule::install(bus, extractor, signal, controller);
    signal->register_handlers();

    auto expiry = use_real_data
        ? chain_adapter->initial_expiry()
        : std::chrono::system_clock::now() + std::chrono::hours(30 * 24);
    double strike = use_real_data ? chain_adapter->initial_strike() : 150.0;

    const std::string CALL_ID = "ATM_CALL";
    const std::string PUT_ID  = "ATM_PUT";

    auto position_mgr = std::make_shared<omm::domain::PositionManager>();
    auto call_atm = omm::domain::InstrumentFactory::make_call("AAPL", strike, expiry);
    auto put_atm  = omm::domain::InstrumentFactory::make_put ("AAPL", strike, expiry);

#ifdef BUILD_ONNX_DEMO
    const bool neural_available = (mode == HedgerMode::NEURAL_BSDE)
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
            "AAPL", CALL_ID, PUT_ID, strike, /*threshold=*/0.3
        );
        neural_hedger->register_handlers();
#endif
    } else if (mode == HedgerMode::ROUGH_DELTA) {
        std::vector<omm::application::DeltaHedger::OptionEntry> opts = {
            {"ATM_CALL", call_atm}, {"ATM_PUT", put_atm}
        };
        delta_hedger = std::make_shared<omm::application::DeltaHedger>(
            bus, position_mgr, rough_engine, opts, "AAPL", /*threshold=*/0.3,
            omm::application::HedgeMode::ROUGH_DELTA
        );
        delta_hedger->register_handlers();
        std::cout << "[Hedger] RoughVolDelta active\n";
    } else {
        std::vector<omm::application::DeltaHedger::OptionEntry> opts = {
            {"ATM_CALL", call_atm}, {"ATM_PUT", put_atm}
        };
        delta_hedger = std::make_shared<omm::application::DeltaHedger>(
            bus, position_mgr, rough_engine, opts, "AAPL", /*threshold=*/0.3,
            omm::application::HedgeMode::BS_DELTA
        );
        delta_hedger->register_handlers();
        std::cout << "[Hedger] BSDelta active\n";
    }

    auto call_pnl = omm::domain::InstrumentFactory::make_call("AAPL", strike, expiry);
    auto put_pnl  = omm::domain::InstrumentFactory::make_put ("AAPL", strike, expiry);
    std::vector<omm::demo::ExtendedPnLTracker::OptionEntry> pnl_opts = {
        {CALL_ID, call_pnl}, {PUT_ID, put_pnl}
    };

    std::string tracker_label = neural_available ? "NeuralBSDEHedger"
        : (mode == HedgerMode::ROUGH_DELTA ? "RoughVolDelta" : "BSDelta");

    auto tracker = std::make_shared<omm::demo::ExtendedPnLTracker>(
        bus, rough_engine, pnl_opts, extractor, tracker_label, /*rate=*/0.05
    );
    tracker->register_handlers();

    std::cout << "── Simulation Start ─────────────────────────────────────\n";
    if (use_real_data) {
        chain_adapter->run([&](double atm_iv, double T_sim, const std::string& /*date*/) {
            if (delta_hedger && atm_iv > 0.0 && T_sim > 0.0)
                delta_hedger->set_market_state(atm_iv, T_sim);
        });
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
              << "║  Alpha PnL Test — IS/OOS Calendar Split                  ║\n"
              << "╠══════════════════════════════════════════════════════════╣\n"
              << "║  IS  (train): 2025-08-07 → 2025-12-31                   ║\n"
              << "║  OOS (test):  2026-01-02 → 2026-02-06                   ║\n"
              << "║  BSDE-IS: trained on real IS-period SPY paths            ║\n"
              << "║  Params: H=0.01  η=0.838  ρ=−0.507  v0=0.077            ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    const bool use_real_data = std::filesystem::exists(REAL_DATA_CSV);
    std::cout << "[Data] " << (use_real_data
        ? "Real OPRA chain: " + REAL_DATA_CSV
        : "Synthetic data:  " + SYNTH_DATA_CSV) << "\n\n";

    // ══════════════════════════════════════════════════════════
    // IN-SAMPLE block (2025-08-07 → 2025-12-31)
    // Baseline only: BSDE-Synth was trained on this period's
    // data (indirectly via IS-calibrated V0), so IS PnL is
    // informational, not a fair OOS test.
    // ══════════════════════════════════════════════════════════
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  IN-SAMPLE  (2025-08-07 → 2025-12-31)                    ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    auto is_bs    = run_simulation(HedgerMode::BS_DELTA,    "BS Delta [IS]",        use_real_data, "", SPLIT_DATE);
    auto is_rough = run_simulation(HedgerMode::ROUGH_DELTA, "Rough Vol Delta [IS]", use_real_data, "", SPLIT_DATE);
    auto is_bsde  = run_simulation(HedgerMode::NEURAL_BSDE, "BSDE-Synth [IS]",      use_real_data, "", SPLIT_DATE);
    omm::demo::ExtendedPnLTracker::print_comparison(is_bs, is_rough, &is_bsde);

    auto is_mod = run_simulation_modern(HedgerMode::BS_DELTA, "2020s Alpha [IS]", use_real_data, "", SPLIT_DATE);
    std::cout << "\n  [IS] 2020s Alpha vs Rough-Vol Composite:\n";
    omm::demo::ExtendedPnLTracker::print_comparison(is_bs, is_mod);

    // ══════════════════════════════════════════════════════════
    // OUT-OF-SAMPLE block (2026-01-02 → 2026-02-06)
    // Key evaluation: BSDE-IS uses the IS-trained model.
    // Does training on real SPY dynamics improve hedging OOS?
    // ══════════════════════════════════════════════════════════
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  OUT-OF-SAMPLE  (2026-01-02 → 2026-02-06)                ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    auto oos_bs       = run_simulation(HedgerMode::BS_DELTA,             "BS Delta [OOS]",          use_real_data, SPLIT_DATE, "");
    auto oos_rough    = run_simulation(HedgerMode::ROUGH_DELTA,          "Rough Vol Delta [OOS]",   use_real_data, SPLIT_DATE, "");
    auto oos_bsde     = run_simulation(HedgerMode::NEURAL_BSDE,          "BSDE-Synth [OOS]",        use_real_data, SPLIT_DATE, "");
    auto oos_is       = run_simulation(HedgerMode::NEURAL_BSDE_IS,       "BSDE-IS [OOS]",           use_real_data, SPLIT_DATE, "");
    auto oos_is_delta = run_simulation(HedgerMode::NEURAL_BSDE_IS_DELTA, "BSDE-IS-Delta [OOS]",     use_real_data, SPLIT_DATE, "");

    // Comparison 1: BS vs Rough vs BSDE-Synth (does rough-vol alpha persist OOS?)
    omm::demo::ExtendedPnLTracker::print_comparison(oos_bs, oos_rough, &oos_bsde);

    // Comparison 2: BS vs Rough vs BSDE-IS full-replication (zeroes out VRP — informational)
    std::cout << "\n  [OOS] IS-trained BSDE (full replication) vs classical hedgers:\n";
    omm::demo::ExtendedPnLTracker::print_comparison(oos_bs, oos_rough, &oos_is);

    // Comparison 3: BS vs Rough vs BSDE-IS-Delta (partial hedge preserves VRP alpha)
    std::cout << "\n  [OOS] IS-trained BSDE (delta-only) vs classical hedgers:\n";
    omm::demo::ExtendedPnLTracker::print_comparison(oos_bs, oos_rough, &oos_is_delta);

    auto oos_mod = run_simulation_modern(HedgerMode::BS_DELTA, "2020s Alpha [OOS]", use_real_data, SPLIT_DATE, "");
    std::cout << "\n  [OOS] 2020s Alpha vs Rough-Vol Composite:\n";
    omm::demo::ExtendedPnLTracker::print_comparison(oos_bs, oos_mod);

    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║  Complete.                                               ║\n"
              << "║  Key: BSDE-IS-Delta PnL ≈ BS Delta → VRP alpha preserved ║\n"
              << "║       BSDE-IS (full replic.) PnL ≈ 0 → VRP eliminated    ║\n"
              << "║  Build with -DBUILD_ONNX_DEMO=ON to activate BSDE passes.║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";

    return 0;
}
