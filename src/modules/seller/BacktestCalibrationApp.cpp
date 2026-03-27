#include "BacktestCalibrationApp.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

// ============================================================
// BacktestCalibrationApp 实现
//
// 校准核心思路：
//   回放阶段（on_market）：收集每个 tick 的原始观测
//     raw_observations_ += { S, option, market_price }
//   校准阶段（finalize）：
//     对每个候选 σ，重新用 model_engine_ 计算所有期权价格，
//     与 market_price 比较得到 MSE 损失，黄金分割搜索最小化损失
// ============================================================

namespace omm::application {

BacktestCalibrationApp::BacktestCalibrationApp(
    std::shared_ptr<events::EventBus>              backtest_bus,
    std::shared_ptr<events::EventBus>              main_bus,
    std::shared_ptr<domain::BlackScholesPricingEngine> market_engine,
    std::shared_ptr<domain::BlackScholesPricingEngine> model_engine,
    std::vector<std::shared_ptr<domain::Option>>   options,
    std::shared_ptr<domain::CalibrationEngine>     calibrator,
    std::string                                    model_id
#ifdef BUILD_GRPC_CLIENT
    , std::shared_ptr<infrastructure::ModelServiceClient> grpc_client
#endif
)
    : backtest_bus_(std::move(backtest_bus))
    , main_bus_(std::move(main_bus))
    , market_engine_(std::move(market_engine))
    , model_engine_(std::move(model_engine))
    , options_(std::move(options))
    , calibrator_(std::move(calibrator))
    , model_id_(std::move(model_id))
    , tick_count_(0)
    , last_spot_(0.0)
#ifdef BUILD_GRPC_CLIENT
    , grpc_client_(std::move(grpc_client))
#endif
{}

void BacktestCalibrationApp::register_handlers() {
    // 在回测专用总线上订阅行情事件（与主仿真完全隔离）
    backtest_bus_->subscribe<events::MarketDataEvent>(
        [this](const events::MarketDataEvent& evt) {
            this->on_market(evt);
        }
    );
}

void BacktestCalibrationApp::on_market(const events::MarketDataEvent& event) {
    ++tick_count_;
    const double S = event.underlying_price;
    last_spot_ = S;  // 记录最新标的价格

    for (const auto& opt : options_) {
        // "市场"价格：高精度 BS（vol=0.25），模拟真实市场报价
        double market_price = market_engine_->price(*opt, S).theo;

        // 模型当前预测价（vol=0.15，初始估计）
        double model_price  = model_engine_->price(*opt, S).theo;

        // 记录到 CalibrationEngine（用于初始 MSE 报告）
        calibrator_->observe(market_price, model_price);

        // 缓存原始观测：(S, option, market_price) — 供 finalize() 的 loss_fn 使用
        raw_observations_.push_back(RawObs{S, opt, market_price});

        std::cout << "[回测|tick" << std::setw(2) << tick_count_ << "] "
                  << opt->id()
                  << "  市场价=$" << std::fixed << std::setprecision(4) << market_price
                  << "  模型价=$" << model_price
                  << "  偏差=$" << std::showpos << (model_price - market_price)
                  << std::noshowpos << "\n";
    }
}

double BacktestCalibrationApp::finalize() {
    std::cout << "\n[回测校准] ──── 开始参数优化 ────\n";
    std::cout << "[回测校准] 模型 ID: " << model_id_
              << "  原始观测数: " << raw_observations_.size()
              << "  初始 MSE (vol=0.15): "
              << std::fixed << std::setprecision(6) << calibrator_->mse() << "\n";

    // ── 损失函数：对给定 σ 重新计算所有观测的 MSE ─────────
    // 这是真正的参数化 loss(σ)，每次调用都重新评估 model_engine
    auto loss_fn = [this](double sigma) -> double {
        model_engine_->set_vol(sigma); // 临时更新模型波动率

        double sum_sq = 0.0;
        for (const auto& obs : raw_observations_) {
            // 用新 σ 重新计算模型预测价
            double model_price = model_engine_->price(*obs.option, obs.underlying_price).theo;
            double err         = model_price - obs.market_price;
            sum_sq += err * err;
        }
        return sum_sq / static_cast<double>(raw_observations_.size());
    };

    // ── 运行黄金分割搜索：在 [0.01, 1.0] 范围内寻找最优 σ ──
    double best_vol = calibrator_->solve(0.01, 1.0, loss_fn);

    // 将校准结果应用到模型引擎
    model_engine_->set_vol(best_vol);

    std::cout << "[回测校准] ✅ 校准完成！\n"
              << "             初始波动率: 0.1500\n"
              << "             校准波动率: " << std::fixed << std::setprecision(4)
              << best_vol << "\n"
              << "             目标波动率: 0.2500（市场真实值）\n"
              << "             校准误差:   "
              << std::abs(best_vol - 0.25) / 0.25 * 100.0 << "%\n";

    // ── 发布 ParamUpdateEvent 到主总线 ────────────────────
    events::ParamUpdateEvent update{
        model_id_,
        {{"vol", best_vol}, {"r", 0.05}},
        std::chrono::system_clock::now()
    };

    std::cout << "[回测校准] 📤 发布 ParamUpdateEvent → 模型: "
              << model_id_ << "  参数: vol=" << best_vol << "\n";

    main_bus_->publish(update);

#ifdef BUILD_GRPC_CLIENT
    // ── Phase 2b：粗糙 Bergomi 校准（仅当 gRPC 客户端已注入时执行）────
    if (grpc_client_) {
        std::cout << "\n[回测校准] ──── 开始 Rough Bergomi 校准（gRPC）────\n";

        // 从原始观测中提取 OptionQuote 列表（去重：用最后一次 tick 的市场价）
        std::vector<infrastructure::OptionQuote> quotes;
        for (const auto& opt : options_) {
            double mp = market_engine_->price(*opt, last_spot_).theo;
            auto now  = std::chrono::system_clock::now();
            double T  = std::chrono::duration<double>(opt->expiry() - now).count()
                        / (365.0 * 24.0 * 3600.0);
            T = std::max(T, 1e-6);
            quotes.push_back(infrastructure::OptionQuote{
                opt->strike(), T,
                opt->option_type() == domain::OptionType::Call,
                mp
            });
        }

        try {
            // n_paths=2000 — 计算轻量（"computationally light" 方案）
            auto result = grpc_client_->calibrate_rough_bergomi(
                last_spot_, quotes, 0.05, 0.0, 2000, 64, 42
            );

            std::cout << "[回测校准] ✅ Rough Bergomi 校准完成！\n"
                      << "             H   = " << result.params.at("hurst") << "\n"
                      << "             eta = " << result.params.at("eta")   << "\n"
                      << "             rho = " << result.params.at("rho")   << "\n"
                      << "             xi0 = " << result.params.at("xi0")   << "\n"
                      << "             MSE = " << result.mse
                      << "  耗时 " << result.elapsed_s << "s\n";

            // 发布粗糙参数到主总线 → ParameterStore 存储
            events::ParamUpdateEvent rough_update{
                "rough_bergomi",
                {
                    {"H",   result.params.at("hurst")},
                    {"eta", result.params.at("eta")},
                    {"rho", result.params.at("rho")},
                    {"xi0", result.params.at("xi0")}
                },
                std::chrono::system_clock::now()
            };
            std::cout << "[回测校准] 📤 发布 ParamUpdateEvent → 模型: rough_bergomi\n";
            main_bus_->publish(rough_update);

        } catch (const std::exception& e) {
            std::cout << "[回测校准] ⚠️  Rough Bergomi 校准失败（服务未启动？）: "
                      << e.what() << "\n";
        }
    }
#endif

    return best_vol;
}

} // namespace omm::application
