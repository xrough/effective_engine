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
    std::string                                    model_id)
    : backtest_bus_(std::move(backtest_bus))
    , main_bus_(std::move(main_bus))
    , market_engine_(std::move(market_engine))
    , model_engine_(std::move(model_engine))
    , options_(std::move(options))
    , calibrator_(std::move(calibrator))
    , model_id_(std::move(model_id))
    , tick_count_(0) {}

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

    return best_vol;
}

} // namespace omm::application
