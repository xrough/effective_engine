#pragma once
#include <memory>
#include <string>
#include <vector>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"
#include "../domain/Instrument.hpp"
#include "../domain/PricingEngine.hpp"
#include "../domain/CalibrationEngine.hpp"

// ============================================================
// 文件：BacktestCalibrationApp.hpp
// 职责：回测与参数校准应用 — 在隔离的事件总线上重放历史行情，
//       比对"市场真实价格"与"模型预测价格"，优化模型参数，
//       最终通过 ParamUpdateEvent 将校准结果发布到系统。
//
// 对应 Risk_Calibration.md §2（Backtest & Calibration Application）
//
// 核心概念（参数校准示例）：
//   "市场引擎" market_engine — 代表真实市场的价格来源（vol=0.25），
//                              在实际场景中替换为交易所行情
//   "模型引擎" model_engine  — 待校准的内部定价模型（初始 vol=0.15），
//                              优化目标：使其预测价格逼近市场价格
//
// 参数反馈闭环（Risk_Calibration.md §2.7）：
//   BacktestCalibrationApp
//         ↓ 发布 ParamUpdateEvent
//   ParameterStore（参数仓库）
//         ↓ 存储带时间戳的校准结果
//   实时定价引擎（可通过 IModelParamSource 查询最新参数）
//
// 事件订阅：MarketDataEvent（由隔离事件总线重放历史行情）
// 事件发布：ParamUpdateEvent（发布到主事件总线）
// ============================================================

namespace omm::application {

class BacktestCalibrationApp {
public:
    BacktestCalibrationApp(
        std::shared_ptr<events::EventBus>              backtest_bus,
        std::shared_ptr<events::EventBus>              main_bus,
        std::shared_ptr<domain::BlackScholesPricingEngine> market_engine,
        std::shared_ptr<domain::BlackScholesPricingEngine> model_engine,
        std::vector<std::shared_ptr<domain::Option>>   options,
        std::shared_ptr<domain::CalibrationEngine>     calibrator,
        std::string                                    model_id
    );

    // register_handlers() — 在回测专用总线上注册行情处理器
    void register_handlers();

    // finalize() — 运行校准优化，发布 ParamUpdateEvent
    //   1. 调用 calibrator_.solve() 执行黄金分割搜索
    //   2. 将最优参数封装为 ParamUpdateEvent 发布到主总线
    //   返回：校准得到的最优隐含波动率
    double finalize();

private:
    // on_market() — 行情事件处理器（回测重放时调用）
    //   对每个期权：
    //     market_price = market_engine_.price(opt, S).theo  （"真实"市场价）
    //     model_price  = model_engine_.price(opt, S).theo   （模型当前预测价）
    //     calibrator_.observe(market_price, model_price)    （记录偏差）
    void on_market(const events::MarketDataEvent& event);

    // RawObs — 原始观测：保存每个 tick 上每个期权的（标的价格、合约指针、市场价格），
    //          供 finalize() 在不同 σ 下重新计算模型价格，实现真正的参数扫描
    struct RawObs {
        double                         underlying_price; // 该 tick 的标的资产价格
        std::shared_ptr<domain::Option> option;          // 对应期权合约
        double                         market_price;     // 市场（高精度 BS）报价
    };

    std::shared_ptr<events::EventBus>              backtest_bus_;
    std::shared_ptr<events::EventBus>              main_bus_;
    std::shared_ptr<domain::BlackScholesPricingEngine> market_engine_;
    std::shared_ptr<domain::BlackScholesPricingEngine> model_engine_;
    std::vector<std::shared_ptr<domain::Option>>   options_;
    std::shared_ptr<domain::CalibrationEngine>     calibrator_;
    std::string                                    model_id_;
    int                                            tick_count_;
    std::vector<RawObs>                            raw_observations_; // 原始观测缓存
};

} // namespace omm::application
