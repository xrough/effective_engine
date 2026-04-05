#pragma once
#include <deque>
#include <cmath>
#include <memory>
#include <numeric>
#include "../../core/events/EventBus.hpp"
#include "../../core/events/Events.hpp"
#include "../../core/interfaces/IAlphaSignal.hpp"
#include "../../core/analytics/RoughVolPricingEngine.hpp"
#include "ImpliedVarianceExtractor.hpp"

// ============================================================
// File: VarianceAlphaSignal.hpp
// Role: variance alpha signal generator, implementing IAlphaSignal interface.
//
// Signal logic:
//   raw_spread  = σ²_atm(market ) − xi0 * T（rough vola proxy）
//   zscore      = (raw_spread − rolling_mean) / rolling_std
//============================================================

// 粗糙模型预测：使用已校准的 RoughVolPricingEngine 的 xi0 参数。
//   E[RV(t,T)] ≈ xi0 * T（粗糙 Bergomi 零阶近似）
//
// 滚动窗口默认 50 个观测，窗口未充满时 valid=false。
// 充满后发布 SignalSnapshotEvent。

namespace omm::buyer {

struct AlphaSignalConfig {
    int    window    = 50;    // 滚动 z-score 窗口长度
    double z_entry   = 1.5;  // 入场阈值
    double z_exit    = 0.5;  // 离场阈值
    double z_cap     = 3.0;  // z-score 上限（仓位缩放用）
    double base_vega = 1000.0; // 基础 vega 预算（美元）
};

class VarianceAlphaSignal : public core::IAlphaSignal {
public:
    VarianceAlphaSignal(
        std::shared_ptr<events::EventBus>              bus,
        std::shared_ptr<ImpliedVarianceExtractor>      extractor,
        std::shared_ptr<domain::RoughVolPricingEngine> rough_engine,
        AlphaSignalConfig cfg = {}
    ) : bus_(std::move(bus))
      , extractor_(std::move(extractor))
      , rough_(std::move(rough_engine))
      , cfg_(cfg) {}

    void register_handlers() {
        bus_->subscribe<events::OptionMidQuoteEvent>(
            [this](const events::OptionMidQuoteEvent& e) { on_option_quote(e); }
        );
    }

    // IAlphaSignal 接口：当前版本仅订阅 OptionMidQuoteEvent 驱动信号，
    // on_market_data() 保留接口但不触发信号更新（预留高频 delta 检查扩展）。
    void on_market_data(const events::MarketDataEvent& /*event*/) override {}

    void on_option_quote(const events::OptionMidQuoteEvent& e) {
        ImpliedVariancePoint iv = extractor_->last_point();
        if (!iv.valid) return;

        // 粗糙模型远期方差代理
        double rough_fv = rough_->get_params().xi0 * iv.time_to_expiry;
        double spread   = iv.atm_implied_variance - rough_fv;

        spread_history_.push_back(spread);
        if ((int)spread_history_.size() > cfg_.window)
            spread_history_.pop_front();

        bool window_full = ((int)spread_history_.size() == cfg_.window);
        double zscore    = 0.0;

        if (window_full) {
            std::pair<double,double> ms = rolling_stats();
            if (ms.second > 1e-12)
                zscore = (spread - ms.first) / ms.second;
        }

        events::SignalSnapshotEvent snap;
        snap.ts                      = e.timestamp;
        snap.valid                   = window_full;
        snap.atm_implied_variance    = iv.atm_implied_variance;
        snap.rough_forecast_variance = rough_fv;
        snap.raw_spread              = spread;
        snap.zscore                  = zscore;
        snap.calibration_ok          = true; // 粗糙引擎校准后始终有效

        bus_->publish(snap);
    }

private:
    // 计算滚动均值和标准差
    std::pair<double, double> rolling_stats() const {
        double mean = std::accumulate(spread_history_.begin(),
                                      spread_history_.end(), 0.0)
                      / spread_history_.size();
        double var = 0.0;
        for (double v : spread_history_)
            var += (v - mean) * (v - mean);
        double std_dev = std::sqrt(var / spread_history_.size());
        return {mean, std_dev};
    }

    std::shared_ptr<events::EventBus>              bus_;
    std::shared_ptr<ImpliedVarianceExtractor>      extractor_;
    std::shared_ptr<domain::RoughVolPricingEngine> rough_;
    AlphaSignalConfig                              cfg_;
    std::deque<double>                             spread_history_;
};

} // namespace omm::buyer
