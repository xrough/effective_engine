#pragma once
#include <memory>
#include <string>
#include <vector>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"
#include "../domain/Instrument.hpp"
#include "../domain/PortfolioAggregate.hpp"
#include "../analytics/PricingEngine.hpp"

// ============================================================
// 文件：PortfolioService.hpp
// 职责：模式无关的持仓追踪与估值服务。
//
// 设计：将"持仓追踪"从风控应用中抽离，使 SellerRiskApp / BuyerRiskApp
//       只需关注策略评估逻辑，不再管理持仓状态。
//
// 事件订阅：
//   FillEvent       → 更新持仓（applyFill），触发重新估值
//   MarketDataEvent → 更新标的价格，触发重新估值
//
// 事件发布：
//   PortfolioUpdateEvent → 含最新 RiskMetrics 快照，供风控应用消费
//
// 生命周期：由 SellerModule / BuyerModule 的 install() 创建并连线。
// ============================================================

namespace omm::application {

class PortfolioService {
public:
    PortfolioService(
        std::shared_ptr<events::EventBus>            bus,
        std::shared_ptr<domain::IPricingEngine>      pricing_engine,
        std::vector<std::shared_ptr<domain::Option>> options,
        std::string                                  underlying_id,
        std::string                                  account_id
    );

    // 向 EventBus 注册 FillEvent / MarketDataEvent 处理器
    void register_handlers();

private:
    // 成交后：更新持仓 → 重新估值 → 发布快照
    void on_fill(const events::FillEvent& event);

    // 行情更新后：刷新价格 → 重新估值 → 发布快照（有持仓时）
    void on_market(const events::MarketDataEvent& event);

    // 估值并发布 PortfolioUpdateEvent
    void publish_snapshot();

    std::shared_ptr<events::EventBus>       bus_;
    std::shared_ptr<domain::IPricingEngine> pricing_engine_;
    std::string                             underlying_id_;
    std::string                             account_id_;

    domain::PortfolioAggregate portfolio_; // 账户持仓聚合（核心状态）
    double                     last_price_; // 最近一次标的价格
};

} // namespace omm::application
