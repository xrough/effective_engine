#pragma once
#include <string>
#include <chrono>
#include <unordered_map>
#include "../domain/RiskMetrics.hpp"

// ============================================================
// 文件：Events.hpp
// 职责：定义系统中所有领域事件（Domain Events）的数据结构。
//
// 设计原则：
//   - 事件是"已发生的事实"，使用结构体表示，无行为方法。
//   - 事件是值对象（Value Object），可安全复制传递。
//   - 所有事件均为纯数据载体，不持有任何 I/O 或业务逻辑。
//
// 事件流概览：
//
//   MarketDataEvent
//        ↓
//   QuoteGeneratedEvent
//        ↓
//   FillEvent                    ← 统一成交事件（我方视角）
//        ↓ (via PortfolioService)
//   PortfolioUpdateEvent         ← 持仓/风险指标快照
//        ↓
//   RiskControlEvent / RiskAlertEvent
//
// Side 语义统一约定（Phase 2 起）：
//   所有 FillEvent 的 side 字段表示"我方方向"：
//     Side::Buy  = 我们买入（持仓增加）
//     Side::Sell = 我们卖出（持仓减少）
// ============================================================

namespace omm::events {

// 时间戳类型别名，统一使用系统时钟
using Timestamp = std::chrono::system_clock::time_point;

// ============================================================
// 1. MarketDataEvent — 行情数据事件
//    发布者：MarketDataAdapter（行情适配器）
//    订阅者：QuoteEngine（报价引擎）、PortfolioService（持仓服务）
// ============================================================
struct MarketDataEvent {
    Timestamp timestamp;        // 行情时间戳
    double    underlying_price; // 标的资产当前价格（如 AAPL 的股价）
};

// ============================================================
// 2. QuoteGeneratedEvent — 报价生成事件
//    发布者：QuoteEngine（报价引擎）
//    订阅者：ProbabilisticTaker（概率成交模拟器）、日志处理器
// ============================================================
struct QuoteGeneratedEvent {
    std::string instrument_id; // 期权合约标识符，如 "AAPL_150_C_20240201"
    double      bid_price;     // 做市商买价（客户可卖出的价格）
    double      ask_price;     // 做市商卖价（客户可买入的价格）
    Timestamp   timestamp;     // 报价时间戳
};

// ============================================================
// 3. FillEvent — 统一成交事件（替代 TradeExecutedEvent）
//    发布者：ProbabilisticTaker（"customer_taker"）
//             BrokerAdapter（"broker"，买方模式）
//             DeltaHedger 内部对冲（"hedge_order"）
//    订阅者：PortfolioService（更新持仓）、DeltaHedger（对冲检查）
//
//    Side 语义（我方视角）：
//      Buy  = 我们买入该合约（持仓增加，如 +1 手看涨期权）
//      Sell = 我们卖出该合约（持仓减少，如 -1 手看涨期权）
// ============================================================
enum class Side {
    Buy,  // 我们买入
    Sell  // 我们卖出
};

struct FillEvent {
    std::string instrument_id; // 成交合约标识符
    Side        side;          // 我方方向（Buy = 我们买入；Sell = 我们卖出）
    double      fill_price;    // 成交价格
    int         fill_qty;      // 成交数量（手）
    std::string producer;      // 填单方标识（"customer_taker"、"broker"、"hedge_order"）
    Timestamp   timestamp;     // 成交时间戳
};

// ============================================================
// 4. OrderSubmittedEvent — 订单提交事件（Command 模式）
//    发布者：DeltaHedger（Delta 对冲器）
//    订阅者：OrderRouter（订单路由存根）、日志处理器
// ============================================================
enum class OrderType {
    Market, // 市价单：立即以市场最优价成交
    Limit   // 限价单：仅在指定价格或更优价格成交
};

struct OrderSubmittedEvent {
    std::string instrument_id; // 订单标的合约（通常为标的资产，如 "AAPL"）
    Side        side;          // 订单方向（我方视角：Buy = 我们买入）
    int         quantity;      // 订单数量
    OrderType   order_type;    // 订单类型（Market / Limit）
};

// ============================================================
// 5. PortfolioUpdateEvent — 持仓指标快照事件
//    发布者：PortfolioService（持仓服务，订阅 FillEvent + MarketDataEvent 后生成）
//    订阅者：SellerRiskApp / BuyerRiskApp（仅做策略评估，不再管理持仓）
//
//    设计：将"持仓追踪"与"风险策略评估"解耦。
//    PortfolioService 负责维护 PortfolioAggregate；
//    RiskApp 只订阅此快照事件，应用 IRiskPolicy，发布风控指令。
// ============================================================
struct PortfolioUpdateEvent {
    std::string         account_id; // 账户标识符
    domain::RiskMetrics metrics;    // 当前风险指标快照
    Timestamp           timestamp;  // 快照生成时间
};

// ============================================================
// 6. RiskControlEvent — 风控指令事件
//    发布者：SellerRiskApp / BuyerRiskApp
//    订阅者：OrderRouter（可阻断新订单）、日志处理器
// ============================================================
enum class RiskAction {
    BlockOrders,  // 冻结账户下单权限
    CancelOrders, // 批量撤单
    ReduceOnly    // 限制为减仓模式
};

struct RiskControlEvent {
    std::string account_id; // 受影响账户 ID
    RiskAction  action;     // 风控动作
    std::string reason;     // 触发原因描述（供日志记录）
};

// ============================================================
// 7. RiskAlertEvent — 风险预警事件
//    发布者：SellerRiskApp / BuyerRiskApp
//    订阅者：日志处理器、监控系统
// ============================================================
struct RiskAlertEvent {
    std::string account_id;   // 告警账户 ID
    std::string metric_name;  // 触发告警的指标名称（如 "intraday_drawdown"）
    double      value;        // 当前指标值
    double      limit;        // 对应风险限额
};

// ============================================================
// 8. ParamUpdateEvent — 模型参数更新事件
//    发布者：BacktestCalibrationApp（回测校准应用）
//    订阅者：ParameterStore（参数仓库）
// ============================================================
struct ParamUpdateEvent {
    std::string model_id;                             // 模型标识符（如 "bs_model"）
    std::unordered_map<std::string, double> params;   // 参数键值对（如 {"vol": 0.22}）
    Timestamp   updated_at;                           // 参数更新时间戳
};

} // namespace omm::events
