#pragma once
#include <string>
#include <chrono>
#include <unordered_map>

// ============================================================
// 文件：Events.hpp
// 职责：定义系统中所有领域事件（Domain Events）的数据结构。
//
// 设计原则：
//   - 事件是"已发生的事实"，使用结构体表示，无行为方法。
//   - 事件是值对象（Value Object），可安全复制传递。
//   - 所有事件均为纯数据载体，不持有任何 I/O 或业务逻辑。
// ============================================================

// ============================================================
// Event 场景简要说明（Program Remark）
//
// 每个 event 表示系统中某个阶段已经发生的业务事实：
//
// 1. MarketDataEvent
//    → 市场行情更新（价格变化）
//    → 触发重新报价、风险计算
//
// 2. QuoteGeneratedEvent
//    → 做市商生成 bid / ask 报价
//    → 等待客户成交或记录日志
//
// 3. TradeExecutedEvent
//    → 客户成交发生
//    → 更新持仓、计算 PnL、触发对冲
//
// 4. OrderSubmittedEvent
//    → 系统提交对冲订单请求
//    → 订单路由执行或记录审计
//
// ------------------------------------------------------------
// 典型流程：
//
// MarketDataEvent
//      ↓
// QuoteGeneratedEvent
//      ↓
// TradeExecutedEvent
//      ↓
// OrderSubmittedEvent
//
// ------------------------------------------------------------
// 目的：
//   - 用“已发生的事实”在模块间通信
//   - 解耦系统组件
//   - 支持回放、测试、日志与扩展
// ============================================================

namespace omm::events {

// 时间戳类型别名，统一使用系统时钟
using Timestamp = std::chrono::system_clock::time_point;

// ============================================================
// 1. MarketDataEvent — 行情数据事件
//    发布者：MarketDataAdapter（行情适配器）
//    订阅者：QuoteEngine（报价引擎）、DeltaHedger（Delta 对冲器）
// ============================================================
struct MarketDataEvent { //struct:结构体，默认成员访问权限为public，适合表示纯数据载体
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
// 3. TradeExecutedEvent — 成交事件
//    发布者：ProbabilisticTaker（概率成交模拟器）
//    订阅者：PositionManager（持仓管理器）、DeltaHedger（Delta 对冲器）
//
//    注意：Side 表示"客户方向"（客户的买/卖），而非做市商方向。
//    客户 Buy  → 客户买入期权，做市商持仓减少（做空）
//    客户 Sell → 客户卖出期权，做市商持仓增加（做多）
// ============================================================
enum class Side {
    Buy,   // 客户买入（lift ask）
    Sell   // 客户卖出（hit bid）
};

struct TradeExecutedEvent {
    std::string instrument_id; // 成交合约标识符
    Side        side;          // 客户方向（Buy / Sell）
    double      price;         // 成交价格
    int         quantity;      // 成交数量（手数）
    Timestamp   timestamp;     // 成交时间戳
};

// ============================================================
// 4. OrderSubmittedEvent — 订单提交事件（Command 模式）
//    发布者：DeltaHedger（Delta 对冲器）
//    订阅者：OrderRouter（订单路由存根）、日志处理器
//
//    模式：Command — 该事件将"提交订单"的意图封装为数据对象，
//    发布者无需知道谁来执行，也无需等待执行结果。
//    这使得订单可被记录、重放或审计，而不依赖具体执行方。
// ============================================================
enum class OrderType {
    Market, // 市价单：立即以市场最优价成交
    Limit   // 限价单：仅在指定价格或更优价格成交
};

struct OrderSubmittedEvent {
    std::string instrument_id; // 订单标的合约（通常为标的资产，如 "AAPL"）
    Side        side;          // 订单方向（Buy / Sell）
    int         quantity;      // 订单数量
    OrderType   order_type;    // 订单类型（Market / Limit）
};

// ============================================================
// 5. RiskControlEvent — 风控指令事件（RealtimeRiskApp 发出）
//    发布者：RealtimeRiskApp（实时风控应用）
//    订阅者：OrderRouter（可阻断新订单）、日志处理器
//
//    RiskAction 枚举说明：
//      BlockOrders  — 冻结账户，禁止提交新订单
//      CancelOrders — 撤销所有未成交订单
//      ReduceOnly   — 仅允许减仓方向的订单
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
// 6. RiskAlertEvent — 风险预警事件
//    发布者：RealtimeRiskApp（实时风控应用）
//    订阅者：日志处理器、监控系统
//
//    用于记录风险指标接近限额但尚未触发强制动作的情形。
// ============================================================
struct RiskAlertEvent {
    std::string account_id;   // 告警账户 ID
    std::string metric_name;  // 触发告警的指标名称（如 "delta"）
    double      value;        // 当前指标值
    double      limit;        // 对应风险限额
};

// ============================================================
// 7. ParamUpdateEvent — 模型参数更新事件
//    发布者：BacktestCalibrationApp（回测校准应用）
//    订阅者：ParameterStore（参数仓库）
//
//    校准引擎完成优化后，通过此事件将新参数广播到系统，
//    实现"校准 → 参数更新 → 实时定价"闭环。
// ============================================================
struct ParamUpdateEvent {
    std::string model_id;                                  // 模型标识符（如 "bs_model"）
    std::unordered_map<std::string, double> params;        // 参数键值对（如 {"vol": 0.22}）
    Timestamp   updated_at;                                // 参数更新时间戳
};

} // namespace omm::events
