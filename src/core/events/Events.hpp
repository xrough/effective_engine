#pragma once
#include <string>
#include <chrono>
#include <unordered_map>
#include "../domain/RiskMetrics.hpp"

// ============================================================
// File: Events.hpp
//
//   - Events are "facts that have occurred", represented as structs without behavior methods.
//   - Events are value objects (Value Object), safe to copy and pass around.
//   - All events are pure data carriers, holding no I/O or business logic.
//
// Event Flow Overview (seller):
//
//   MarketDataEvent
//        ↓
//   QuoteGeneratedEvent
//        ↓
//   FillEvent   ← Deal execution (neutral perspective, unified for both sides)
//        ↓ (via PortfolioService)
//   PortfolioUpdateEvent         ← RiskMetrics snapshot
//        ↓
//   RiskControlEvent / RiskAlertEvent
//
// Side Semantics (from 'our' perspective after initialization)
// ============================================================

namespace omm::events {

// unified timestamp type for all events
using Timestamp = std::chrono::system_clock::time_point;

// ============================================================
// MarketDataEvent from MarketDataAdapter to QuoteEngine / PortfolioService
// ============================================================
struct MarketDataEvent {
    Timestamp timestamp;        // 行情时间戳
    double    underlying_price; // 标的资产当前价格（如 AAPL 的股价）
};

// ============================================================
// QuoteGeneratedEvent (seller only) from QuoteEngine to ProbabilisticTaker (intention, not a real market quote)
// ============================================================
struct QuoteGeneratedEvent {
    std::string instrument_id; // e.g. "AAPL_150_C_20240201"
    double      bid_price;     
    double      ask_price;     
    Timestamp   timestamp;    
};

// ============================================================
// FillEvent from execution to PortfolioService / DeltaHedger
//  side semantics: Buy = we buy
// ============================================================
enum class Side {
    Buy,  
    Sell  
};

struct FillEvent {
    std::string instrument_id; 
    Side        side;          // out side
    double      fill_price;    
    int         fill_qty;      // deal quantity
    std::string producer;      //（"customer_taker"、"broker"、"hedge_order"）
    Timestamp   timestamp; 
};

// ============================================================
// OrderSubmittedEvent from Delta OrderRouter to execution (optional, for logging / monitoring)
// ============================================================
enum class OrderType {
    Market, // Market order: executed immediately at the best available price
    Limit   // Limit order: only executed at the specified price or better
};

// OrderSubmittedEvent: intention to submit an order, the excution will be decided by OrderRouter -> FillEvent is the actual deal execution result (can also used by logging / monitoring)
struct OrderSubmittedEvent {
    std::string instrument_id; // Order target contract (usually the underlying asset, e.g., "AAPL")
    Side        side;          // Order direction (from our perspective: Buy = we buy)
    int         quantity;      // Order quantity
    OrderType   order_type;    // Order type (Market / Limit)
};

// ============================================================
// PortfolioUpdateEvent from PortfolioService to RiskApp 
// ============================================================
struct PortfolioUpdateEvent {
    std::string         account_id; 
    domain::RiskMetrics metrics;    // Current risk metrics snapshot for the account
    Timestamp           timestamp; 
};

// ============================================================
// 6. RiskControlEvent from RiskApp to execution / order management (e.g., OrderRouter)
// ============================================================
enum class RiskAction {
    BlockOrders,  
    CancelOrders, 
    ReduceOnly   
};

struct RiskControlEvent {
    std::string account_id; 
    RiskAction  action;     
    std::string reason;     // reason for the risk control action (e.g., "intraday_drawdown_exceeded") for logging and monitoring purposes
};

// ============================================================
// RiskAlertEvent from RiskApp to monitoring / alerting system (e.g., send email or Slack notification)
// ============================================================
struct RiskAlertEvent {
    std::string account_id;  
    std::string metric_name;  
    double      value;        // Current metric value
    double      limit;        // Corresponding risk limit
};

// ============================================================
// 8. ParamUpdateEvent from model calibration to monitoring / alerting system
// ============================================================
struct ParamUpdateEvent {
    std::string model_id;                
    std::unordered_map<std::string, double> params;   // Parameter key-value pairs (e.g., {"vol": 0.22})
    Timestamp   updated_at;                       
};

} // namespace omm::events
