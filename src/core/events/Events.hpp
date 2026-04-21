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

// ============================================================
// OptionMidQuoteEvent — ATM 期权中间价行情（买方 alpha 入口）
// Published by: MarketDataAdapter
// Consumed by: ImpliedVarianceExtractor
// ============================================================
struct OptionMidQuoteEvent {
    std::string instrument_id;
    double      mid_price;      // (bid + ask) / 2 — used for IV extraction and MTM
    double      bid_price = 0.0;  // best bid from exchange (Databento OPRA)
    double      ask_price = 0.0;  // best ask from exchange (Databento OPRA)
    double      underlying;
    double      strike;
    double      time_to_expiry;
    bool        is_call;
    Timestamp   timestamp;
};

// ============================================================
// SmileSnapshotEvent — full smile snapshot per bar (real data mode)
// Published by: HistoricalChainAdapter
// Consumed by: SmileVarianceExtractor
// All fields default to 0/false; subscribers check has_vix / has_ssvi before using.
// ============================================================
struct SmileSnapshotEvent {
    double    vix_varswap    = 0.0;  // model-free σ²_varswap (VIX-style, annualised)
    double    atm_iv         = 0.0;  // BS IV at ATM
    double    rr25_iv        = 0.0;  // risk reversal: IV(25Δ call) − IV(25Δ put)
    double    bf25_iv        = 0.0;  // butterfly: (IV(25Δ call)+IV(25Δ put))/2 − atm_iv
    double    ssvi_theta     = 0.0;  // atm_iv² × T
    double    ssvi_rho       = 0.0;  // SSVI skew parameter ρ
    double    ssvi_phi       = 0.0;  // SSVI smoothing φ
    double    time_to_expiry = 0.0;
    double    underlying     = 0.0;
    bool      has_vix        = false;
    bool      has_ssvi       = false;
    Timestamp timestamp;
};

// ============================================================
// SignalSnapshotEvent — variance alpha signal snapshot
// Published by: VarianceAlphaSignal
// Consumed by: StrategyController + monitoring / logging
// ============================================================
struct SignalSnapshotEvent {
    Timestamp ts;
    bool      valid;                      // 滚动窗口已充满，z-score 可信
    double    atm_implied_variance;       // σ²_atm（市场端）
    double    rough_forecast_variance;    // xi0 * T（粗糙模型端）
    double    raw_spread;                 // atm_iv - rough_forecast
    double    zscore;                     // (spread - mean) / std
    bool      calibration_ok;            // 校准质量标记
    double    atm_vega    = 0.0;         // BS vega of ATM call (用于仓位缩放)
    double    atm_spot    = 0.0;         // 当前标的价格（供 StrategyController 校验）
};

} // namespace omm::events
