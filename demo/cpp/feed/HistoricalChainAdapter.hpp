#pragma once
// ============================================================
// File: HistoricalChainAdapter.hpp  (demo/cpp/)
// Role: real-data counterpart to SyntheticOptionFeed + MarketDataAdapter.
//
// Reads demo/data/spy_chain_panel.csv (produced by prepare_spy_data.py)
// and publishes three events per row:
//   1. MarketDataEvent        — SPY spot (from put-call parity recovery)
//   2. OptionMidQuoteEvent    — ATM call ("ATM_CALL") with bid/ask
//   3. OptionMidQuoteEvent    — ATM put  ("ATM_PUT")  with bid/ask
//
// CSV columns — base (cols 0-11, same as original spy_atm_chain.csv):
//   timestamp_utc, underlying_price, atm_strike, expiry_date,
//   time_to_expiry, call_mid, put_mid,
//   call_bid, call_ask, put_bid, put_ask, rv5_ann
//
// Extended columns (cols 12+, present in spy_chain_panel.csv):
//   atm_iv  — pre-computed BS IV at ATM; exposed via row.atm_iv for
//             use by DeltaHedger::set_market_state() in the main loop.
//   (cols 13-21: 25Δ strike data — stored but not yet published as events)
//
// bid_price / ask_price are passed through to OptionMidQuoteEvent so that
// SimpleExecSim can fill options at the correct side of the market.
//
// initial_strike() / initial_expiry() — peek at row 0 before run() so
// alpha_main.cpp can construct correctly-struck option objects.
//
// on_row callback — optional per-row hook, called after publishing events,
// allowing alpha_main.cpp to call set_market_state() with correct IV + T.
// ============================================================

#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <chrono>

#include "core/events/EventBus.hpp"
#include "core/events/Events.hpp"

namespace omm::demo {

class HistoricalChainAdapter {
public:
    HistoricalChainAdapter(
        std::shared_ptr<events::EventBus> bus,
        const std::string& csv_path
    ) : bus_(std::move(bus))
    {
        load_csv(csv_path);
        if (rows_.empty())
            throw std::runtime_error("[HistoricalChainAdapter] CSV 无有效行: " + csv_path);

        double lo = rows_[0].underlying, hi = lo;
        double sum_call_spread = 0.0, sum_put_spread = 0.0;
        int    spread_count = 0;
        for (auto& r : rows_) {
            lo = std::min(lo, r.underlying);
            hi = std::max(hi, r.underlying);
            if (r.call_ask > r.call_bid && r.put_ask > r.put_bid) {
                sum_call_spread += r.call_ask - r.call_bid;
                sum_put_spread  += r.put_ask  - r.put_bid;
                ++spread_count;
            }
        }
        double avg_call_spread = spread_count > 0 ? sum_call_spread / spread_count : 0.0;
        double avg_put_spread  = spread_count > 0 ? sum_put_spread  / spread_count : 0.0;
        std::cout << std::fixed << std::setprecision(4)
                  << "[HistoricalChainAdapter] 已载入 " << rows_.size()
                  << " 行真实 SPY 期权数据\n"
                  << "  行情区间: " << rows_.front().timestamp_str
                  << " … " << rows_.back().timestamp_str << "\n"
                  << "  SPY 区间: " << lo << " – " << hi << "\n"
                  << "  平均买卖价差: 认购 $" << avg_call_spread
                  << "  认沽 $" << avg_put_spread << "\n";
    }

    // ── 前置信息（在 run() 前调用）──────────────────────────────
    double initial_strike() const { return rows_[0].atm_strike; }

    std::chrono::system_clock::time_point initial_expiry() const {
        return parse_date_to_tp(rows_[0].expiry_date);
    }

    // Expose row count for diagnostics
    std::size_t row_count() const { return rows_.size(); }

    // ── 主循环 ─────────────────────────────────────────────────
    // on_row: optional callback invoked AFTER publishing all three events for
    // a row. Receives (atm_iv, time_to_expiry, date_str) so the caller can
    // call delta_hedger->set_market_state() and detect session boundaries.
    void run(std::function<void(double /*atm_iv*/, double /*T_sim*/,
                                const std::string& /*date*/)> on_row = nullptr) {
        std::cout << "[HistoricalChainAdapter] 开始回放…\n";
        for (const auto& r : rows_) {
            auto ts = parse_ts(r.timestamp_str);

            // 1. MarketDataEvent — 标的价格
            bus_->publish(events::MarketDataEvent{ts, r.underlying});

            // 2. ATM 认购期权行情（含真实买卖价）
            bus_->publish(events::OptionMidQuoteEvent{
                "ATM_CALL",
                r.call_mid,
                r.call_bid,
                r.call_ask,
                r.underlying,
                r.atm_strike,
                r.time_to_expiry,
                /*is_call=*/true,
                ts
            });

            // 3. ATM 认沽期权行情（含真实买卖价）
            bus_->publish(events::OptionMidQuoteEvent{
                "ATM_PUT",
                r.put_mid,
                r.put_bid,
                r.put_ask,
                r.underlying,
                r.atm_strike,
                r.time_to_expiry,
                /*is_call=*/false,
                ts
            });

            // 4. Per-row callback — invoked after all events for this bar.
            // Allows caller to call set_market_state(atm_iv, T_sim) and
            // detect session boundaries using date_str().
            if (on_row)
                on_row(r.atm_iv, r.time_to_expiry, r.date_str());
        }
        std::cout << "[HistoricalChainAdapter] 回放完成 (" << rows_.size() << " 行)\n";
    }

private:
    struct Row {
        std::string timestamp_str;
        double      underlying;
        double      atm_strike;
        std::string expiry_date;
        double      time_to_expiry;
        double      call_mid;
        double      put_mid;
        double      call_bid = 0.0;
        double      call_ask = 0.0;
        double      put_bid  = 0.0;
        double      put_ask  = 0.0;
        double      rv5_ann  = 0.0;   // rolling 5-bar annualised realised vol
        double      atm_iv   = 0.0;   // pre-computed BS IV (col 12, spy_chain_panel.csv)
        // date portion (YYYY-MM-DD) for session boundary detection
        std::string date_str() const {
            return timestamp_str.size() >= 10 ? timestamp_str.substr(0, 10) : "";
        }
    };

    std::shared_ptr<events::EventBus> bus_;
    std::vector<Row>                  rows_;

    // ── CSV 解析 ────────────────────────────────────────────────
    void load_csv(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("[HistoricalChainAdapter] 无法打开: " + path);

        std::string line;
        std::getline(f, line);  // skip header

        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::vector<std::string> cols;
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ','))
                cols.push_back(tok);
            if (cols.size() < 7) continue;

            Row r;
            r.timestamp_str  = cols[0];
            r.underlying     = std::stod(cols[1]);
            r.atm_strike     = std::stod(cols[2]);
            r.expiry_date    = cols[3];
            r.time_to_expiry = std::stod(cols[4]);
            r.call_mid       = std::stod(cols[5]);
            r.put_mid        = std::stod(cols[6]);
            // Extended columns (cols 7-11) — present in new CSV, optional for backward compat
            if (cols.size() >= 12) {
                r.call_bid = std::stod(cols[7]);
                r.call_ask = std::stod(cols[8]);
                r.put_bid  = std::stod(cols[9]);
                r.put_ask  = std::stod(cols[10]);
                r.rv5_ann  = std::stod(cols[11]);
            } else {
                // Backward compat: synthesise bid/ask from mid with a 1-cent half-spread
                r.call_bid = r.call_mid - 0.01;
                r.call_ask = r.call_mid + 0.01;
                r.put_bid  = r.put_mid  - 0.01;
                r.put_ask  = r.put_mid  + 0.01;
            }
            // Col 12: atm_iv (spy_chain_panel.csv only; backward compat: stays 0.0)
            if (cols.size() >= 13 && !cols[12].empty() && cols[12] != "nan") {
                try { r.atm_iv = std::stod(cols[12]); } catch (...) { r.atm_iv = 0.0; }
            }
            rows_.push_back(r);
        }
    }

    // ── 时间戳解析 ──────────────────────────────────────────────
    // 解析 "2025-08-07T13:31:00+00:00" → system_clock::time_point
    static std::chrono::system_clock::time_point parse_ts(const std::string& s) {
        // 提取 "2025-08-07T13:31:00" 部分（忽略时区，视为 UTC）
        std::tm tm = {};
        std::istringstream iss(s.substr(0, 19));
        iss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
        std::time_t t = timegm(&tm);  // UTC mktime
        return std::chrono::system_clock::from_time_t(t);
    }

    // 解析 "2025-08-18" → system_clock::time_point (midnight UTC)
    static std::chrono::system_clock::time_point parse_date_to_tp(const std::string& s) {
        std::tm tm = {};
        std::istringstream iss(s);
        iss >> std::get_time(&tm, "%Y-%m-%d");
        std::time_t t = timegm(&tm);
        return std::chrono::system_clock::from_time_t(t);
    }
};

} // namespace omm::demo
