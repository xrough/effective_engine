#pragma once
// ============================================================
// File: HistoricalChainAdapter.hpp  (demo/cpp/)
// Role: real-data counterpart to SyntheticOptionFeed + MarketDataAdapter.
//
// Reads demo/data/spy_atm_chain.csv (produced by prepare_spy_data.py)
// and publishes three events per row:
//   1. MarketDataEvent        — SPY spot (from put-call parity recovery)
//   2. OptionMidQuoteEvent    — ATM call ("ATM_CALL")
//   3. OptionMidQuoteEvent    — ATM put  ("ATM_PUT")
//
// CSV columns:
//   timestamp_utc, underlying_price, atm_strike, expiry_date,
//   time_to_expiry, call_mid, put_mid
//
// initial_strike() / initial_expiry() — peek at row 0 before run() so
// alpha_main.cpp can construct correctly-struck option objects.
// ============================================================

#include <fstream>
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
        for (auto& r : rows_) {
            lo = std::min(lo, r.underlying);
            hi = std::max(hi, r.underlying);
        }
        std::cout << "[HistoricalChainAdapter] 已载入 " << rows_.size()
                  << " 行真实 SPY 期权数据\n"
                  << "  行情区间: " << rows_.front().timestamp_str
                  << " … " << rows_.back().timestamp_str << "\n"
                  << "  SPY 区间: " << lo << " – " << hi << "\n";
    }

    // ── 前置信息（在 run() 前调用）──────────────────────────────
    double initial_strike() const { return rows_[0].atm_strike; }

    std::chrono::system_clock::time_point initial_expiry() const {
        return parse_date_to_tp(rows_[0].expiry_date);
    }

    // ── 主循环 ─────────────────────────────────────────────────
    void run() {
        std::cout << "[HistoricalChainAdapter] 开始回放…\n";
        for (const auto& r : rows_) {
            auto ts = parse_ts(r.timestamp_str);

            // 1. MarketDataEvent — 标的价格
            bus_->publish(events::MarketDataEvent{ts, r.underlying});

            // 2. ATM 认购期权中间价
            bus_->publish(events::OptionMidQuoteEvent{
                "ATM_CALL",
                r.call_mid,
                r.underlying,
                r.atm_strike,
                r.time_to_expiry,
                /*is_call=*/true,
                ts
            });

            // 3. ATM 认沽期权中间价
            bus_->publish(events::OptionMidQuoteEvent{
                "ATM_PUT",
                r.put_mid,
                r.underlying,
                r.atm_strike,
                r.time_to_expiry,
                /*is_call=*/false,
                ts
            });
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
