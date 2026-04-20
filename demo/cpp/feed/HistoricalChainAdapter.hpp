#pragma once
// ============================================================
// File: HistoricalChainAdapter.hpp  (demo/cpp/)
// Role: real-data counterpart to SyntheticOptionFeed + MarketDataAdapter.
//
// Reads demo/data/spy_chain_panel.csv (produced by prepare_spy_data.py)
// and publishes four events per row:
//   1. MarketDataEvent        — SPY spot (from put-call parity recovery)
//   2. OptionMidQuoteEvent    — ATM call ("ATM_CALL") with bid/ask
//   3. OptionMidQuoteEvent    — ATM put  ("ATM_PUT")  with bid/ask
//   4. SmileSnapshotEvent     — VIX variance swap rate + SSVI parameters
//
// CSV column layout (0-indexed):
//   0=timestamp_utc  1=underlying_price  2=atm_strike  3=expiry_date
//   4=time_to_expiry 5=call_mid          6=put_mid
//   7=call_bid       8=call_ask          9=put_bid      10=put_ask
//   11=atm_iv        12=call25d_strike   13=call25d_mid 14=call25d_bid
//   15=call25d_ask   16=put25d_strike    17=put25d_mid  18=put25d_bid
//   19=put25d_ask    20=rr25_iv          21=bf25_iv
//   22=vix_varswap   23=ssvi_rho         24=ssvi_phi    25=rv5_ann
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
        const std::string& csv_path,
        const std::string& start_date = "",  // "YYYY-MM-DD" inclusive, "" = no filter
        const std::string& end_date   = ""   // "YYYY-MM-DD" inclusive, "" = no filter
    ) : bus_(std::move(bus))
      , start_date_(start_date)
      , end_date_(end_date)
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

            // 4. SmileSnapshotEvent — VIX variance + SSVI fit (real data only)
            events::SmileSnapshotEvent smile;
            smile.vix_varswap    = r.vix_varswap;
            smile.atm_iv         = r.atm_iv;
            smile.rr25_iv        = r.rr25_iv;
            smile.bf25_iv        = r.bf25_iv;
            smile.ssvi_theta     = (r.atm_iv > 0 && r.time_to_expiry > 0)
                                   ? r.atm_iv * r.atm_iv * r.time_to_expiry : 0.0;
            smile.ssvi_rho       = r.ssvi_rho;
            smile.ssvi_phi       = r.ssvi_phi;
            smile.time_to_expiry = r.time_to_expiry;
            smile.underlying     = r.underlying;
            smile.has_vix        = (r.vix_varswap > 0.0);
            smile.has_ssvi       = (r.ssvi_phi > 0.0 && std::abs(r.ssvi_rho) < 1.0);
            smile.timestamp      = ts;
            bus_->publish(smile);

            // 5. Per-row callback — invoked after all events for this bar.
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
        double      call_bid    = 0.0;
        double      call_ask    = 0.0;
        double      put_bid     = 0.0;
        double      put_ask     = 0.0;
        double      atm_iv      = 0.0;  // col 11 — BS IV at ATM
        double      rr25_iv     = 0.0;  // col 20 — risk reversal IV
        double      bf25_iv     = 0.0;  // col 21 — butterfly IV
        double      vix_varswap = 0.0;  // col 22 — VIX-style σ²_varswap
        double      ssvi_rho    = 0.0;  // col 23 — SSVI skew ρ
        double      ssvi_phi    = 0.0;  // col 24 — SSVI smoothing φ
        double      rv5_ann     = 0.0;  // col 25 — realised vol 5-bar
        // date portion (YYYY-MM-DD) for session boundary detection
        std::string date_str() const {
            return timestamp_str.size() >= 10 ? timestamp_str.substr(0, 10) : "";
        }
    };

    std::shared_ptr<events::EventBus> bus_;
    std::vector<Row>                  rows_;
    std::string                       start_date_;  // "YYYY-MM-DD" or "" = no filter
    std::string                       end_date_;    // "YYYY-MM-DD" or "" = no filter

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

            // Date-range filtering (YYYY-MM-DD prefix comparison)
            if (!start_date_.empty() || !end_date_.empty()) {
                std::string row_date = r.timestamp_str.size() >= 10
                    ? r.timestamp_str.substr(0, 10) : "";
                if (!start_date_.empty() && row_date < start_date_) continue;
                if (!end_date_.empty()   && row_date > end_date_)   break;  // sorted → early exit
            }

            r.underlying     = std::stod(cols[1]);
            r.atm_strike     = std::stod(cols[2]);
            r.expiry_date    = cols[3];
            r.time_to_expiry = std::stod(cols[4]);
            r.call_mid       = std::stod(cols[5]);
            r.put_mid        = std::stod(cols[6]);
            // Extended columns — present in spy_chain_panel.csv; backward-compat fallback for legacy CSV
            auto safe_d = [&](int col, double def = 0.0) -> double {
                if ((int)cols.size() <= col || cols[col].empty() || cols[col] == "nan")
                    return def;
                try { return std::stod(cols[col]); } catch (...) { return def; }
            };

            if (cols.size() >= 12) {
                r.call_bid = safe_d(7);
                r.call_ask = safe_d(8);
                r.put_bid  = safe_d(9);
                r.put_ask  = safe_d(10);
                r.atm_iv   = safe_d(11);
            } else {
                r.call_bid = r.call_mid - 0.01;
                r.call_ask = r.call_mid + 0.01;
                r.put_bid  = r.put_mid  - 0.01;
                r.put_ask  = r.put_mid  + 0.01;
            }
            r.rr25_iv     = safe_d(20);
            r.bf25_iv     = safe_d(21);
            r.vix_varswap = safe_d(22);
            r.ssvi_rho    = safe_d(23);
            r.ssvi_phi    = safe_d(24);
            r.rv5_ann     = safe_d(25);
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
