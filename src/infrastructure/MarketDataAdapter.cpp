#include "MarketDataAdapter.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

// 模式：适配器模式（Adapter Pattern）
// 本文件实现从原始数据源到领域事件的翻译逻辑。
namespace omm::infrastructure {

MarketDataAdapter::MarketDataAdapter(
    std::shared_ptr<events::EventBus> bus,
    const std::string& csv_path)
    : bus_(std::move(bus))
    , csv_path_(csv_path) {}

void MarketDataAdapter::run() {
    // 优先尝试从 CSV 文件加载，失败则使用硬编码数据
    std::vector<RawTick> ticks = (!csv_path_.empty())
        ? load_from_csv(csv_path_)
        : hardcoded_ticks();

    if (ticks.empty()) {
        // 数据源为空时，回退到硬编码数据
        ticks = hardcoded_ticks();
    }

    std::cout << "[MarketDataAdapter] 开始行情推送，共 " << ticks.size() << " 条 tick\n\n";

    for (const auto& raw : ticks) {
        // --------------------------------------------------
        // 模式：适配器 — 翻译点
        // 输入：RawTick（适配者格式：原始文本 + double）
        // 输出：MarketDataEvent（目标格式：领域事件）
        // --------------------------------------------------
        events::MarketDataEvent evt{
            parse_timestamp(raw.timestamp_str),
            raw.price
        };

        std::cout << "─────────────────────────────────────────\n";
        std::cout << "[行情] " << raw.timestamp_str
                  << "  AAPL = $" << raw.price << "\n";

        // 发布到 EventBus，触发所有订阅了 MarketDataEvent 的组件
        bus_->publish(evt);
    }

    std::cout << "─────────────────────────────────────────\n";
    std::cout << "[MarketDataAdapter] 行情推送完毕\n";
}

std::vector<MarketDataAdapter::RawTick>
MarketDataAdapter::load_from_csv(const std::string& path) const {
    std::vector<RawTick> ticks;
    std::ifstream file(path);

    if (!file.is_open()) {
        // 文件无法打开时，打印警告并返回空向量（调用方会自动回退到硬编码数据）
        std::cerr << "[MarketDataAdapter] 警告：无法打开文件 " << path
                  << "，将使用内置行情数据\n";
        return ticks;
    }

    std::string line;
    std::getline(file, line); // 跳过表头行（timestamp,underlying_price）

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string ts, price_str;
        std::getline(ss, ts, ',');
        std::getline(ss, price_str, ',');
        if (!ts.empty() && !price_str.empty()) {
            ticks.push_back({ts, std::stod(price_str)});
        }
    }
    return ticks;
}

std::vector<MarketDataAdapter::RawTick>
MarketDataAdapter::hardcoded_ticks() const {
    // 20 条内置 AAPL tick 数据，价格围绕 $150 上下波动
    // 这些数据覆盖了实值、平值、虚值三种期权状态，
    // 确保仿真可以触发报价生成和 Delta 对冲
    return {
        {"2024-01-01T09:30:00", 150.00},
        {"2024-01-01T09:30:01", 150.25},
        {"2024-01-01T09:30:02", 150.10},
        {"2024-01-01T09:30:03", 149.80},
        {"2024-01-01T09:30:04", 149.50},
        {"2024-01-01T09:30:05", 149.90},
        {"2024-01-01T09:30:06", 150.30},
        {"2024-01-01T09:30:07", 150.75},
        {"2024-01-01T09:30:08", 151.00},
        {"2024-01-01T09:30:09", 151.20},
        {"2024-01-01T09:30:10", 151.00},
        {"2024-01-01T09:30:11", 150.80},
        {"2024-01-01T09:30:12", 150.50},
        {"2024-01-01T09:30:13", 150.20},
        {"2024-01-01T09:30:14", 149.90},
        {"2024-01-01T09:30:15", 149.70},
        {"2024-01-01T09:30:16", 150.00},
        {"2024-01-01T09:30:17", 150.40},
        {"2024-01-01T09:30:18", 150.60},
        {"2024-01-01T09:30:19", 150.90},
    };
}

events::Timestamp
MarketDataAdapter::parse_timestamp(const std::string& ts_str) const {
    // MVP 简化：基于第一次调用的系统时间加上递增偏移量生成时间戳
    // 生产环境中应解析完整的 ISO-8601 字符串
    static auto base = std::chrono::system_clock::now();
    static int  counter = 0;
    return base + std::chrono::seconds(counter++);
}

} // namespace omm::infrastructure
