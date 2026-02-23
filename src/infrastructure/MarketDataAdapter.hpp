#pragma once
#include <string>
#include <vector>
#include <memory>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"

// ============================================================
// 文件：MarketDataAdapter.hpp
// 职责：将原始行情数据源（CSV 文件或硬编码数据）适配为领域事件。
//
// 模式：适配器模式（Adapter Pattern）
//   "适配者（Adaptee）"：CSV 文件中的原始文本行 / 内置数组
//   "目标接口（Target）"：EventBus 上的 MarketDataEvent
//   "适配器（Adapter）"：MarketDataAdapter::run() 执行翻译
//
//   优势：
//   - 领域层（QuoteEngine、DeltaHedger）完全不知道数据来自 CSV 还是 WebSocket。
//   - 替换数据源只需重写此类的 run()，其他所有代码零改动。
//   - 这正是"依赖倒置原则（DIP）"：高层模块依赖抽象事件，
//     不依赖具体的文件格式或网络协议。
// ============================================================

namespace omm::infrastructure {

class MarketDataAdapter {
public:
    // 构造函数注入 EventBus 和可选的 CSV 路径。
    // 若 csv_path 为空或文件无法打开，自动回退到内置硬编码数据。
    explicit MarketDataAdapter(
        std::shared_ptr<events::EventBus> bus,
        const std::string& csv_path = ""
    );

    // ----------------------------------------------------------
    // run() — 驱动仿真循环
    //
    // 逐条读取行情 tick，将每条原始数据翻译为 MarketDataEvent，
    // 并发布到 EventBus。此方法是整个仿真循环的驱动点。
    //
    // 模式：适配器 — 翻译发生在此方法内部。
    // ----------------------------------------------------------
    void run();

private:
    // 原始 tick 数据结构（适配者格式）
    struct RawTick {
        std::string timestamp_str; // 原始时间戳字符串
        double      price;         // 标的资产价格
    };

    // 从 CSV 文件加载原始 tick 数据
    std::vector<RawTick> load_from_csv(const std::string& path) const;

    // 回退方案：硬编码的 tick 数据（不依赖文件 I/O，确保仿真始终可运行）
    std::vector<RawTick> hardcoded_ticks() const;

    // 将时间戳字符串转换为 Timestamp 类型（MVP 简化实现）
    events::Timestamp parse_timestamp(const std::string& ts_str) const;

    std::shared_ptr<events::EventBus> bus_;      // 事件总线（注入）
    std::string                       csv_path_; // CSV 文件路径
};

} // namespace omm::infrastructure
