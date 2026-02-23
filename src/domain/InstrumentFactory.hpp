#pragma once
#include "Instrument.hpp"
#include <memory>
#include <string>
#include <chrono>

// ============================================================
// 文件：InstrumentFactory.hpp
// 职责：提供创建金融工具对象的工厂方法。
//
// 模式：工厂模式（Factory Pattern）
//   集中管理所有 Instrument 子类的创建逻辑，使调用方：
//   1. 无需直接使用 new 或知晓具体子类构造参数
//   2. 合约 ID 字符串的生成规则在一处定义，保证全系统一致
//   3. 若未来新增 FutureOption 等子类，只需扩展此工厂，其他代码无需改动
//
// 所有方法均为静态方法，工厂本身无状态。
// ============================================================

namespace omm::domain {

class InstrumentFactory {
public:
    // ----------------------------------------------------------
    // make_underlying() — 创建标的资产
    //
    // 参数：
    //   id — 资产代码，如 "AAPL"
    // 返回：
    //   std::shared_ptr<Underlying>（共享所有权，供多个组件持有）
    // ----------------------------------------------------------
    static std::shared_ptr<Underlying> make_underlying(
        const std::string& id
    );

    // ----------------------------------------------------------
    // make_call() — 创建看涨期权（Call Option）
    //
    // 自动生成合约 ID，格式："{underlying_id}_{strike}_C_{YYYYMMDD}"
    // 例：AAPL_150_C_20240201
    // ----------------------------------------------------------
    static std::shared_ptr<Option> make_call(
        const std::string& underlying_id,
        double             strike,
        std::chrono::system_clock::time_point expiry
    );

    // ----------------------------------------------------------
    // make_put() — 创建看跌期权（Put Option）
    //
    // 自动生成合约 ID，格式："{underlying_id}_{strike}_P_{YYYYMMDD}"
    // 例：AAPL_148_P_20240201
    // ----------------------------------------------------------
    static std::shared_ptr<Option> make_put(
        const std::string& underlying_id,
        double             strike,
        std::chrono::system_clock::time_point expiry
    );

private:
    // 内部工具方法：根据标的、行权价、期权类型和到期日生成标准合约 ID 字符串
    static std::string make_option_id(
        const std::string& underlying_id,
        double             strike,
        OptionType         option_type,
        const std::chrono::system_clock::time_point& expiry
    );
};

} // namespace omm::domain
