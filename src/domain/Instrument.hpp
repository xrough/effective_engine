#pragma once
#include <string>
#include <chrono>

// ============================================================
// 文件：Instrument.hpp
// 职责：定义金融工具（Instrument）的领域模型。
//
// 类层次结构：
//   Instrument（抽象基类）
//   ├── Underlying（标的资产，如 AAPL 股票）
//   └── Option（香草欧式期权，含行权价、到期日、看涨/看跌方向）
//
// 模式：工厂模式（Factory）所创建的对象类型
//   具体实例由 InstrumentFactory 创建，调用方不直接使用 new。
// ============================================================

namespace omm::domain {

// 期权类型：看涨（Call）或看跌（Put）
enum class OptionType {
    Call, // 看涨期权：赋予持有者以行权价买入标的资产的权利
    Put   // 看跌期权：赋予持有者以行权价卖出标的资产的权利
};

// ============================================================
// Instrument — 抽象基类
// 代表任何可交易的金融工具。
//
// instrument_id 是系统中唯一标识一个工具的字符串键，
// 在事件、持仓管理、报价引擎中统一使用此 ID。
// ============================================================
class Instrument {
public:
    explicit Instrument(std::string id);
    virtual ~Instrument() = default;

    // 返回工具的唯一标识符（只读）
    const std::string& id() const;

    // 每个子类声明自己的工具类型名称（用于日志输出）
    virtual std::string type_name() const = 0;

protected:
    std::string id_; // 合约唯一标识符
};

// ============================================================
// Underlying — 标的资产（现货股票）
// 例如：Underlying("AAPL") 代表苹果公司股票。
//
// Delta 恒等于 1.0：标的资产价格每变动 $1，头寸价值同步变动 $1。
// ============================================================
class Underlying final : public Instrument {
public:
    explicit Underlying(std::string id);
    std::string type_name() const override;
};

// ============================================================
// Option — 香草欧式期权
//
// 合约 ID 命名规范（由 InstrumentFactory 生成）：
//   格式："{标的资产}_{行权价}_{C/P}_{到期日YYYYMMDD}"
//   示例：AAPL_150_C_20240201（苹果 150 行权价看涨期权，2024-02-01 到期）
//
// 字段说明：
//   underlying_id — 关联的标的资产 ID（如 "AAPL"）
//   strike        — 行权价（美元）
//   expiry        — 到期时间点
//   option_type   — Call（看涨）或 Put（看跌）
// ============================================================
class Option final : public Instrument {
public:
    Option(std::string id,
           std::string underlying_id,
           double      strike,
           std::chrono::system_clock::time_point expiry,
           OptionType  option_type);

    std::string type_name() const override;

    // 访问器（只读，保持值对象语义）
    const std::string& underlying_id() const;
    double             strike()         const;
    std::chrono::system_clock::time_point expiry() const;
    OptionType         option_type()    const;

private:
    std::string                            underlying_id_; // 标的资产 ID
    double                                 strike_;        // 行权价（美元）
    std::chrono::system_clock::time_point  expiry_;        // 到期时间
    OptionType                             option_type_;   // 看涨或看跌
};

} // namespace omm::domain
