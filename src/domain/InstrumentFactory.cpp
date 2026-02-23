#include "InstrumentFactory.hpp"
#include <sstream>
#include <iomanip>
#include <ctime>

// 模式：工厂模式（Factory Pattern）
// 本文件集中实现所有 Instrument 子类的构造逻辑
namespace omm::domain {

std::shared_ptr<Underlying> InstrumentFactory::make_underlying(
    const std::string& id) {
    // 直接委托构造，shared_ptr 管理生命周期
    return std::make_shared<Underlying>(id);
}

std::shared_ptr<Option> InstrumentFactory::make_call(
    const std::string& underlying_id,
    double strike,
    std::chrono::system_clock::time_point expiry) {
    // 先生成标准 ID，再构造 Option 对象
    auto id = make_option_id(underlying_id, strike, OptionType::Call, expiry);
    return std::make_shared<Option>(
        std::move(id), underlying_id, strike, expiry, OptionType::Call);
}

std::shared_ptr<Option> InstrumentFactory::make_put(
    const std::string& underlying_id,
    double strike,
    std::chrono::system_clock::time_point expiry) {
    auto id = make_option_id(underlying_id, strike, OptionType::Put, expiry);
    return std::make_shared<Option>(
        std::move(id), underlying_id, strike, expiry, OptionType::Put);
}

std::string InstrumentFactory::make_option_id(
    const std::string& underlying_id,
    double strike,
    OptionType option_type,
    const std::chrono::system_clock::time_point& expiry) {
    // 将时间点转换为可读日期字符串（格式：YYYYMMDD）
    std::time_t t = std::chrono::system_clock::to_time_t(expiry);
    std::tm tm_val{};
#ifdef _WIN32
    // Windows 使用 gmtime_s（参数顺序与 POSIX 相反）
    gmtime_s(&tm_val, &t);
#else
    // POSIX（Linux/macOS）使用 gmtime_r
    gmtime_r(&t, &tm_val);
#endif

    // 拼装合约 ID："{标的资产}_{行权价}_{C/P}_{YYYYMMDD}"
    // 例：AAPL_150_C_20240201
    std::ostringstream oss;
    oss << underlying_id
        << "_" << static_cast<int>(strike)
        << "_" << (option_type == OptionType::Call ? "C" : "P")
        << "_" << std::put_time(&tm_val, "%Y%m%d");
    return oss.str();
}

} // namespace omm::domain
