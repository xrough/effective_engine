#include "Instrument.hpp"
#include <utility> // std::move

namespace omm::domain {

// ============================================================
// Instrument（基类）实现
// ============================================================

Instrument::Instrument(std::string id)
    : id_(std::move(id)) {}

const std::string& Instrument::id() const {
    return id_;
}

// ============================================================
// Underlying（标的资产）实现
// ============================================================

Underlying::Underlying(std::string id)
    : Instrument(std::move(id)) {}

std::string Underlying::type_name() const {
    // 返回可读的类型名称，用于日志输出
    return "Underlying";
}

// ============================================================
// Option（欧式期权）实现
// ============================================================

Option::Option(std::string id,
               std::string underlying_id,
               double strike,
               std::chrono::system_clock::time_point expiry,
               OptionType option_type)
    : Instrument(std::move(id))
    , underlying_id_(std::move(underlying_id))
    , strike_(strike)
    , expiry_(expiry)
    , option_type_(option_type) {}

std::string Option::type_name() const {
    // 根据期权方向返回类型名称
    return option_type_ == OptionType::Call ? "Call" : "Put";
}

const std::string& Option::underlying_id() const { return underlying_id_; }
double             Option::strike()         const { return strike_; }
std::chrono::system_clock::time_point Option::expiry() const { return expiry_; }
OptionType         Option::option_type()    const { return option_type_; }

} // namespace omm::domain
