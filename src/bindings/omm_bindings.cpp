// ============================================================
// 文件：omm_bindings.cpp
// 职责：pybind11 绑定模块 — 将 C++ 核心层（events / domain / infrastructure）
//       暴露为 Python 扩展模块 "omm_core"。
//
// 架构定位：
//   C++ 层（omm_core.so）：EventBus、领域对象、基础设施适配器
//   Python 层（hybrid_application/）：应用层业务逻辑（QuoteEngine 等）
//
// 编译：
//   cmake -S . -B build -Dpybind11_DIR=<pybind11-cmake-dir>
//   cmake --build build --target omm_core
// ============================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // std::vector, std::unordered_map, std::shared_ptr
#include <pybind11/functional.h> // std::function (CalibrationEngine::solve)
#include <pybind11/chrono.h>     // std::chrono::time_point ↔ datetime.datetime

// ── 领域事件层 ────────────────────────────────────────────────
#include "events/EventBus.hpp"
#include "events/Events.hpp"

// ── 领域层 ───────────────────────────────────────────────────
#include "domain/Instrument.hpp"
#include "domain/InstrumentFactory.hpp"
#include "domain/PricingEngine.hpp"
#include "domain/PositionManager.hpp"
#include "domain/RiskMetrics.hpp"
#include "domain/PortfolioAggregate.hpp"
#include "domain/RiskPolicy.hpp"
#include "domain/CalibrationEngine.hpp"

// ── 基础设施层 ────────────────────────────────────────────────
#include "infrastructure/MarketDataAdapter.hpp"
#include "infrastructure/ProbabilisticTaker.hpp"
#include "infrastructure/ParameterStore.hpp"

namespace py = pybind11;

using namespace omm::events;
using namespace omm::domain;
using namespace omm::infrastructure;

// ── IRiskPolicy trampoline（支持 Python 子类化，虽 MVP 中暂不使用）────
class PyIRiskPolicy : public IRiskPolicy {
public:
    std::vector<RiskControlEvent> evaluate(
        const std::string& account_id,
        const RiskMetrics& metrics
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<RiskControlEvent>,
            IRiskPolicy,
            evaluate,
            account_id, metrics
        );
    }
};

// ── IPricingEngine trampoline ─────────────────────────────────
class PyIPricingEngine : public IPricingEngine {
public:
    PriceResult price(
        const Option& option,
        double        underlying_price
    ) const override {
        PYBIND11_OVERRIDE_PURE(
            PriceResult,
            IPricingEngine,
            price,
            option, underlying_price
        );
    }
};

// ════════════════════════════════════════════════════════════
// 模块定义
// ════════════════════════════════════════════════════════════
PYBIND11_MODULE(omm_core, m) {
    m.doc() = "omm_core — C++ 期权做市商核心层 Python 绑定（pybind11）";

    // ────────────────────────────────────────────────────────
    // §1  枚举类型
    // ────────────────────────────────────────────────────────
    py::enum_<Side>(m, "Side")
        .value("Buy",  Side::Buy)
        .value("Sell", Side::Sell)
        .export_values();

    py::enum_<OrderType>(m, "OrderType")
        .value("Market", OrderType::Market)
        .value("Limit",  OrderType::Limit)
        .export_values();

    py::enum_<RiskAction>(m, "RiskAction")
        .value("BlockOrders",  RiskAction::BlockOrders)
        .value("CancelOrders", RiskAction::CancelOrders)
        .value("ReduceOnly",   RiskAction::ReduceOnly)
        .export_values();

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put",  OptionType::Put)
        .export_values();

    // ────────────────────────────────────────────────────────
    // §2  领域事件（纯数据，可从 Python 创建 + 读写字段）
    // ────────────────────────────────────────────────────────
    py::class_<MarketDataEvent>(m, "MarketDataEvent")
        .def(py::init<>())
        .def_readwrite("timestamp",        &MarketDataEvent::timestamp)
        .def_readwrite("underlying_price", &MarketDataEvent::underlying_price);

    py::class_<QuoteGeneratedEvent>(m, "QuoteGeneratedEvent")
        .def(py::init<>())
        .def_readwrite("instrument_id", &QuoteGeneratedEvent::instrument_id)
        .def_readwrite("bid_price",     &QuoteGeneratedEvent::bid_price)
        .def_readwrite("ask_price",     &QuoteGeneratedEvent::ask_price)
        .def_readwrite("timestamp",     &QuoteGeneratedEvent::timestamp);

    py::class_<TradeExecutedEvent>(m, "TradeExecutedEvent")
        .def(py::init<>())
        .def_readwrite("instrument_id", &TradeExecutedEvent::instrument_id)
        .def_readwrite("side",          &TradeExecutedEvent::side)
        .def_readwrite("price",         &TradeExecutedEvent::price)
        .def_readwrite("quantity",      &TradeExecutedEvent::quantity)
        .def_readwrite("timestamp",     &TradeExecutedEvent::timestamp);

    py::class_<OrderSubmittedEvent>(m, "OrderSubmittedEvent")
        .def(py::init<>())
        .def_readwrite("instrument_id", &OrderSubmittedEvent::instrument_id)
        .def_readwrite("side",          &OrderSubmittedEvent::side)
        .def_readwrite("quantity",      &OrderSubmittedEvent::quantity)
        .def_readwrite("order_type",    &OrderSubmittedEvent::order_type);

    py::class_<RiskControlEvent>(m, "RiskControlEvent")
        .def(py::init<>())
        .def_readwrite("account_id", &RiskControlEvent::account_id)
        .def_readwrite("action",     &RiskControlEvent::action)
        .def_readwrite("reason",     &RiskControlEvent::reason);

    py::class_<RiskAlertEvent>(m, "RiskAlertEvent")
        .def(py::init<>())
        .def_readwrite("account_id",   &RiskAlertEvent::account_id)
        .def_readwrite("metric_name",  &RiskAlertEvent::metric_name)
        .def_readwrite("value",        &RiskAlertEvent::value)
        .def_readwrite("limit",        &RiskAlertEvent::limit);

    py::class_<ParamUpdateEvent>(m, "ParamUpdateEvent")
        .def(py::init<>())
        .def_readwrite("model_id",   &ParamUpdateEvent::model_id)
        .def_readwrite("params",     &ParamUpdateEvent::params)
        .def_readwrite("updated_at", &ParamUpdateEvent::updated_at);

    // ────────────────────────────────────────────────────────
    // §3  事件总线（EventBus）
    //
    // C++ 模板接口无法直接导出，改为每种事件类型各自提供
    // subscribe_*/publish_* 具名包装方法。
    // C++ 基础设施组件（MarketDataAdapter 等）使用 C++ 模板直接调用，
    // Python 应用层通过此包装 API 订阅/发布。
    // ────────────────────────────────────────────────────────
    py::class_<EventBus, std::shared_ptr<EventBus>>(m, "EventBus")
        .def(py::init<>())
        .def("clear", &EventBus::clear)

        // ── 订阅接口（Python 应用层使用）─────────────────
        .def("subscribe_market_data", [](EventBus& bus, py::function fn) {
            bus.subscribe<MarketDataEvent>(
                [fn](const MarketDataEvent& e) { fn(e); });
        })
        .def("subscribe_trade_executed", [](EventBus& bus, py::function fn) {
            bus.subscribe<TradeExecutedEvent>(
                [fn](const TradeExecutedEvent& e) { fn(e); });
        })
        .def("subscribe_quote_generated", [](EventBus& bus, py::function fn) {
            bus.subscribe<QuoteGeneratedEvent>(
                [fn](const QuoteGeneratedEvent& e) { fn(e); });
        })
        .def("subscribe_risk_control", [](EventBus& bus, py::function fn) {
            bus.subscribe<RiskControlEvent>(
                [fn](const RiskControlEvent& e) { fn(e); });
        })
        .def("subscribe_risk_alert", [](EventBus& bus, py::function fn) {
            bus.subscribe<RiskAlertEvent>(
                [fn](const RiskAlertEvent& e) { fn(e); });
        })
        .def("subscribe_param_update", [](EventBus& bus, py::function fn) {
            bus.subscribe<ParamUpdateEvent>(
                [fn](const ParamUpdateEvent& e) { fn(e); });
        })
        .def("subscribe_order_submitted", [](EventBus& bus, py::function fn) {
            bus.subscribe<OrderSubmittedEvent>(
                [fn](const OrderSubmittedEvent& e) { fn(e); });
        })

        // ── 发布接口（Python 应用层使用）─────────────────
        .def("publish_quote_generated", [](EventBus& bus, const QuoteGeneratedEvent& e) {
            bus.publish(e);
        })
        .def("publish_trade_executed", [](EventBus& bus, const TradeExecutedEvent& e) {
            bus.publish(e);
        })
        .def("publish_order_submitted", [](EventBus& bus, const OrderSubmittedEvent& e) {
            bus.publish(e);
        })
        .def("publish_risk_control", [](EventBus& bus, const RiskControlEvent& e) {
            bus.publish(e);
        })
        .def("publish_risk_alert", [](EventBus& bus, const RiskAlertEvent& e) {
            bus.publish(e);
        })
        .def("publish_param_update", [](EventBus& bus, const ParamUpdateEvent& e) {
            bus.publish(e);
        });

    // ────────────────────────────────────────────────────────
    // §4  金融工具（Instrument 层级）
    // ────────────────────────────────────────────────────────
    py::class_<Instrument, std::shared_ptr<Instrument>>(m, "Instrument")
        .def_property_readonly("id", [](const Instrument& i) {
            return i.id();  // const std::string& → Python str
        })
        .def("type_name", &Instrument::type_name);

    py::class_<Underlying, Instrument, std::shared_ptr<Underlying>>(m, "Underlying")
        .def("type_name", &Underlying::type_name);

    py::class_<Option, Instrument, std::shared_ptr<Option>>(m, "Option")
        .def("type_name", &Option::type_name)
        .def_property_readonly("underlying_id", [](const Option& o) {
            return o.underlying_id();
        })
        .def_property_readonly("strike",      &Option::strike)
        .def_property_readonly("expiry",      &Option::expiry)      // → datetime
        .def_property_readonly("option_type", &Option::option_type);

    // InstrumentFactory — 静态工厂方法
    py::class_<InstrumentFactory>(m, "InstrumentFactory")
        .def_static("make_underlying", &InstrumentFactory::make_underlying,
            py::arg("id"))
        .def_static("make_call", &InstrumentFactory::make_call,
            py::arg("underlying_id"), py::arg("strike"), py::arg("expiry"))
        .def_static("make_put", &InstrumentFactory::make_put,
            py::arg("underlying_id"), py::arg("strike"), py::arg("expiry"));

    // ────────────────────────────────────────────────────────
    // §5  定价引擎
    // ────────────────────────────────────────────────────────
    py::class_<PriceResult>(m, "PriceResult")
        .def(py::init<>())
        .def_readwrite("theo",  &PriceResult::theo)
        .def_readwrite("delta", &PriceResult::delta);

    py::class_<IPricingEngine, PyIPricingEngine, std::shared_ptr<IPricingEngine>>(
            m, "IPricingEngine")
        .def(py::init<>())
        .def("price", &IPricingEngine::price,
            py::arg("option"), py::arg("underlying_price"));

    py::class_<SimplePricingEngine, IPricingEngine,
               std::shared_ptr<SimplePricingEngine>>(m, "SimplePricingEngine")
        .def(py::init<>())
        .def("price", &SimplePricingEngine::price,
            py::arg("option"), py::arg("underlying_price"));

    py::class_<BlackScholesPricingEngine, IPricingEngine,
               std::shared_ptr<BlackScholesPricingEngine>>(m, "BlackScholesPricingEngine")
        .def(py::init<double, double>(),
            py::arg("vol") = 0.20, py::arg("r") = 0.05)
        .def("price",   &BlackScholesPricingEngine::price,
            py::arg("option"), py::arg("underlying_price"))
        .def("set_vol", &BlackScholesPricingEngine::set_vol, py::arg("vol"))
        .def("get_vol", &BlackScholesPricingEngine::get_vol);

    // ────────────────────────────────────────────────────────
    // §6  持仓管理
    // ────────────────────────────────────────────────────────
    py::class_<PositionManager, std::shared_ptr<PositionManager>>(m, "PositionManager")
        .def(py::init<>())
        .def("on_trade_executed",     &PositionManager::on_trade_executed,
            py::arg("event"))
        .def("get_position",          &PositionManager::get_position,
            py::arg("instrument_id"))
        .def("compute_portfolio_delta",
            &PositionManager::compute_portfolio_delta,
            py::arg("deltas"))
        .def("print_positions",       &PositionManager::print_positions);

    // ────────────────────────────────────────────────────────
    // §7  投资组合聚合根 + 风险指标
    // ────────────────────────────────────────────────────────
    py::class_<RiskMetrics>(m, "RiskMetrics")
        .def(py::init<>())
        .def_readwrite("realized_pnl",      &RiskMetrics::realized_pnl)
        .def_readwrite("unrealized_pnl",    &RiskMetrics::unrealized_pnl)
        .def_readwrite("delta",             &RiskMetrics::delta)
        .def_readwrite("gamma",             &RiskMetrics::gamma)
        .def_readwrite("vega",              &RiskMetrics::vega)
        .def_readwrite("theta",             &RiskMetrics::theta)
        .def_readwrite("var_1d",            &RiskMetrics::var_1d)
        .def_readwrite("intraday_drawdown", &RiskMetrics::intraday_drawdown);

    py::class_<PortfolioAggregate>(m, "PortfolioAggregate")
        .def(py::init<std::string, std::vector<std::shared_ptr<Option>>>(),
            py::arg("account_id"), py::arg("options"))
        .def("applyTrade",     &PortfolioAggregate::applyTrade,    py::arg("event"))
        .def("markToMarket",   &PortfolioAggregate::markToMarket,
            py::arg("engine"), py::arg("underlying_price"))
        .def("computeMetrics", &PortfolioAggregate::computeMetrics,
            py::arg("engine"), py::arg("underlying_price"))
        .def("get_position",   &PortfolioAggregate::get_position,
            py::arg("instrument_id"));

    // ────────────────────────────────────────────────────────
    // §8  风险策略
    // ────────────────────────────────────────────────────────
    py::class_<IRiskPolicy, PyIRiskPolicy, std::shared_ptr<IRiskPolicy>>(
            m, "IRiskPolicy")
        .def(py::init<>())
        .def("evaluate", &IRiskPolicy::evaluate,
            py::arg("account_id"), py::arg("metrics"));

    py::class_<SimpleRiskPolicy, IRiskPolicy,
               std::shared_ptr<SimpleRiskPolicy>>(m, "SimpleRiskPolicy")
        .def(py::init<double, double, double>(),
            py::arg("loss_limit")     = 1e6,
            py::arg("delta_limit")    = 10000.0,
            py::arg("drawdown_limit") = 5e5)
        .def("evaluate", &SimpleRiskPolicy::evaluate,
            py::arg("account_id"), py::arg("metrics"));

    // ────────────────────────────────────────────────────────
    // §9  校准引擎（黄金分割搜索）
    // ────────────────────────────────────────────────────────
    py::class_<CalibrationEngine, std::shared_ptr<CalibrationEngine>>(
            m, "CalibrationEngine")
        .def(py::init<>())
        .def("observe", &CalibrationEngine::observe,
            py::arg("market_price"), py::arg("model_price"))
        .def("solve",   &CalibrationEngine::solve,
            py::arg("lo"), py::arg("hi"), py::arg("loss_fn"),
            py::arg("tol") = 1e-6)
        .def("mse",                &CalibrationEngine::mse)
        .def("observation_count",  &CalibrationEngine::observation_count);

    // ────────────────────────────────────────────────────────
    // §10 基础设施层（适配器 / 模拟成交 / 参数仓库）
    // ────────────────────────────────────────────────────────
    py::class_<MarketDataAdapter>(m, "MarketDataAdapter")
        .def(py::init<std::shared_ptr<EventBus>, std::string>(),
            py::arg("bus"), py::arg("csv_path") = "")
        .def("run", &MarketDataAdapter::run);

    py::class_<ProbabilisticTaker>(m, "ProbabilisticTaker")
        .def(py::init<std::shared_ptr<EventBus>, double, unsigned int>(),
            py::arg("bus"),
            py::arg("trade_probability") = 0.10,
            py::arg("seed")              = 42u)
        .def("register_handlers", &ProbabilisticTaker::register_handlers);

    py::class_<ParameterStore, std::shared_ptr<ParameterStore>>(m, "ParameterStore")
        .def(py::init<std::shared_ptr<EventBus>>(), py::arg("bus"))
        .def("subscribe_handlers", &ParameterStore::subscribe_handlers)
        .def("get_params",         &ParameterStore::get_params,
            py::arg("model_id"))
        .def("print_all",          &ParameterStore::print_all);
}
