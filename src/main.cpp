// ============================================================
// 文件：main.cpp
// 职责：组合根（Composition Root）— 系统唯一的组件连线点。
//
// 这是系统中唯一出现具体类名（而非接口/抽象类）的地方。
// 所有其他模块通过接口和事件总线相互通信，不持有对彼此的直接引用。
//
// 设计模式汇总（本文件是所有模式交汇的地方）：
//
//   观察者模式（Observer Pattern）
//   ─────────────────────────────
//   EventBus 是核心"发布-订阅"基础设施。
//   各组件调用 register_handlers() 或 bus->subscribe<T>()
//   将自身注册为特定事件类型的"观察者"。
//   事件链（因果顺序）：
//     MarketDataEvent → QuoteGeneratedEvent → TradeExecutedEvent → OrderSubmittedEvent
//
//   策略模式（Strategy Pattern）
//   ─────────────────────────────
//   IPricingEngine     → SimplePricingEngine（此处注入，一行可换）
//   IQuoteStrategy     → QuoteEngine（此处注入）
//   IDeltaHedgeStrategy→ DeltaHedger（此处注入）
//
//   适配器模式（Adapter Pattern）
//   ─────────────────────────────
//   MarketDataAdapter 将 CSV 文件（或硬编码数组）翻译为 MarketDataEvent。
//   领域层不感知数据来源。
//
//   命令模式（Command Pattern）
//   ─────────────────────────────
//   DeltaHedger 发布的 OrderSubmittedEvent 是一个命令对象。
//   此处注册的 lambda 是"命令接收者（Receiver）"存根。
//
//   工厂模式（Factory Pattern）
//   ─────────────────────────────
//   InstrumentFactory::make_call/make_put/make_underlying
//   集中管理合约对象的创建，调用方无需知晓构造细节。
// ============================================================

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>

// 事件层
#include "events/EventBus.hpp"
#include "events/Events.hpp"

// 领域层
#include "domain/Instrument.hpp"
#include "domain/InstrumentFactory.hpp"
#include "domain/PositionManager.hpp"
#include "domain/PricingEngine.hpp"

// 基础设施层
#include "infrastructure/MarketDataAdapter.hpp"
#include "infrastructure/ProbabilisticTaker.hpp"

// 应用层
#include "application/QuoteEngine.hpp"
#include "application/DeltaHedger.hpp"

int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n"
              << "║     期权做市商 MVP — 事件驱动仿真系统       ║\n"
              << "╚══════════════════════════════════════════════╝\n\n";

    // --------------------------------------------------------
    // 第一步：创建 EventBus（观察者模式的核心基础设施）
    // 所有组件共享同一个 EventBus 实例，通过 shared_ptr 管理生命周期
    // --------------------------------------------------------
    auto bus = std::make_shared<omm::events::EventBus>();
    std::cout << "[初始化] EventBus 已创建（观察者模式骨架）\n";

    // --------------------------------------------------------
    // 第二步：通过 InstrumentFactory 创建金融工具（工厂模式）
    // 到期日设置为约 30 天后（~30 天期期权）
    // --------------------------------------------------------
    auto expiry = std::chrono::system_clock::now()
                  + std::chrono::hours(30 * 24);

    auto underlying = omm::domain::InstrumentFactory::make_underlying("AAPL");

    // 三个期权合约：
    //   AAPL_150_C — 平值看涨期权（ATM Call），Delta ≈ +0.5（内在价值约为 0）
    //   AAPL_145_C — 实值看涨期权（ITM Call），Delta ≈ +0.5，内在价值 = S - 145
    //   AAPL_155_P — 实值看跌期权（ITM Put），Delta ≈ -0.5，内在价值 = 155 - S
    auto call_150 = omm::domain::InstrumentFactory::make_call("AAPL", 150.0, expiry);
    auto call_145 = omm::domain::InstrumentFactory::make_call("AAPL", 145.0, expiry);
    auto put_155  = omm::domain::InstrumentFactory::make_put ("AAPL", 155.0, expiry);

    std::vector<std::shared_ptr<omm::domain::Option>> options = {
        call_150, call_145, put_155
    };

    std::cout << "[初始化] 工厂模式创建合约:\n";
    std::cout << "  " << underlying->id() << " (" << underlying->type_name() << ")\n";
    for (const auto& opt : options) {
        std::cout << "  " << opt->id() << " (" << opt->type_name() << ")\n";
    }
    std::cout << "\n";

    // --------------------------------------------------------
    // 第三步：创建领域服务
    // 策略模式：pricing_engine 是 IPricingEngine 的具体实现
    // 换模型只需替换 SimplePricingEngine 为其他实现（一行改动）
    // --------------------------------------------------------
    auto pricing_engine = std::make_shared<omm::domain::SimplePricingEngine>();
    auto position_mgr   = std::make_shared<omm::domain::PositionManager>();
    std::cout << "[初始化] 定价策略: SimplePricingEngine（内在价值定价，可替换为 Black-Scholes）\n";

    // --------------------------------------------------------
    // 第四步：将 PositionManager 注册为 TradeExecutedEvent 的订阅者
    // 注意：DeltaHedger 内部也会直接调用 PositionManager（绕过总线，防递归）
    // 此处注册用于记录期权成交后的持仓更新
    // --------------------------------------------------------
    // （PositionManager 的 on_trade_executed 由 DeltaHedger 直接调用，
    //  此处不再二次注册，以避免重复更新持仓）

    // --------------------------------------------------------
    // 第五步：创建应用层策略组件并注册到 EventBus
    // 策略模式：QuoteEngine 和 DeltaHedger 均为可替换策略
    // --------------------------------------------------------
    auto quote_engine = std::make_shared<omm::application::QuoteEngine>(
        bus, pricing_engine, options,
        0.05  // half_spread = $0.05，总价差 = $0.10
    );
    quote_engine->register_handlers(); // 订阅 MarketDataEvent
    std::cout << "[初始化] QuoteEngine 已注册（策略：固定价差 ±$0.05）\n";

    auto delta_hedger = std::make_shared<omm::application::DeltaHedger>(
        bus, position_mgr, pricing_engine, options,
        underlying->id(),
        0.5   // Delta 阈值：|Δ| > 0.5 时触发对冲（MVP 低阈值，便于演示）
    );
    delta_hedger->register_handlers(); // 订阅 TradeExecutedEvent
    std::cout << "[初始化] DeltaHedger 已注册（阈值: ±0.5 Delta）\n";

    // 额外订阅 MarketDataEvent，让 DeltaHedger 始终掌握最新标的价格
    bus->subscribe<omm::events::MarketDataEvent>(
        [&dh = *delta_hedger](const omm::events::MarketDataEvent& evt) {
            dh.update_market_price(evt.underlying_price);
        }
    );

    // --------------------------------------------------------
    // 第六步：创建基础设施组件
    // 适配器模式：MarketDataAdapter 将 CSV/内置数据翻译为领域事件
    // --------------------------------------------------------
    auto taker = std::make_shared<omm::infrastructure::ProbabilisticTaker>(
        bus,
        0.30,  // 30% 成交概率（MVP 提高概率，便于在 20 条 tick 内观察到成交）
        42     // 固定随机种子，确保仿真结果可重现
    );
    taker->register_handlers(); // 订阅 QuoteGeneratedEvent
    std::cout << "[初始化] ProbabilisticTaker 已注册（成交概率: 30%，种子: 42）\n";

    // --------------------------------------------------------
    // 命令模式：注册 OrderSubmittedEvent 的"接收者（Receiver）"
    // 此处为存根日志处理器，代表真实系统中的订单路由模块
    // --------------------------------------------------------
    bus->subscribe<omm::events::OrderSubmittedEvent>(
        [](const omm::events::OrderSubmittedEvent& evt) {
            std::cout << "[订单路由] ★ 收到对冲命令（命令模式）: "
                      << (evt.side == omm::events::Side::Buy ? "买入" : "卖出")
                      << " " << evt.quantity << " 股 " << evt.instrument_id
                      << " [" << (evt.order_type == omm::events::OrderType::Market
                                  ? "市价单" : "限价单") << "]\n";
        }
    );
    std::cout << "[初始化] OrderRouter 存根已注册（命令模式接收者）\n\n";

    // --------------------------------------------------------
    // 第七步：打印事件订阅关系总览（教学用途）
    // --------------------------------------------------------
    std::cout << "┌─────────────────── 事件订阅关系图 ───────────────────┐\n"
              << "│ MarketDataEvent    → QuoteEngine（生成报价）          │\n"
              << "│                   → DeltaHedger（更新标的价格）       │\n"
              << "│ QuoteGeneratedEvent→ ProbabilisticTaker（模拟成交）   │\n"
              << "│ TradeExecutedEvent → DeltaHedger（更新持仓+对冲检查） │\n"
              << "│ OrderSubmittedEvent→ OrderRouter 存根（命令接收者）   │\n"
              << "└────────────────────────────────────────────────────────┘\n\n";

    // --------------------------------------------------------
    // 第八步：运行仿真
    // 适配器模式：MarketDataAdapter 是整个仿真循环的驱动器
    // --------------------------------------------------------
    std::cout << "═══════════════ 仿真开始 ═══════════════\n\n";

    omm::infrastructure::MarketDataAdapter adapter(
        bus,
        "data/market_data.csv" // 优先读取 CSV；文件不存在时自动使用内置数据
    );
    adapter.run();

    // --------------------------------------------------------
    // 第九步：打印最终持仓汇总
    // --------------------------------------------------------
    std::cout << "\n═══════════════ 仿真结束 ═══════════════\n\n";
    position_mgr->print_positions();

    std::cout << "\n[提示] 将 SimplePricingEngine 替换为 BlackScholesPricingEngine\n"
              << "       可获得更精确的期权定价（策略模式 — 零代码改动于其他模块）\n";

    return 0;
}
