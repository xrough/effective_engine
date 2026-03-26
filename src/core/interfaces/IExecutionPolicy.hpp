#pragma once
#include "IEntryPolicy.hpp"

// ============================================================
// 文件：IExecutionPolicy.hpp
// 职责：执行策略纯接口 — 定义"如何提交一笔订单请求"。
//
// 设计原则：
//   - submit() 是命令式 API：调用方（OrderEngine）说"提交这笔订单"，
//     不关心底层机制
//   - 具体实现决定机制：
//       SimulatedExecution — 在内部发布 OrderRequestEvent，
//                            由 BrokerAdapter 模拟成交
//       LiveExecution      — 直接调用 FIX/gRPC 券商适配器
//   - 不含 EventBus 引用（实现可以持有，但接口不暴露）
// ============================================================

namespace omm::core {

class IExecutionPolicy {
public:
    virtual ~IExecutionPolicy() = default;

    // submit() — 提交订单请求；实现负责路由到券商或模拟引擎
    virtual void submit(const OrderRequest& request) = 0;
};

} // namespace omm::core
