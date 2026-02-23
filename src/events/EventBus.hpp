#pragma once
#include <functional>
#include <typeindex>
#include <unordered_map>
#include <vector>
#include <any>

// ============================================================
// 文件：EventBus.hpp
// 职责：实现同步事件总线（Event Bus）。
//
// 模式：观察者模式（Observer Pattern）
//   - EventBus 是"发布者/主题（Subject）"
//   - 调用 subscribe<T>() 的组件是"观察者（Observer）"
//   - 调用 publish<T>() 时，总线将事件同步分发给所有已注册的观察者
//
// 设计选择：std::type_index + std::any（无需基类，无需 dynamic_cast）
//   - subscribe<T>() 将类型化处理器包装为类型擦除的 lambda，存入映射表
//   - publish<T>()   将事件包装为 std::any，依次调用已注册的处理器
//   - 类型安全在调用点（subscribe/publish）由模板参数保证
//   - 内部映射表无需感知具体事件类型，保持对领域层零依赖
//
// 注意：本实现为单线程同步版本，publish() 调用是阻塞的，
//       直到所有处理器执行完毕后才返回。
//       不支持多线程并发调用，无需加锁。
// ============================================================

namespace omm::events {

class EventBus {
public:
    // ----------------------------------------------------------
    // subscribe<T>() — 注册事件处理器
    //
    // 模式：观察者 — 观察者调用此方法将自身注册为 T 类型事件的接收方。
    //
    // 参数：
    //   handler — 可调用对象，接受 const T& 类型的事件实参
    //
    // 实现细节：
    //   将 handler 包裹在一个捕获了类型信息的 lambda 中，
    //   lambda 接受 const std::any&，内部调用 std::any_cast<const T&>
    //   恢复原始类型后再调用 handler。
    //   这样内部存储只需要 std::function<void(const std::any&)>，
    //   无需感知具体事件类型 T。
    // ----------------------------------------------------------
    template<typename T>
    void subscribe(std::function<void(const T&)> handler) {
        auto key = std::type_index(typeid(T));
        // 将类型化处理器包装为类型擦除的 lambda，统一存储
        handlers_[key].emplace_back(
            [h = std::move(handler)](const std::any& event) {
                h(std::any_cast<const T&>(event));
            }
        );
    }

    // ----------------------------------------------------------
    // publish<T>() — 发布事件，通知所有观察者
    //
    // 模式：观察者 — 发布者（Subject）调用此方法广播事件。
    //
    // 参数：
    //   event — 待发布的事件实例（以 const 引用传入，内部包装为 std::any）
    //
    // 行为：
    //   同步依次调用所有已注册 T 类型的处理器，按注册顺序执行。
    //   若无处理器，静默忽略（不抛出异常）。
    // ----------------------------------------------------------
    template<typename T>
    void publish(const T& event) {
        auto key = std::type_index(typeid(T));
        auto it = handlers_.find(key);
        if (it == handlers_.end()) {
            // 无订阅者，静默跳过
            return;
        }
        // 将事件包装为 std::any，避免多次拷贝
        std::any wrapped(event);
        for (auto& handler_wrapper : it->second) {
            handler_wrapper(wrapped);
        }
    }

    // ----------------------------------------------------------
    // clear() — 清空所有订阅关系
    // 主要用于测试隔离：每个测试用例可重置总线状态。
    // ----------------------------------------------------------
    void clear() {
        handlers_.clear();
    }

private:
    // 核心数据结构：
    //   键   — std::type_index，唯一标识一种事件类型
    //   值   — 类型擦除的处理器列表，按注册顺序排列
    std::unordered_map<
        std::type_index,
        std::vector<std::function<void(const std::any&)>>
    > handlers_;
};

} // namespace omm::events
