#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "../events/EventBus.hpp"
#include "../events/Events.hpp"

// ============================================================
// 文件：ParameterStore.hpp
// 职责：参数仓库 — 订阅 ParamUpdateEvent，持久化模型参数，
//       并提供带版本查询接口（按模型 ID 获取最新参数）。
//
// 对应 Risk_Calibration.md §2.7（Parameter Feedback）
//
// 这是"校准 → 实时定价"反馈闭环中的存储节点：
//   BacktestCalibrationApp → ParamUpdateEvent → ParameterStore
//                                                     ↓
//                                   实时引擎查询 get_params(model_id)
//
// 接口说明：
//   subscribe_handlers()           — 向 EventBus 注册 ParamUpdateEvent 处理器
//   get_params(model_id) → map     — 获取指定模型的最新参数
//   print_all()                    — 打印所有已存储的参数（供 main 日志输出）
//
// 对应 Risk_Calibration.md 中的 IModelParamSource 接口：
//   virtual ParamMap getParams(ModelId id, Timestamp asOf) = 0;
// ============================================================

namespace omm::infrastructure {

// VersionedParams — 带时间戳的参数版本（支持历史回溯）
struct VersionedParams {
    std::unordered_map<std::string, double> params;     // 参数键值对
    events::Timestamp                       updated_at; // 更新时间戳
};

class ParameterStore {
public:
    explicit ParameterStore(std::shared_ptr<events::EventBus> bus);

    // subscribe_handlers() — 注册 ParamUpdateEvent 订阅器
    void subscribe_handlers();

    // get_params() — 获取指定模型的最新参数
    //   若模型未注册，返回空 map
    std::unordered_map<std::string, double> get_params(
        const std::string& model_id
    ) const;

    // print_all() — 打印所有模型的最新参数（供仿真结束时汇报使用）
    void print_all() const;

private:
    // on_param_update() — ParamUpdateEvent 处理器
    //   追加新版本到历史记录，保留所有历史版本
    void on_param_update(const events::ParamUpdateEvent& event);

    std::shared_ptr<events::EventBus> bus_;

    // 参数历史：model_id → 版本列表（按时间顺序追加）
    std::unordered_map<std::string, std::vector<VersionedParams>> history_;
};

} // namespace omm::infrastructure
