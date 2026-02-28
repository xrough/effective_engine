#include "ParameterStore.hpp"
#include <iostream>
#include <iomanip>

// ============================================================
// ParameterStore 实现
//
// 设计说明：
//   - 每次收到 ParamUpdateEvent，在对应模型的版本列表末尾追加新版本
//   - get_params() 始终返回最新版本（列表末尾元素）
//   - 历史版本保留，支持未来扩展（如 get_params(model_id, asOf)）
// ============================================================

namespace omm::infrastructure {

ParameterStore::ParameterStore(std::shared_ptr<events::EventBus> bus)
    : bus_(std::move(bus)) {}

void ParameterStore::subscribe_handlers() {
    // 订阅 ParamUpdateEvent — 由 BacktestCalibrationApp 发布
    bus_->subscribe<events::ParamUpdateEvent>(
        [this](const events::ParamUpdateEvent& evt) {
            this->on_param_update(evt);
        }
    );
}

void ParameterStore::on_param_update(const events::ParamUpdateEvent& event) {
    // 追加新版本到历史记录（保留所有历史版本，支持回溯查询）
    history_[event.model_id].push_back(
        VersionedParams{event.params, event.updated_at}
    );

    std::cout << "[参数仓库] 📥 接收参数更新  模型: " << event.model_id
              << "  版本: v" << history_[event.model_id].size() << "  参数: ";
    for (const auto& [key, val] : event.params) {
        std::cout << key << "=" << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "\n";
}

std::unordered_map<std::string, double> ParameterStore::get_params(
    const std::string& model_id) const {

    auto it = history_.find(model_id);
    if (it == history_.end() || it->second.empty()) {
        return {}; // 模型未注册，返回空参数
    }

    // 返回最新版本（版本列表末尾）
    return it->second.back().params;
}

void ParameterStore::print_all() const {
    std::cout << "\n[参数仓库] ══════ 已存储参数汇总 ══════\n";
    if (history_.empty()) {
        std::cout << "[参数仓库] （无已存储参数）\n";
        return;
    }

    for (const auto& [model_id, versions] : history_) {
        std::cout << "  模型 [" << model_id << "]  共 "
                  << versions.size() << " 个版本\n";

        // 打印最新版本参数
        const auto& latest = versions.back();
        std::cout << "  最新参数:";
        for (const auto& [key, val] : latest.params) {
            std::cout << "  " << key << " = "
                      << std::fixed << std::setprecision(6) << val;
        }
        std::cout << "\n";
    }
    std::cout << "[参数仓库] ════════════════════════════\n";
}

} // namespace omm::infrastructure
