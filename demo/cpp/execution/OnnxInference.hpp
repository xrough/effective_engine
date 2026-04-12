#pragma once
#ifdef BUILD_ONNX_DEMO

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace demo {

// ============================================================
// ONNX推理结果
// ============================================================
struct HedgeSignal {
    float  Y;           // 诊断用连续值（仅用于监控）
    float  Z_spot;      // 现货对冲控制 Z[0]（生产关键输出）
    double latency_us;  // 单次推理延迟（微秒）
};

// ============================================================
// OnnxInference — 封装ONNX Runtime单步MLP推理
//
// 使用方式：
//   OnnxInference infer(onnx_path, norm_stats_path);
//   auto sig = infer.run(normalized_state_vec);  // 已归一化的状态
//
// 注意：
//   - Ort::Session在构造函数中初始化一次，后续复用（开销仅在首次）
//   - 输入状态必须已经过归一化（由调用方负责）
//   - BSDE时间循环在C++主程序中执行，此处仅为单步前向传播
// ============================================================
class OnnxInference {
public:
    OnnxInference(const std::string& onnx_model_path,
                  const std::string& norm_stats_path);

    // 运行单步推理
    // state_vec: 已归一化的状态向量，大小 = state_dim
    HedgeSignal run(const std::vector<float>& state_vec) const;

    // 延迟统计（基于历史推理记录）
    struct LatencyStats { double p50_us, p99_us; };
    LatencyStats get_latency_stats() const;

    int state_dim() const { return state_dim_; }

    // 归一化参数（供外部使用）
    const std::vector<float>& norm_mean() const { return norm_mean_; }
    const std::vector<float>& norm_std()  const { return norm_std_;  }

    // Training nominal strike (K_train from normalization.json).
    // Models trained with K=100 (synthetic LRH) output Z ≈ N(d1)·σ·K_train.
    // Use this to recover delta: delta = Z_spot / (σ · K_train), not Z_spot / (σ · spot).
    float k_train() const { return k_train_; }

private:
    void load_norm_stats(const std::string& path);

    Ort::Env                       env_;
    mutable Ort::Session           session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<float> norm_mean_;
    std::vector<float> norm_std_;
    float              k_train_   = 100.0f;  // default = 100 (synthetic LRH training scale)
    int                state_dim_ = 0;

    // 输入/输出名称（ONNX图定义的）
    static constexpr const char* INPUT_NAME  = "state";
    static constexpr const char* OUTPUT_Y    = "Y";
    static constexpr const char* OUTPUT_Z    = "Z";

    // 延迟采样（微秒）
    mutable std::vector<double> latency_us_;
};

} // namespace demo

#endif // BUILD_ONNX_DEMO
