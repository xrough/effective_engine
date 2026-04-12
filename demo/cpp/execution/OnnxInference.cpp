#ifdef BUILD_ONNX_DEMO

#include "OnnxInference.hpp"
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>

// 最小JSON解析（仅用于读取normalization.json中的mean/std数组）
// 避免引入外部JSON库依赖
namespace {

// Parse a top-level scalar float from JSON: "key": value
// Returns default_val if the key is not present.
float parse_float_scalar(const std::string& json, const std::string& key, float default_val = 0.0f) {
    std::string search = "\"" + key + "\": ";
    auto pos = json.find(search);
    if (pos == std::string::npos) return default_val;
    pos += search.size();
    // skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    try {
        return std::stof(json.substr(pos));
    } catch (...) {
        return default_val;
    }
}

// 从形如 "[1.2, 3.4, 5.6]" 的字符串中提取float数组
std::vector<float> parse_float_array(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\": [";
    auto pos = json.find(search);
    if (pos == std::string::npos)
        throw std::runtime_error("[OnnxInference] normalization.json中找不到key: " + key);

    pos += search.size();
    auto end = json.find(']', pos);
    if (end == std::string::npos)
        throw std::runtime_error("[OnnxInference] normalization.json格式错误");

    std::string arr_str = json.substr(pos, end - pos);
    std::vector<float> result;
    std::istringstream ss(arr_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // 去除空白
        token.erase(0, token.find_first_not_of(" \t\n\r"));
        token.erase(token.find_last_not_of(" \t\n\r") + 1);
        if (!token.empty()) result.push_back(std::stof(token));
    }
    return result;
}

} // anonymous namespace

namespace demo {

// ============================================================
// 构造函数：加载ONNX模型和归一化统计
// ============================================================
OnnxInference::OnnxInference(const std::string& onnx_path,
                             const std::string& norm_stats_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "neural_bsde")
    , session_(nullptr)
{
    // 会话选项（单线程推理，降低小模型开销）
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, onnx_path.c_str(), opts);

    load_norm_stats(norm_stats_path);

    std::cout << "[OnnxInference] 模型已加载: " << onnx_path << "\n"
              << "  state_dim=" << state_dim_ << "\n";
}

// ============================================================
// load_norm_stats: 解析normalization.json中的mean和std
// ============================================================
void OnnxInference::load_norm_stats(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("[OnnxInference] 无法读取: " + path);

    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    norm_mean_ = parse_float_array(json, "mean");
    norm_std_  = parse_float_array(json, "std");

    if (norm_mean_.size() != norm_std_.size())
        throw std::runtime_error("[OnnxInference] mean和std维度不匹配");

    state_dim_ = static_cast<int>(norm_mean_.size());

    // K_train: nominal strike used during BSDE training.
    // Z_spot ≈ N(d1)·σ·K_train, so delta = Z_spot / (σ·K_train).
    // Default 100.0 (synthetic LRH training with S0=K=100).
    k_train_ = parse_float_scalar(json, "K_train", 100.0f);
}

// ============================================================
// run: 单步推理
// state_vec: 已归一化的状态向量（大小必须等于state_dim_）
// ============================================================
HedgeSignal OnnxInference::run(const std::vector<float>& state_vec) const {
    if ((int)state_vec.size() != state_dim_) {
        throw std::invalid_argument(
            "[OnnxInference] 输入大小不匹配: 期望" +
            std::to_string(state_dim_) + "，实际" +
            std::to_string(state_vec.size()));
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // 构建输入张量：shape (1, state_dim)
    std::array<int64_t, 2> input_shape{1, state_dim_};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ONNX Runtime要求非const指针，但不会修改数据
    std::vector<float> state_copy = state_vec;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, state_copy.data(), state_copy.size(),
        input_shape.data(), 2);

    // 运行推理
    const char* input_names[]  = { INPUT_NAME };
    const char* output_names[] = { OUTPUT_Y, OUTPUT_Z };

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 2
    );

    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    latency_us_.push_back(us);

    // 提取输出
    float Y      = outputs[0].GetTensorData<float>()[0];
    float Z_spot = outputs[1].GetTensorData<float>()[0];  // Z[0]：现货对冲分量

    return HedgeSignal{Y, Z_spot, us};
}

// ============================================================
// get_latency_stats: 计算p50和p99延迟
// ============================================================
OnnxInference::LatencyStats OnnxInference::get_latency_stats() const {
    if (latency_us_.empty()) return {0.0, 0.0};

    std::vector<double> sorted = latency_us_;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&](double p) {
        int idx = static_cast<int>(p * sorted.size());
        idx = std::max(0, std::min(idx, (int)sorted.size() - 1));
        return sorted[idx];
    };

    return {percentile(0.50), percentile(0.99)};
}

} // namespace demo

#endif // BUILD_ONNX_DEMO
