#pragma once
// ============================================================
// NpyWriter.hpp — 极简 .npy 文件写入器
//
// 仅支持 float32 和 float64 的 C 连续数组。
// 格式参考: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
// ============================================================
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <sstream>

namespace demo {

inline void save_npy_float32(
    const std::string& path,
    const float* data,
    const std::vector<int>& shape)
{
    // 构建 numpy header 字符串
    // 格式: {'descr': '<f4', 'fortran_order': False, 'shape': (d0, d1, ...), }
    std::ostringstream hdr;
    hdr << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (int i = 0; i < (int)shape.size(); ++i) {
        hdr << shape[i];
        if (i + 1 < (int)shape.size()) hdr << ", ";
    }
    // numpy要求单元素tuple有尾随逗号
    if (shape.size() == 1) hdr << ",";
    hdr << "), }";

    // header必须以'\n'结尾，总长度对齐到64字节
    std::string header_str = hdr.str();
    // 魔数(6) + version(2) + header_len(2) = 10 bytes prefix
    int prefix_len = 10;
    int hdr_len = static_cast<int>(header_str.size()) + 1; // +1 for '\n'
    int padded  = ((prefix_len + hdr_len + 63) / 64) * 64;
    int padding = padded - prefix_len - hdr_len;
    header_str.append(padding, ' ');
    header_str += '\n';

    uint16_t header_data_len = static_cast<uint16_t>(header_str.size());

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("[NpyWriter] 无法写入: " + path);

    // 魔数 + 版本 1.0
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    f.write(magic, 8);

    // header长度（小端16位）
    f.write(reinterpret_cast<const char*>(&header_data_len), 2);

    // header内容
    f.write(header_str.data(), header_str.size());

    // 数据（float32，小端）
    long long n = 1;
    for (int d : shape) n *= d;
    f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
}

// 便利重载：接受 std::vector<float>
inline void save_npy_float32(
    const std::string& path,
    const std::vector<float>& data,
    const std::vector<int>& shape)
{
    save_npy_float32(path, data.data(), shape);
}

} // namespace demo
