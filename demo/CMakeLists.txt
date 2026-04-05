cmake_minimum_required(VERSION 3.16)
project(neural_bsde_demo CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 主可执行文件：路径生成 + 可选ONNX推理
add_executable(demo_runner
    cpp/main.cpp
    cpp/LiftedHestonSim.cpp
)

target_include_directories(demo_runner PRIVATE cpp)

# 可选：启用ONNX Runtime推理基准测试
option(BUILD_ONNX_DEMO "Enable ONNX Runtime inference benchmark" OFF)

if(BUILD_ONNX_DEMO)
    target_sources(demo_runner PRIVATE cpp/OnnxInference.cpp)

    find_path(ORT_INCLUDE_DIR "onnxruntime_cxx_api.h"
        HINTS
            "${ONNXRUNTIME_ROOT}/include"
            "$ENV{HOME}/onnxruntime/include"
            "/usr/local/include/onnxruntime"
            "/opt/homebrew/include")

    find_library(ORT_LIB onnxruntime
        HINTS
            "${ONNXRUNTIME_ROOT}/lib"
            "$ENV{HOME}/onnxruntime/lib"
            "/usr/local/lib"
            "/opt/homebrew/lib")

    if(ORT_INCLUDE_DIR AND ORT_LIB)
        target_include_directories(demo_runner PRIVATE ${ORT_INCLUDE_DIR})
        target_link_libraries(demo_runner PRIVATE ${ORT_LIB})
        target_compile_definitions(demo_runner PRIVATE BUILD_ONNX_DEMO)
        message(STATUS "[ONNX Runtime] 找到: ${ORT_LIB}")
    else()
        message(WARNING "[ONNX Runtime] 未找到 — ONNX推理已禁用。请设置 -DONNXRUNTIME_ROOT=<路径>")
        set(BUILD_ONNX_DEMO OFF)
    endif()
endif()

# ── Alpha Pipeline Demo (buyer-side) ─────────────────────────
# Collects all src/core/*.cpp (domain, analytics, infrastructure)
# plus DeltaHedger from the seller module.
file(GLOB_RECURSE ALPHA_CORE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../src/core/**/*.cpp")

add_executable(alpha_runner
    cpp/alpha_main.cpp
    ${ALPHA_CORE_SRCS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/modules/seller/DeltaHedger.cpp
)

target_include_directories(alpha_runner PRIVATE
    cpp                                               # demo-local headers
    ${CMAKE_CURRENT_SOURCE_DIR}/../src                # resolves core/... and modules/...
)

# artifacts目录输出路径（运行时相对路径提示）
message(STATUS "构建完成后请在 demo/ 目录下运行: ./build/demo_runner")
message(STATUS "Alpha pipeline: ./build/alpha_runner")
