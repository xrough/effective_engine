cmake_minimum_required(VERSION 3.16)
project(OptionsMarketMaker CXX)

# 使用 C++17 标准，启用严格模式
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 开启警告：帮助捕获潜在问题
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# 自动收集 src/ 下所有 .cpp 文件
# ModelServiceClient.cpp requires gRPC headers — excluded from default build;
# included only when BUILD_GRPC_CLIENT=ON (see below).
file(GLOB_RECURSE SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/core/infrastructure/ModelServiceClient.cpp")

add_executable(market_maker ${SOURCES})

# 将 src/ 加入包含路径，使所有头文件可通过相对路径引用
target_include_directories(market_maker PRIVATE src)

# ════════════════════════════════════════════════════════════
# Python 绑定模块（omm_core）— 通过 pybind11 构建
#
# 用途：将 C++ 核心层（events / domain / infrastructure）
#       暴露为 Python 扩展模块，供 python/hybrid_application/ 使用。
#
# 构建方式：
#   cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
#   cmake --build build --target omm_core
# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# gRPC Client Library (ModelServiceClient) — optional
#
# Requires: brew install grpc protobuf
# Requires: make proto-cpp  (generates generated/cpp/ stubs)
#
# Build:
#   cmake -S . -B build -DBUILD_GRPC_CLIENT=ON
#   cmake --build build --target model_service_client
# ════════════════════════════════════════════════════════════
option(BUILD_GRPC_CLIENT "Build C++ gRPC client for RoughPricingService" OFF)

if(BUILD_GRPC_CLIENT)
    find_package(Protobuf REQUIRED)
    find_package(gRPC REQUIRED)

    # Generated protobuf/gRPC stubs — run `make proto-cpp` to produce these
    set(GRPC_GEN_SOURCES
        generated/cpp/rough_pricing.pb.cc
        generated/cpp/rough_pricing.grpc.pb.cc
    )

    add_library(model_service_client STATIC
        src/core/infrastructure/ModelServiceClient.cpp
        ${GRPC_GEN_SOURCES}
    )
    target_include_directories(model_service_client PUBLIC
        src
        generated/cpp
    )
    target_link_libraries(model_service_client PUBLIC
        gRPC::grpc++
        protobuf::libprotobuf
    )

    # Link into market_maker when gRPC client is enabled
    target_link_libraries(market_maker PRIVATE model_service_client)
    # 传递预处理器宏，使 C++ 源文件可以通过 #ifdef 感知 gRPC 是否启用
    target_compile_definitions(market_maker PRIVATE BUILD_GRPC_CLIENT)
endif()

