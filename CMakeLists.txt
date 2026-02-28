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
file(GLOB_RECURSE SOURCES "src/*.cpp")

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
option(BUILD_PYTHON_BINDINGS "构建 Python pybind11 扩展模块 omm_core" OFF)

if(BUILD_PYTHON_BINDINGS)
    # 明确指定使用 Python 3.13（pybind11 所在的解释器），避免多版本冲突
    set(Python3_EXECUTABLE "/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/bin/python3.13"
        CACHE FILEPATH "Python3 interpreter for pybind11 bindings")

    # 从 Python 3.13 包获取 pybind11 CMake 配置路径
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(PYBIND11_FINDPYTHON ON)   # 使用现代 FindPython3 机制
    find_package(pybind11 CONFIG REQUIRED)

    # 收集除 main.cpp 之外的所有源文件（.so 中不能有 main()）
    file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
    list(REMOVE_ITEM LIB_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")

    # 构建 Python 扩展模块
    pybind11_add_module(omm_core
        src/bindings/omm_bindings.cpp
        ${LIB_SOURCES}
    )
    target_include_directories(omm_core PRIVATE src)
endif()
