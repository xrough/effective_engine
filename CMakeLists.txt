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
