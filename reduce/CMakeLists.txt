# 设置最低 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 项目名称和语言
project(MyCudaProject LANGUAGES CXX CUDA)

# 指定 C++ 编译标准
set(CMAKE_CXX_STANDARD 20)
# 为 .cu 文件中的主机代码设置 C++ 标准 (这是解决你问题的关键)
set(CMAKE_CUDA_STANDARD 20)

# 设置 CUDA 编译器的标准
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

# 设置编译输出路径（可选）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)

# 添加可执行文件目标，并指定源文件
add_executable(sum sum.cu)

# 添加父目录作为头文件路径
target_include_directories(sum PRIVATE ${CMAKE_SOURCE_DIR}/..)

# 设置 CUDA 编译选项
target_compile_options(sum PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>  # 根据 GPU 架构选择
)
