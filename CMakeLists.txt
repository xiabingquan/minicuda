# 要求 CMake 最低版本 3.18, 这是 CUDA 作为一等语言 (first-class language) 被支持的最低版本
cmake_minimum_required(VERSION 3.18)

# 声明项目名称, 并启用 C++ 和 CUDA 两种语言的编译支持
# 设置 LANGUAGES 后, CMake 会自动寻找 g++/clang++ 和 nvcc
project(cuda_learning LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 的语言标准为 C++17
# PyTorch 和 CUTLASS 等库要求至少 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找系统中安装的 CUDA Toolkit (头文件、库文件、nvcc 等)
# REQUIRED 表示找不到就直接报错终止, 不会静默跳过
find_package(CUDAToolkit REQUIRED)

# 将项目根目录下的 include/ 添加到全局头文件搜索路径
# 这样所有子目录中的 .cu/.cpp 文件都可以直接 #include 项目公共头文件
include_directories(${CMAKE_SOURCE_DIR}/include)

# 按需取消注释, 添加各 kernel 的子目录
# 每个子目录应有自己的 CMakeLists.txt, 定义 add_executable 或 add_library
# add_subdirectory(csrc/vector_add)
