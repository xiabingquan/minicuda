# 课程 1: 环境搭建 & Hello CUDA

## 目标

跑通从编写、编译、运行到 Python 调用的完整流程。理解 CUDA 编程模型和 GPU 内存管理，能独立写 CMakeLists.txt 编译 CUDA 项目，能通过 pybind11 / torch.utils.cpp_extension 将 kernel 导出为 Python 模块。

## 学习步骤

### Step 1: 理解 CUDA 项目的构建与导出流程

在写任何 kernel 之前，先搞清楚整条工具链：.cu 文件如何被 nvcc 编译、如何链接成 .so、如何通过 pybind11 导出 Python 接口、如何通过 setup.py 打包成 pip 可安装的包。

具体内容：
- 阅读并理解项目根目录下 CMakeLists.txt 和 setup.py 的每一行配置。
- 理解 nvcc 编译 .cu、g++ 编译 .cpp、链接生成 .so、Python dlopen 加载 .so 的完整链路。
- 理解 `pip install -e .` 开发模式和 `pip install .` 正式安装的区别。
- 理解 minicuda/_C 与 minicuda/__init__.py 的关系。
- 理解类型标注机制：为什么需要 .pyi stub 文件、文件命名约束（必须与模块名一致）、为什么用显式 import 而非 `import *`（类型检查器对 .so 不可读，无法展开 `*`）、大型 stub 如何拆分到 stub-only 子目录。

产出：将构建流程整理为笔记，写入 notes/course-01.md。

### Step 2: 编写第一个 CUDA kernel — vector_add

目标是用最简单的 kernel 跑通"写代码 -> 编译 -> 运行 -> 验证"的完整流程。

具体内容：
- 学习 CUDA 编程模型：grid、block、thread 三级层级结构。
- 学习 `__global__` 函数的声明和 `<<<grid, block>>>` 的调用语法。
- 实现 vector_add kernel：每个线程计算一个元素 `c[i] = a[i] + b[i]`。
- 通过 torch::Tensor 管理 GPU 内存（分配、data_ptr 取指针、返回结果）。
- 写一个带 main() 的独立可执行文件，用 CMake 编译运行，打印结果验证正确性。

产出：csrc/vector_add.cu + CMakeLists.txt，能编译运行。

### Step 3: 编写 saxpy kernel

在 vector_add 的基础上增加标量参数，学习 grid-stride loop 模式。

学习目的：vector_add 中每个 thread 只处理一个元素，grid 大小必须 >= 元素数量。grid-stride loop 让固定数量的 thread 循环处理多个元素，将问题规模与 grid 大小解耦，是生产代码的标准写法。

具体内容：
- 实现 y = a * x + y，kernel 接收标量参数 a 和指针参数 x, y。
- 学习 grid-stride loop：`for (int i = idx; i < n; i += stride)`，使一个 kernel 能处理任意长度输入，不受 grid 大小限制。
- 对比有无 grid-stride loop 的区别：不用 loop 时，thread 数必须 >= 元素数。

产出：csrc/saxpy.cu，能处理任意长度输入。

### Step 4: 编写 matrix_add kernel

从一维索引扩展到二维索引。

具体内容：
- 学习 2D grid 和 2D block 的配置：`dim3 grid(gx, gy)`, `dim3 block(bx, by)`。
- 学习二维线程索引的计算：`row = blockIdx.y * blockDim.y + threadIdx.y`。
- 实现二维矩阵逐元素相加：C[i][j] = A[i][j] + B[i][j]。
- 理解行优先存储下二维索引到一维地址的映射：`offset = row * width + col`。
- 处理矩阵尺寸不是 block 大小整数倍的边界情况。

产出：csrc/matrix_add.cu，支持任意大小矩阵。

### Step 5: 编写 rgb_to_grayscale kernel

一个更贴近实际应用的 elementwise kernel，涉及多通道数据处理。

具体内容：
- 理解图像在 GPU 内存中的存储方式：HWC（高度-宽度-通道，OpenCV 默认）vs CHW（通道-高度-宽度，PyTorch 默认）。
- 实现灰度化公式：gray = 0.299 * R + 0.587 * G + 0.114 * B。
- 每个线程处理一个像素，从 3 个通道读取 RGB 值，写出 1 个灰度值。
- （可选）了解 pitch memory：cudaMallocPitch 分配对齐内存以提高访存效率。

产出：csrc/rgb_to_grayscale.cu。

### Step 6: 将 kernel 导出为 Python/PyTorch 模块

将前面写好的 kernel（以 vector_add 为例）封装为 Python 可调用的接口。

具体内容：
- 编写 binding/bind.cpp：用 pybind11 将 C++ wrapper 函数注册到 Python。
- C++ wrapper 函数负责：接收 torch::Tensor 参数 -> 提取 data_ptr -> 启动 CUDA kernel -> 返回结果 Tensor。
- 用 `pip install -e .` 编译安装。
- 编写 tests/test_vector_add.py：生成随机 Tensor，调用 minicuda.vector_add，与 PyTorch 的 `torch.add` 对比，用 `torch.allclose` 验证精度。

产出：binding/bind.cpp + tests/test_vector_add.py，`from minicuda import vector_add` 可用且精度对齐。

## 验收标准

- [x] 能独立编写 CMakeLists.txt 并用 cmake + make 编译出 CUDA 可执行文件。
- [x] 能解释 grid/block/thread 三级层级的含义和索引计算方式。
- [x] 能解释 grid-stride loop 的作用及为什么需要它。
- [x] 能用 `pip install -e .` 编译项目，在 Python 中 `from minicuda import xxx` 调用。
- [x] 所有 kernel（vector_add / saxpy / matrix_add）的 Python 单测通过，与 PyTorch 参考实现 allclose（fp32, atol=1e-6）。
- [x] 理解 setup.py (CUDAExtension) 与 CMake 两种构建方式的区别和各自适用场景。
