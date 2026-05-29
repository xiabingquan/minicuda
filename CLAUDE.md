# CUDA Learning

## 项目目标

通过由易到难实现各种 CUDA kernel，系统学习 CUDA 编程，最终具备阅读和理解 flash-attn、CUTLASS、cuBLAS、DeepEP、DeepGEMM 等大型 CUDA 代码库的能力。

## 学习者背景

- Python 熟练，算法功底扎实。
- 了解训练 infra 知识（数据并行、张量并行、流水线并行等）。
- 学过 C++ 但实践经验少，需要在项目中同步提升 C++ 工程能力。
- CUDA 初学者，从零开始。

## 学习形式

以实践为主：写 kernel -> 精度测试（对齐 PyTorch 参考实现）-> nsys/ncu profile -> 迭代优化。逐步覆盖高阶特性（shared memory、warp primitives、Tensor Core、TMA、CuTe 等）。

## 开发环境

- 本地 Mac (VSCode) 编辑代码，远程 GPU 服务器编译和执行。
- 使用 CMake 构建，支持 PyTorch C++ extension 集成。
- 精度测试用 Python (PyTorch) 作为 reference。
- 性能分析使用 nsys (timeline) 和 ncu (kernel-level)。
- C++/CUDA 代码通过 pybind11 或 torch.utils.cpp_extension 导出到 Python。

## 项目结构

- plans/：学习计划和路线图。
- notes/course-0x.md：每个课程的学习笔记。
- 每个 kernel 放在独立目录下，包含 .cu（kernel 实现）、CMakeLists.txt、test.py（精度 & 性能测试）。

## 代码规范

- Kernel 文件使用 Google-style 注释，关键步骤标注意图。
- Commit message 简洁说明实现了什么、用了什么优化技巧。
- 课程笔记不需要"Course XX 笔记"这类顶层标题，直接从具体内容的一级标题开始。
- 课程笔记中一级标题表示一个独立的知识板块，必须带数字编号（如"# 1. 如何将 CUDA 仓库导出 Python 包并调用"）。
