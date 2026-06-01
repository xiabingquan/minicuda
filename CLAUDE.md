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

## 文档撰写规则

- 区分"当前讨论内容"和"应该记录到文档的内容"。文档是独立、自洽、可公开发行的版本，读者接触到的只有最终结果，不应包含当前会话的讨论过程。例如：方案从 a 改为 b 后，文档只写 b 的独立内容；不写"为什么是 b 而不是 a"或"原本是 a, 后来改成 b"这类对比，否则会给只看到 b 的读者造成困扰。
- 不要滥用粗体。仅在确实需要视觉强调（如关键警告）时使用，避免一段文字里出现多处粗体导致重点失焦。
- 如果用户没有要求，不要写"实际扩展"、"容易踩的坑"、"扩展思考"这类冗余、信息密度低、和当前内容无关的知识点。

## 通用学习流程

1. 用户针对项目计划进行提问，提问完毕后撰写笔记。
2. 用户写 .cu 代码，agent 辅助补全（如提供 main 函数，用 libtorch 对比结果做精度测试）。添加 main 函数时同步修改 run.sh 中的 SRC 指向该文件。
3. 用户用 nvcc 单独编译该 kernel（run.sh），确保单测通过。
4. Agent 辅助完成集成：删除main函数、添加 pybind 绑定、.pyi 类型文件、Python 单测等。
5. 用户跑 Python 单测，确保测试通过。
6. 补全笔记。

## 交互规则

- 讨论过程中，除非用户显式要求，否则永远不要直接输出代码答案。

## Excalidraw 规范

- 默认文本使用 Excalifont（fontFamily: 8）。
- 代码/等宽文本使用 Monospace（fontFamily: 5）。
