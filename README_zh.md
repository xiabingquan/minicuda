# minicuda

[English](README.md)

一个实验性的 CUDA 学习项目，尝试探索 LLM 时代的学习范式。不同于传统的教材或视频课程，这个仓库尝试**全程在大模型的辅助下**学习 CUDA 编程 —— 从制定学习计划、讨论概念，到记录学习笔记，全程由 LLM 参与协作。

## 学习路线

![CUDA 学习路线图](assets/roadmap.excalidraw.svg)

课程体系共 10 个课程，从基础环境搭建逐步推进到阅读和复现工业级 CUDA 代码库。

| 课程 | 主题 | 核心内容 |
|------|------|----------|
| 1 | 环境搭建 & Hello CUDA | CMake, nvcc, pybind11, PyTorch C++ extension |
| 2 | 内存层级 & 矩阵运算 | shared memory, 合并访存, bank conflict, 分块 GEMM |
| 3 | Profile & 性能分析 | nsys, ncu, roofline model, CUDA stream |
| 4 | Reduce & ML Kernel | warp 原语, softmax, RMSNorm, LayerNorm, 激活函数 |
| 5 | 进阶 Kernel & 融合优化 | online 算法, prefix sum, TopK, RoPE, kernel fusion |
| 6 | Tensor Core & 高性能 GEMM | WMMA, double buffering, CuTe, CUTLASS |
| 7 | Hopper & 异步执行 | TMA, pipeline, warpgroup, thread block cluster |
| 8 | Flash Attention | online softmax + tiling, 前向/反向, causal masking |
| 9 | DeepEP | MoE expert parallelism, all-to-all 通信 |
| 10 | DeepGEMM | FP8 GEMM, JIT 编译, persistent kernel |

完整学习计划（含每个课程的学习目标、知识点、实践项目和验收标准）见 [plans/overview.md](plans/overview.md)。

## 仓库结构

```
csrc/             CUDA kernel 实现 (.cu)
binding/          pybind11 / PyTorch extension 绑定 (.cpp)
tests/            Python 精度 & 性能测试
include/          共享 C++ 头文件
3rdparty/         第三方依赖 (如 CUTLASS)
plans/            学习计划与课程大纲
  overview.md     课程总览
  course-XX.md    各课程详细计划
notes/            按课程组织的学习笔记
  course-XX/      笔记、总结与踩坑记录
assets/           静态资源 (图表、图片)
```

## 学习方式

每个 kernel 遵循统一的工作流：

1. **实现** — 在 `csrc/` 中编写 CUDA kernel
2. **验证** — 用 `torch.allclose` 对齐 PyTorch 参考实现
3. **分析** — 用 `nsys` (时间线) 和 `ncu` (kernel 级指标) 进行性能分析
4. **优化** — 根据 profiling 结果迭代优化，保留多版本对比

## 许可证

[MIT](LICENSE)
