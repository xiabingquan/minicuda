# 课程 2: 内存层级 & 矩阵运算

## 目标

理解 GPU 内存层级（global → shared → register），掌握 CUDA 原生内存管理 API，理解 memory coalescing 和 bank conflict，能编写使用 shared memory 的 tiled GEMM kernel。

## 学习步骤

### Step 1: CUDA 原生内存管理

课程 1 通过 PyTorch tensor 管理 GPU 内存，屏蔽了底层细节。本节学习 CUDA 原生 API，理解显存分配、Host/Device 数据搬运的完整流程。

具体内容：
- `cudaMalloc(&ptr, bytes)`: 在 GPU 上分配显存，返回 device 指针。
- `cudaMemcpy(dst, src, bytes, direction)`: Host ↔ Device 数据拷贝（同步）。
- `cudaFree(ptr)`: 释放 GPU 显存。
- `cudaMemcpyAsync` + `cudaStream_t`: 异步拷贝，与 kernel 执行重叠。
- `cudaMallocPitch`: 对齐分配，了解 pitch memory 的用途和代价。
- 练习：用纯 CUDA API（不依赖 PyTorch）重写 vector_add，手动管理 malloc/memcpy/free 全流程。

产出：csrc/vector_add_raw.cu，不依赖 PyTorch，纯 `nvcc` 编译运行。

### Step 2: GPU 内存层级

具体内容：
- Global memory: 大容量（数十 GB）、高延迟（~400 cycles），所有 thread 可见。
- Shared memory: 小容量（每 SM ~48-164 KB）、低延迟（~5 cycles），同 block 内 thread 共享。
- Register: 最快、每 thread 私有，数量有限。
- L1/L2 cache: 硬件自动管理，了解但不直接控制。
- Memory coalescing: 同一 warp 内 32 个 thread 访问连续地址时合并为一次事务。

产出：笔记总结各层级的容量、延迟、可见性对比。

### Step 3: 矩阵转置 — 体会访存模式的影响

具体内容：
- Naive 实现：读连续（coalesced）、写跨步（uncoalesced），或反过来。
- Shared memory 版本：先将 tile 加载到 shared memory，再从 shared memory 写出（读写都 coalesced）。
- Bank conflict：shared memory 按 32 bank 分布，同 warp 内访问同一 bank 会串行化。通过 padding 避免。
- 用 ncu 对比 naive vs shared memory 版本的 memory throughput。

产出：csrc/transpose_naive.cu + csrc/transpose_shared.cu。

### Step 4: Tiled GEMM — shared memory 实战

具体内容：
- Naive GEMM：每个 thread 计算 C 的一个元素，重复从 global memory 读 A 和 B。
- Tiled GEMM：将 A、B 分块加载到 shared memory，block 内 thread 协作加载一个 tile，复用数据。
- `__syncthreads()`: block 内同步屏障，保证 shared memory 写入完成后再读取。
- 分析 tile size 对性能的影响（occupancy vs 数据复用）。

产出：csrc/sgemm_naive.cu + csrc/sgemm_tiled.cu。

## 验收标准

- [ ] 能用 cudaMalloc/cudaMemcpy/cudaFree 手动管理显存，完成完整的 H2D → kernel → D2H 流程。
- [ ] 能说出 global / shared / register 三级内存的容量、延迟和可见性区别。
- [ ] 能解释 memory coalescing 的含义及违反时的性能影响。
- [ ] 能解释 bank conflict 的成因和解决方法。
- [ ] tiled GEMM 精度对齐 PyTorch，shared memory 版本性能优于 naive 版本。
