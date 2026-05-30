# 课程 2: 内存层级 & 矩阵运算

## 目标

理解 GPU 内存层级（global → shared → register），掌握 CUDA 原生内存管理 API，理解 memory coalescing 和 bank conflict，能编写使用 shared memory 的 kernel。通过矩阵转置和矩阵乘法两个经典问题，体会内存访问模式对性能的决定性影响。

## 学习步骤

### Step 1: CUDA 基础内存管理

课程 1 通过 PyTorch tensor 管理 GPU 内存，屏蔽了底层细节。本节学习 CUDA 原生 API，理解显存分配、Host/Device 数据搬运的完整流程。

具体内容：
- `cudaMalloc(&ptr, bytes)`: 在 GPU 上分配显存，返回 device 指针。
- `cudaMemcpy(dst, src, bytes, direction)`: Host ↔ Device 数据拷贝（同步）。
- `cudaFree(ptr)`: 释放 GPU 显存。
- `cudaMemset`: device 显存清零/初始化。
- 错误处理：用 `CUDA_CHECK` 宏包裹每个 cuda* 调用，避免静默失败。
- 练习：用纯 CUDA API（不依赖 PyTorch）重写 vector_add，手动管理 malloc/memcpy/free 全流程。

产出：csrc/vector_add_raw.cu，不依赖 PyTorch，纯 `nvcc` 编译运行。

### Step 2: 异步执行与 stream

理解同步 API 的瓶颈，学习 stream + async 实现 H2D / kernel / D2H 的流水线重叠。适用场景：输入输出在 CPU、计算在 GPU，数据量较大的批处理任务。

具体内容：
- `cudaMallocHost` / `cudaFreeHost`: 分配 pinned memory（不可换页），是 async 拷贝生效的前提。
- `cudaMemcpyAsync`: 异步拷贝，立即返回不阻塞 CPU。
- `cudaStream_t` + `cudaStreamCreate` / `cudaStreamDestroy`: stream 是 GPU 上的操作队列，同 stream 内串行、跨 stream 可并行。
- `cudaStreamSynchronize`: 等待某个 stream 上所有操作完成。
- kernel launch 的第 4 个参数指定 stream：`kernel<<<grid, block, smem, stream>>>(...)`。
- 三阶段流水线模式：将大数组分成 N 个 chunk，每个 chunk 在自己的 stream 上完成 H2D → kernel → D2H，相邻 chunk 重叠不同阶段。
- 用 nsys timeline 直观观察 stream 的并行执行。

产出：csrc/async_vector_op.cu — 输入输出为 CPU 数组，GPU 上做复合运算（如 `sin(x)*exp(x)+sqrt(|x|)`），分块 + 多 stream 实现流水线，对比同步版本的性能差异。

### Step 3: GPU 内存层级

具体内容：
- Global memory: 大容量（数十 GB）、高延迟（~400 cycles），所有 thread 可见。
- Shared memory: 小容量（每 SM ~48-164 KB）、低延迟（~5 cycles），同 block 内 thread 共享。
- Register: 最快、每 thread 私有，数量有限。
- L1/L2 cache: 硬件自动管理，了解但不直接控制。
- Memory coalescing: 同一 warp 内 32 个 thread 访问连续地址时合并为一次事务。
- `__syncthreads()`: block 内同步屏障的语义和使用时机。

产出：笔记总结各层级的容量、延迟、可见性对比。

### Step 4: 矩阵转置 — 体会访存模式的影响

具体内容：
- Naive 实现：读连续（coalesced）、写跨步（uncoalesced），或反过来。
- Shared memory 版本：先将 tile 加载到 shared memory，再从 shared memory 写出（读写都 coalesced）。
- Bank conflict：shared memory 按 32 bank 分布，同 warp 内访问同一 bank 会串行化。通过 padding 避免。
- 用 ncu 对比 naive vs shared memory 版本的 memory throughput。

产出：csrc/transpose_naive.cu + csrc/transpose_shared.cu。

### Step 5: dot_product — shared memory reduce

具体内容：
- 每个 block 用 shared memory 做 block 内 reduction（归约求和）。
- `__syncthreads()` 保证每轮归约前数据写入完成。
- 最终用 atomicAdd 将各 block 的部分和累加到 global 结果。

产出：csrc/dot_product.cu。

### Step 6: GEMV — 矩阵向量乘

具体内容：
- 实现 y = A * x，每个 thread/warp 负责一行的点积。
- 对比按行访问 vs 按列访问 A 的性能差异（coalescing 影响）。
- 学习 warp 内协作：一个 warp 处理一行，用 warp shuffle 做归约。

产出：csrc/gemv.cu。

### Step 7: Naive GEMM

具体内容：
- 最朴素的 C = A × B 实现：每个 thread 计算 C 的一个元素，循环 K 次从 global memory 读 A 和 B。
- 计算 GFLOPS：`2 * M * N * K / time_seconds / 1e9`。
- 理解为什么 naive 版本 GFLOPS 很低（大量重复全局内存访问）。

产出：csrc/sgemm_naive.cu。

### Step 8: Tiled GEMM — shared memory 实战

具体内容：
- 将 A、B 分块（tile）加载到 shared memory，block 内 thread 协作加载一个 tile，复用数据。
- `__syncthreads()` 保证 tile 加载完成后再计算。
- 分析 tile size 对性能的影响（occupancy vs 数据复用 tradeoff）。
- 对比 naive vs tiled 版本的 GFLOPS 提升。

产出：csrc/sgemm_tiled.cu。

### Step 9: Vectorized GEMM — 向量化访存

具体内容：
- 学习 `float4` 向量化加载：一次读 128 bits（4 个 float），提高带宽利用率。
- 寄存器分块（register tiling）：每个 thread 计算多个输出元素，减少 shared memory 访问次数。
- 对比 tiled vs vectorized 版本的性能。

产出：csrc/sgemm_vectorized.cu。

## 验收标准

- [ ] 能用 cudaMalloc/cudaMemcpy/cudaFree 手动管理显存，完成 H2D → kernel → D2H 流程。
- [ ] 能解释 stream 的作用，能用 cudaMemcpyAsync + 多 stream 实现 H2D / kernel / D2H 流水线重叠。
- [ ] 能说出 global / shared / register 三级内存的容量、延迟和可见性区别。
- [ ] 能解释 memory coalescing 的含义及违反时的性能影响。
- [ ] 能解释 bank conflict 的成因和解决方法（padding）。
- [ ] matrix_transpose_shared 相比 naive 有明显加速，能解释原因。
- [ ] tiled GEMM 精度对齐 PyTorch，性能优于 naive 版本。
- [ ] 能计算 GEMM 的理论 GFLOPS 并与实测对比。
