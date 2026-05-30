# 1. CUDA 原生显存管理与 PyTorch Tensor 互转

课程 1 全程用 PyTorch tensor 管理显存（`torch::empty`、`data_ptr`），底层细节被屏蔽。本节学习 CUDA 原生 API 自己管理显存，以及如何在原生指针和 `torch::Tensor` 之间互相转换。

## 核心 API 速查

| 函数 | 作用 |
|---|---|
| `cudaMalloc(&ptr, bytes)` | GPU 上分配显存。注意第 1 个参数是**指针的地址** (`void**`) |
| `cudaFree(ptr)` | 释放 GPU 显存 |
| `cudaMemcpy(dst, src, bytes, dir)` | 同步拷贝, dir 取 `cudaMemcpyHostToDevice` / `DeviceToHost` 等 |
| `cudaMemset(ptr, val, bytes)` | 按字节填充 (val 是 int, 通常用 0 清零) |

`cudaMalloc` 返回的指针只能在 GPU kernel 里解引用，CPU 端直接 `*ptr` 会 segfault。

## 完整流程示例

```cuda
int n = 1024;
float h_a[n] = {...};       // host 数据

float *d_a, *d_b;
cudaMalloc(&d_a, n * sizeof(float));   // 分配
cudaMalloc(&d_b, n * sizeof(float));

cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);  // H2D
my_kernel<<<grid, block>>>(d_a, d_b, n);
cudaMemcpy(h_a, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);  // D2H

cudaFree(d_a);
cudaFree(d_b);
```

## 错误处理: CUDA_CHECK 宏

`<cuda_runtime.h>` 只提供 `cudaError_t` 和 `cudaGetErrorString`，没有 `CUDA_CHECK` 宏 — 这是社区惯例，每个项目自己定义：

```cpp
#define CUDA_CHECK(call) do {                                              \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                cudaGetErrorString(err), __FILE__, __LINE__);              \
        exit(1);                                                           \
    }                                                                      \
} while (0)

CUDA_CHECK(cudaMalloc(&d_a, bytes));   // 分配失败时打印错误并退出
```

## 与 PyTorch Tensor 互转

### Tensor → 原生指针

直接用 `data_ptr<T>()`。这是个**视图**，不转移所有权：

```cpp
torch::Tensor t = torch::randn({n}, opts.device(torch::kCUDA));
float* ptr = t.data_ptr<float>();   // GPU 指针, t 析构时显存才释放
my_kernel<<<...>>>(ptr, ...);
```

### 原生指针 → Tensor (`torch::from_blob`)

把 `cudaMalloc` 出来的指针包装成 tensor：

```cpp
float* d_ptr;
cudaMalloc(&d_ptr, n * sizeof(float));
my_kernel<<<...>>>(d_ptr, ...);

auto t = torch::from_blob(
    d_ptr, {n},
    /*deleter=*/[](void* p) { cudaFree(p); },   // tensor 析构时调用
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
```

# 2. CPU/GPU 异步

## 场景

输入输出都在 CPU 上、GPU 只负责中间计算的操作。
> 这类场景在实际场景中并不常见，此处主主要是讲解基本思路，起学习示范作用。

本节选取的例子是矩阵加法。
> 实测发现async版本比sync版本更慢，不过起教学作用还是可以的。

## 流水线设计

每个 chunk 的完整路径包含五步：

```
regular CPU a/b → pinned CPU buffer → GPU buffer → kernel → GPU output → CPU output
       │                │                  │           │          │           │
       └─ memcpy ───────┘  cudaMemcpyAsync ┘  vector_add_kernel  cudaMemcpy   memcpy
```

其中：

- 普通 `malloc` 出来的 CPU 内存 → pinned buffer：必须用 host 端 `memcpy`，CPU 同步执行。
- pinned buffer → device buffer：`cudaMemcpyAsync` 异步发起，立即返回。
- kernel：`<<<>>>` 异步发起，立即返回。
- device buffer → CPU 输出：完整流水线最后一次性 D2H 即可。

CPU 主线程顺序执行，但提交完 `cudaMemcpyAsync` 和 kernel 后立刻返回，于是 CPU 接下来做的事（写下一个 chunk 的 pinned buffer）能与 GPU 后台正在跑的 H2D / kernel 同时进行 — 这就是要利用的异步窗口。

## 哪些地方需要同步

`cudaMemcpyAsync` 要求源/目标内存地址在 DMA 期间不被改写。所以唯一需要保护的是 host 端 pinned buffer：当前 chunk 写入这块 buffer 时，前一次以这块 buffer 为源的 H2D 必须已完成。

实现上每轮 GPU 提交前调一次 `cudaStreamSynchronize` 即可，单 stream 内其它顺序由 stream 自身保证，不需要额外同步。

## Ping-pong Buffer

如果只有一块 pinned buffer，本轮 CPU 想写 buffer 之前必须等当前 H2D 跑完，CPU 全程被卡住。

分配两块 pinned buffer 轮流使用，CPU 写 next、GPU 读 cur，每轮结束 `std::swap` 切换索引。input 和 output 各需要两块。

## 用 nsys 验证 overlap

代码中用 NVTX 标记标注 CPU 端关键步骤：

```cpp
#include <nvtx3/nvToolsExt.h>

nvtxRangePushA("cpu_memcpy");
memcpy(...);
nvtxRangePop();
```

NVTX 3 是 header-only，只 include 头文件即可，不需要链接 `.so`。

```bash
nsys profile -t nvtx,cuda,osrt --stats=true -o /tmp/profile ./prog
```

GUI 里展开主线程 → NVTX 行，可以看到 cpu_memcpy 与 CUDA HW 行上的 H2D / kernel 在同一时间段并行：

![CPU/GPU overlap timeline](../assets/course-02/cpu-gpu-overlap.png)
