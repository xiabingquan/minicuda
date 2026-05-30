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
