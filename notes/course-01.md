# 1. 如何将 CUDA 仓库导出 Python 包并调用

## 项目结构

将 CUDA 代码导出为 Python 包，需要以下几个组成部分：

```
csrc/              CUDA kernel 实现 (.cu 文件)
binding/           pybind11 绑定代码 (.cpp 文件)
minicuda/
  __init__.py      Python 包入口，从 _C 导出符号
setup.py           构建配置，描述如何编译和打包
```

- `csrc/` 放纯 CUDA/C++ 实现，只关心计算逻辑。
- `binding/` 放 pybind11 绑定代码，负责把 C++ 函数注册为 Python 可调用的接口。
- `minicuda/__init__.py` 是 Python 包的入口，通过 `from minicuda._C import *` 把底层 C++ 接口导出到顶层命名空间，用户直接 `from minicuda import xxx` 即可使用。

## setup.py 的作用

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="minicuda",
    packages=["minicuda"],
    ext_modules=[
        CUDAExtension(
            name="minicuda._C",
            sources=[...],     # csrc/*.cu + binding/*.cpp
            include_dirs=["include"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

关键组件：

- **CUDAExtension**: PyTorch 提供的 setuptools.Extension 子类，自动处理 nvcc 路径、PyTorch 头文件、pybind11 头文件、CUDA toolkit 链接库等配置。如果不用它，这些都需要手动配置。
- **BuildExtension**: 替代 setuptools 默认的编译流程，核心作用是让 `.cu` 文件走 nvcc 编译，`.cpp` 文件走 g++/clang++ 编译。
- **name="minicuda._C"**: 编译产物的模块路径。编译后生成 `minicuda/_C.cpython-xxx.so`，作为 `minicuda` 包的子模块存在。`_C` 是 PyTorch 社区的惯例命名（PyTorch 自身的 C++ 后端也叫 `torch._C`），下划线前缀表示内部实现。
- **packages=["minicuda"]**: 告诉 setuptools `minicuda/` 是一个 Python 包，需要一起安装。

## 编译过程

执行 `pip install -e .` 后，实际发生以下步骤：

### Step 1: 分文件编译

BuildExtension 按文件后缀分发到不同编译器：

```bash
# .cu 文件 -> nvcc
nvcc -c csrc/vector_add.cu -o build/vector_add.o \
     -I torch/include -I cuda/include \
     --expt-relaxed-constexpr -O2

# .cpp 文件 -> g++
g++ -c binding/bind.cpp -o build/bind.o \
    -I torch/include -I pybind11/include \
    -std=c++17 -fPIC -O2
```

每个源文件编译成一个 `.o` 目标文件。

### Step 2: 链接生成 .so

所有 `.o` 链接成一个共享库：

```bash
g++ -shared build/vector_add.o build/bind.o \
    -L torch/lib -ltorch -lc10 \
    -L cuda/lib64 -lcudart \
    -o minicuda/_C.cpython-311-x86_64-linux-gnu.so
```

这个 `.so` 是一个 Python C 扩展模块，内部通过 pybind11 注册了 Python 可调用的函数。

### Step 3: 安装 Python 包

- 开发模式 (`pip install -e .`): 在 `site-packages/` 下创建 `.egg-link` 文件指向项目目录，不拷贝文件。改代码后重新 build 即可，不需要重装。
- 正式安装 (`pip install .`): 将 `minicuda/` 和 `.so` 拷贝到 `site-packages/minicuda/`。

## 生成 .whl 分发包

```bash
pip wheel . -w dist/
```

产物是一个平台相关的 wheel 文件，因为包含编译好的 `.so`，不能跨平台使用：

```
dist/minicuda-0.0.0-cp311-cp311-linux_x86_64.whl
```

whl 本质是 zip，内容：

```
minicuda/__init__.py
minicuda/_C.cpython-311-x86_64-linux-gnu.so
minicuda-0.0.0.dist-info/METADATA
minicuda-0.0.0.dist-info/RECORD
```

## 用户调用链路

```python
from minicuda import vector_add
result = vector_add(x, y)
```

Python 解释器的完整执行过程：

```
import minicuda
  -> 找到 site-packages/minicuda/__init__.py
  -> 执行 from minicuda._C import *
    -> 找到 _C.cpython-xxx.so
    -> dlopen() 加载 .so
    -> 调用 PyInit__C() (pybind11 生成的入口函数)
    -> 注册 vector_add 等函数到 Python 命名空间
  -> vector_add 可用

vector_add(x, y)
  -> pybind11 将 torch.Tensor 转为 C++ 的 at::Tensor
  -> 调用 C++ 函数
  -> C++ 函数内部启动 CUDA kernel <<<grid, block>>>(...)
  -> GPU 执行计算
  -> 结果写回 at::Tensor, 转回 torch.Tensor 返回
```

## 与纯 CMake 方案的对比

| | setup.py + CUDAExtension | 纯 CMake |
|---|---|---|
| Python 集成 | 自动处理 pybind11 + PyTorch 链接 | 需要手动 find_package(Torch) |
| 安装方式 | `pip install -e .` 一步完成 | cmake + make，手动管理 PYTHONPATH |
| 分发 | 可以打 wheel 分发 | 不支持 pip 分发 |
| 适合场景 | 最终要从 Python 调用 | 独立 C++/CUDA 可执行文件，或需要细粒度编译控制 |

## 类型导出机制

C 扩展 `_C.so` 是二进制产物，类型检查器（pyright/mypy）和 IDE 看不懂里面的函数签名。需要手写 `.pyi` stub 文件描述接口，类型检查器的查找顺序是 `module.pyi` → `module.py` → 编译产物（被忽略）。

### 基本结构

```
minicuda/
├── __init__.py    显式 import + __all__
├── _C.pyi         描述 _C.so 的函数签名
└── _C.cpython-xxx.so   编译产物
```

`_C.pyi`:

```python
import torch
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
```

`__init__.py`:

```python
from minicuda._C import vector_add
__all__ = ["vector_add"]
```

### 为什么不能用 `from _C import *`

类型检查器需要知道 `_C` 导出哪些符号才能展开 `*`。但 `_C.so` 是二进制不可读，`.pyi` 里如果没写 `__all__`，pyright 严格模式会拒绝展开，导致 `minicuda.vector_add` 类型变成 `Unknown`。**显式 import 一个个点名最稳**。

### 命名约束

- `.pyi` 文件名必须与所描述模块名一致（`_C.pyi` ↔ `minicuda._C` 模块）
- `_C` 这个名字来自 [setup.py](../setup.py) 中 `CUDAExtension(name="minicuda._C", ...)`，是 PyTorch/numpy/scipy 的社区惯例（`_` 表示内部实现，`C` 表示 C/C++ 后端）。改名需要 `setup.py`、`.pyi` 文件名、`__init__.py` 的 import 三处保持一致

### 拆分大型 stub

单个 `.so` 模块只能对应一个 `.pyi` 入口文件，但内容可拆到 stub-only 子目录中再 re-export：

```
minicuda/
├── _C.pyi              # 聚合入口
└── _stubs/             # 不带 __init__.py, 仅类型检查器可见
    ├── __init__.pyi
    ├── vector_ops.pyi
    └── matmul_ops.pyi
```

`_C.pyi` 只做 re-export：

```python
from minicuda._stubs.vector_ops import vector_add, vector_sub
from minicuda._stubs.matmul_ops import matmul
```

`_stubs/` 没有 `__init__.py`，运行时不存在，纯粹是给类型检查器组织代码用。
