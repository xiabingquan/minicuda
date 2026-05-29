#!/bin/bash
# 编译单个 .cu 文件并链接 libtorch
# 修改下面的 SRC 切换目标文件

set -e

SRC="csrc/vector_add/a.cu"
OUT="/tmp/$(basename ${SRC%.cu})"

TORCH_DIR=".venv/lib/python3.12/site-packages/torch"

nvcc "$SRC" -o "$OUT" \
  -I include \
  -I "$TORCH_DIR/include" \
  -I "$TORCH_DIR/include/torch/csrc/api/include" \
  -L "$TORCH_DIR/lib" \
  -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda \
  -Xlinker -rpath -Xlinker "$TORCH_DIR/lib" \
  -std=c++17 -O2

echo "=== Running $OUT ==="
"$OUT"
