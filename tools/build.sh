#!/bin/bash
# 编译 tools/ 下的 benchmark
# 用法: ./tools/build.sh sgemm_bench [额外 nvcc 参数]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TORCH_DIR="$ROOT_DIR/.venv/lib/python3.12/site-packages/torch"

NAME="${1:?用法: $0 <name>  (例: sgemm_bench)}"
shift
SRC="$SCRIPT_DIR/${NAME}.cu"
OUT="$SCRIPT_DIR/$NAME"

[ -f "$SRC" ] || { echo "找不到 $SRC"; exit 1; }

echo "Compiling $NAME ..."

nvcc "$SRC" -o "$OUT" \
  -I "$ROOT_DIR/include" \
  -I "$TORCH_DIR/include" \
  -I "$TORCH_DIR/include/torch/csrc/api/include" \
  -L "$TORCH_DIR/lib" \
  -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda \
  -Xlinker -rpath -Xlinker "$TORCH_DIR/lib" \
  -std=c++17 -O2 \
  "$@"

echo "Done: $OUT"
