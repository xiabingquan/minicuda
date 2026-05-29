#!/bin/bash
# 编译并安装 minicuda 包到 .venv, 然后跑精度测试
# 用法: ./test.sh [pytest额外参数]

set -e

# 增量编译 (复用 build/ 下的 .o 缓存) + inplace 安装到当前目录
uv run python setup.py build_ext --inplace

uv run pytest tests/ "$@"
