#!/usr/bin/env bash
# 用户独立测试入口：只运行评分和实验流程两个核心验证，外部服务均使用模拟对象。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python -B -m pytest -q \
  test/test_psu_submission_comparison.py \
  test/test_psu_direct_runner.py
