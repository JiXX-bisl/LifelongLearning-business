#!/usr/bin/env bash
# 用户独立测试入口；助手不执行。先回归v2及经验流程，再检查v3。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

bash scripts/enterprise_query_bench/run_direct_sql_tests.sh
python -B -m pytest -q \
  test/test_psu_coverage_comparison.py \
  test/test_psu_coverage_evidence.py \
  test/test_psu_coverage_runner.py
