#!/usr/bin/env bash
# 用户运行入口：expected核心完整正确，补充业务列须逐项核验。
# v3使用独立变量及新目录，不改变正在运行的v2脚本或目录。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

psu_coverage_case_limit="${PSU_COVERAGE_CASE_LIMIT:-10}"
export PSU_COVERAGE_RUN="${PSU_COVERAGE_RUN:-$PWD/data/derived/psu_memory/validation/natural-v030-coverage-v3-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$PSU_COVERAGE_RUN" || -L "$PSU_COVERAGE_RUN" ]]; then
  printf '运行目录已存在，请 unset PSU_COVERAGE_RUN 或指定全新目录：%s\n' "$PSU_COVERAGE_RUN" >&2
  exit 1
fi
printf '本轮目录：%s\n' "$PSU_COVERAGE_RUN"
printf '评分版本：expected_coverage_v3；每组题数：%s\n' "$psu_coverage_case_limit"

python -B -m src.psu_memory.cli.validate_local prepare \
  --config configs/validation/enterprise_v30_120_v1_0_2_natural_coverage_v3.json \
  --run-dir "$PSU_COVERAGE_RUN" --case-limit "$psu_coverage_case_limit"

for psu_coverage_condition in baseline fifo adaptive; do
  python -B -m src.psu_memory.cli.validate_local run \
    --run-dir "$PSU_COVERAGE_RUN" --mode "$psu_coverage_condition"
done

python -B -m src.psu_memory.cli.validate_local summarize \
  --run-dir "$PSU_COVERAGE_RUN"
