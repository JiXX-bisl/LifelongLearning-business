#!/usr/bin/env bash
# 用户运行入口：按最终结果值对齐，不比较SQL写法、列名或表格布局。
# 依据题目确定必要值并认可已确认的等价状态表示；使用全新运行目录。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

psu_values_case_limit="${PSU_VALUES_CASE_LIMIT:-10}"
export PSU_VALUES_RUN="${PSU_VALUES_RUN:-$PWD/data/derived/psu_memory/validation/natural-v030-subset-v2-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$PSU_VALUES_RUN" || -L "$PSU_VALUES_RUN" ]]; then
  printf '运行目录已存在，请 unset PSU_VALUES_RUN 或指定全新目录：%s\n' "$PSU_VALUES_RUN" >&2
  exit 1
fi
printf '本轮目录：%s\n' "$PSU_VALUES_RUN"
printf '评分版本：expected_subset_v2；每组题数：%s\n' "$psu_values_case_limit"

python -B -m src.psu_memory.cli.validate_local prepare \
  --config configs/validation/enterprise_v30_120_v1_0_2_natural_values.json \
  --run-dir "$PSU_VALUES_RUN" --case-limit "$psu_values_case_limit"

for psu_values_condition in baseline fifo adaptive; do
  python -B -m src.psu_memory.cli.validate_local run \
    --run-dir "$PSU_VALUES_RUN" --mode "$psu_values_condition"
done

python -B -m src.psu_memory.cli.validate_local summarize \
  --run-dir "$PSU_VALUES_RUN"
