#!/usr/bin/env bash
# 用户运行入口：默认三组各跑前10题；PSU_DIRECT_CASE_LIMIT=72可运行完整测试分区。
# 准备新目录，顺序运行三组并打印汇总；独立测试另行执行。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

psu_direct_case_limit="${PSU_DIRECT_CASE_LIMIT:-10}"
export PSU_DIRECT_RUN="${PSU_DIRECT_RUN:-$PWD/data/derived/psu_memory/validation/natural-v030-seeded-dev-v0.1-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$PSU_DIRECT_RUN" || -L "$PSU_DIRECT_RUN" ]]; then
  printf '运行目录已存在，请将 PSU_DIRECT_RUN 指向全新目录或 unset PSU_DIRECT_RUN 后重试：%s\n' "$PSU_DIRECT_RUN" >&2
  exit 1
fi
printf '本轮目录：%s\n' "$PSU_DIRECT_RUN"
printf '每组题数：%s（按测试分区原顺序）\n' "$psu_direct_case_limit"

python -B -m src.psu_memory.cli.validate_local prepare \
  --config configs/validation/enterprise_v30_120_v1_0_2_natural_direct_v1.json \
  --run-dir "$PSU_DIRECT_RUN" \
  --case-limit "$psu_direct_case_limit"

for psu_direct_condition in baseline fifo adaptive; do
  python -B -m src.psu_memory.cli.validate_local run \
    --run-dir "$PSU_DIRECT_RUN" --mode "$psu_direct_condition"
done

python -B -m src.psu_memory.cli.validate_local summarize \
  --run-dir "$PSU_DIRECT_RUN"
