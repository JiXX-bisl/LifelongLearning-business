#!/usr/bin/env bash
# 用户运行入口：准备新目录，顺序运行三组并打印汇总；独立测试另行执行。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export PSU_DIRECT_RUN="${PSU_DIRECT_RUN:-$PWD/data/derived/psu_memory/validation/natural-v021-dev-v0.1-$(date +%Y%m%d-%H%M%S)}"
printf '本轮目录：%s\n' "$PSU_DIRECT_RUN"

if [[ ! -f "$PSU_DIRECT_RUN/config.json" ]]; then
  python -B -m src.psu_memory.cli.validate_local prepare \
    --config configs/validation/enterprise_v30_120_v1_0_2_natural_direct_v1.json \
    --run-dir "$PSU_DIRECT_RUN"
fi

for psu_direct_condition in baseline fifo adaptive; do
  python -B -m src.psu_memory.cli.validate_local run \
    --run-dir "$PSU_DIRECT_RUN" --mode "$psu_direct_condition"
done

python -B -m src.psu_memory.cli.validate_local summarize \
  --run-dir "$PSU_DIRECT_RUN"
