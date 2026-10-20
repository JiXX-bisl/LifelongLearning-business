#!/usr/bin/env bash
# 用户运行入口：仅运行 baseline，固定测试分区原顺序的前3题。
# 使用独立配置和全新目录，不受三组启动脚本的环境变量影响。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

export PSU_BASELINE3_RUN="${PSU_BASELINE3_RUN:-$PWD/data/derived/psu_memory/validation/natural-v030-baseline3-dev-v0.1-$(date +%Y%m%d-%H%M%S)}"
if [[ -e "$PSU_BASELINE3_RUN" || -L "$PSU_BASELINE3_RUN" ]]; then
  printf '运行目录已存在，请将 PSU_BASELINE3_RUN 指向全新目录或 unset PSU_BASELINE3_RUN 后重试：%s\n' "$PSU_BASELINE3_RUN" >&2
  exit 1
fi
printf '本轮目录：%s\n' "$PSU_BASELINE3_RUN"
printf '本轮仅 baseline，固定前3题（按测试分区原顺序）。\n'

python -B -m src.psu_memory.cli.validate_local prepare \
  --config configs/validation/enterprise_v30_120_v1_0_2_natural_baseline3_v1.json \
  --run-dir "$PSU_BASELINE3_RUN" \
  --case-limit 3

python -B -m src.psu_memory.cli.validate_local run \
  --run-dir "$PSU_BASELINE3_RUN" --mode baseline

python -B -m src.psu_memory.cli.validate_local summarize \
  --run-dir "$PSU_BASELINE3_RUN"
