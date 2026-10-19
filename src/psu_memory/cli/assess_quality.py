"""python -m src.psu_memory.cli.assess_quality ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.project_paths import resolve_project_path
from ..evaluation.quality_store import SQLiteQualityStore


def main():
    parser = argparse.ArgumentParser(description="评价历史候选的二态准入资格并生成排序")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--report", type=Path, help="可选：完整 JSON 报告的输出位置")
    args = parser.parse_args()
    report = SQLiteQualityStore(args.store, resolve_path=resolve_project_path).assess_batch(
        namespace=args.namespace, data_version=args.data_version)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key not in ("decisions", "ranked_execution_ids")}
    summary["ranked_preview"] = report["ranked_execution_ids"][:5]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
