"""读取结构化反馈文件；成功返回前已完成停用或修正版状态更新。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.project_paths import resolve_project_path
from ..evaluation.quality_models import Feedback
from ..evaluation.quality_store import SQLiteQualityStore


def main():
    parser = argparse.ArgumentParser(description="向 PSU 组件提交反馈或明确确认的修正 SQL")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    if not isinstance(value, dict):
        parser.error("反馈文件必须是 JSON 对象")
    try:
        feedback = Feedback(**value)
        report = SQLiteQualityStore(args.store, resolve_path=resolve_project_path).submit_feedback(feedback)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
