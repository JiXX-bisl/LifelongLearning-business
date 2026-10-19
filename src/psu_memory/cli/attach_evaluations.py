from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.project_paths import resolve_project_path
from ..evaluation.evaluation_evidence import attach_evaluations


def main() -> int:
    parser = argparse.ArgumentParser(description="关联历史经验的评价证据")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--data-version", required=True)
    args = parser.parse_args()
    report = attach_evaluations(
        args.store, args.source_root,
        namespace=args.namespace,
        data_version=args.data_version,
        resolve_path=resolve_project_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
