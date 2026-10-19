from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from ..history.history_adapter import read_history_file
from ..history.history_store import SQLiteHistoryStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="导入本地业务日志为候选经验档案"
    )
    parser.add_argument(
        "--source-root", type=Path, required=True
    )
    parser.add_argument(
        "--store", type=Path, required=True
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--data-version", required=True)
    args = parser.parse_args()

    root = args.source_root.resolve()
    if not root.is_dir():
        parser.error(f"日志目录不存在：{root}")

    if (
        not args.namespace.strip()
        or not args.data_version.strip()
    ):
        parser.error("namespace 和 data-version 不能为空")

    started = perf_counter()
    paths = sorted(root.glob("**/sessions/*/*.json"))
    if not paths:
        parser.error(
            "没有找到 sessions/<会话ID>/*.json 日志"
        )

    records = []
    errors = []

    for path in paths:
        try:
            records.append(read_history_file(
                path,
                namespace=args.namespace,
                data_version=args.data_version,
            ))
        except (OSError, ValueError) as error:
            errors.append({
                "source_ref": str(path),
                "reason": str(error),
            })

    store = SQLiteHistoryStore(args.store)
    report = store.import_batch(
        records,
        namespace=args.namespace,
        data_version=args.data_version,
        source_root=str(root),
        scanned_files=len(paths),
        errors=errors,
        started_at=started,
    )

    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ))

    return 2 if errors or report["conflicts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())