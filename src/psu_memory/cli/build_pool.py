"""python -m src.psu_memory.cli.build_pool ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.project_paths import resolve_project_path
from ..evaluation.quality_store import SQLiteQualityStore
from ..pool.difficulty import DifficultyAssessor
from ..pool.difficulty_llm import LocalDifficultyClient
from ..pool.maintenance import run_pool_maintenance
from ..pool.pool_models import PoolConfig
from ..pool.pool_store import SQLitePoolStore


def _show_progress(event: dict) -> None:
    label = (
        f"[{event['index']}/{event['total']}] "
        f"{event['execution_id'][-12:]}"
    )
    if event["stage"] == "start":
        message = f"{label} 准备标注"
    else:
        message = (
            f"{label} {event['operation']}"
            f"，难度={event.get('level')}"
            f"，缓存命中={event.get('cache_hit')}"
        )
    print(message, file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量准备 PSU 标注、更新经验池并生成成本报告"
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--pool-config", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--schema-context-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="只刷新质量并更新已有成员，不评估难度、不加载模型客户端",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.sync_only and args.model_config is None:
        parser.error("准备标注需要 --model-config；仅更新请使用 --sync-only")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须大于零")

    store_path = args.store.expanduser().resolve()
    if not store_path.is_file():
        parser.error(f"候选库不存在：{store_path}")

    protected = {store_path}
    for path in (
        args.pool_config,
        args.model_config,
        args.schema_context_file,
    ):
        if path is not None:
            protected.add(path.expanduser().resolve())

    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None else None
    )
    if report_path in protected:
        parser.error("报告路径不能覆盖候选库或配置输入文件")

    client = None
    try:
        config = None
        if args.pool_config is not None:
            payload = json.loads(
                args.pool_config.expanduser().read_text(encoding="utf-8")
            )
            config = PoolConfig(**payload)

        schema_context = ""
        if args.schema_context_file is not None and not args.sync_only:
            schema_context = args.schema_context_file.expanduser().read_text(
                encoding="utf-8"
            )

        pool = SQLitePoolStore(store_path)
        quality = SQLiteQualityStore(
            store_path,
            resolve_path=resolve_project_path,
        )

        assessor = None
        if not args.sync_only:
            client = LocalDifficultyClient.from_yaml(
                args.model_config.expanduser(),
                model_name=args.model_name,
            )
            assessor = DifficultyAssessor(client, cache=pool)

        report = run_pool_maintenance(
            quality_store=quality,
            pool_store=pool,
            namespace=args.namespace,
            data_version=args.data_version,
            assessor=assessor,
            config=config,
            sync_only=args.sync_only,
            limit=args.limit,
            schema_context=schema_context,
            retry_failed=args.retry_failed,
            progress=None if args.quiet else _show_progress,
        )

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )

        summary = dict(report)
        summary["report_file"] = (
            str(report_path) if report_path is not None else None
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 2 if report["error_count"] else 0
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())