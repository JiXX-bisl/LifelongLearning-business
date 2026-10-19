"""python -m src.psu_memory.cli.build_retrieval_index ..."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from src.project_paths import resolve_project_path
from ..pool.pool_store import SQLitePoolStore
from ..retrieval.embedding_service import EmbeddingService
from ..retrieval.index_store import SQLiteRetrievalIndex
from ..retrieval.indexing import OfflineIndexBuilder
from ..retrieval.retrieval_models import RetrievalConfig
from ..retrieval.vector_cache import SQLiteVectorCache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="离线构建当前有效 PSU 成员的双表示索引"
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=Path(
            "src/psu_memory/retrieval/retrieval_config.example.json"
        ),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    store_path = resolve_project_path(args.store)
    config_path = resolve_project_path(args.retrieval_config)
    report_path = (
        resolve_project_path(args.report)
        if args.report is not None else None
    )

    if not store_path.is_file():
        parser.error(f"候选库不存在：{store_path}")
    if report_path in {store_path, config_path}:
        parser.error("报告不能覆盖数据库或配置文件")

    try:
        config = RetrievalConfig.from_json(config_path)
        pool = SQLitePoolStore(store_path)
        cache = SQLiteVectorCache(pool)
        index = SQLiteRetrievalIndex(cache, config)
        index.initialize()

        # 整个构建过程共用这个服务实例。
        embeddings = EmbeddingService(config, cache)
        report = OfflineIndexBuilder(index, embeddings).build(
            namespace=args.namespace,
            data_version=args.data_version,
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

        summary = {
            key: value
            for key, value in report.items()
            if key not in ("bindings", "pending")
        }
        summary["report_file"] = (
            str(report_path) if report_path is not None else None
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "ok" else 2

    except (OSError, ValueError, sqlite3.Error) as error:
        print(json.dumps({
            "status": "failed",
            "error_type": type(error).__name__,
            "reason": str(error),
        }, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())