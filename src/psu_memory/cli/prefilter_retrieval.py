"""读取 RetrievalRequest JSON，执行语义初筛并生成报告。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from src.project_paths import resolve_project_path
from ..models import RetrievalRequest
from ..pool.pool_store import SQLitePoolStore
from ..retrieval.embedding_service import EmbeddingService
from ..retrieval.index_store import SQLiteRetrievalIndex
from ..retrieval.prefilter import SemanticPrefilter
from ..retrieval.retrieval_models import RetrievalConfig
from ..retrieval.vector_cache import SQLiteVectorCache


def main() -> int:
    parser = argparse.ArgumentParser(description="PSU 新问题语义初筛")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
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
    request_path = resolve_project_path(args.request)
    config_path = resolve_project_path(args.retrieval_config)
    report_path = (
        resolve_project_path(args.report)
        if args.report is not None else None
    )

    if not store_path.is_file():
        parser.error(f"候选库不存在：{store_path}")
    if report_path in {store_path, request_path, config_path}:
        parser.error("报告不能覆盖数据库或输入文件")

    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求必须是 JSON 对象")
        request = RetrievalRequest(**payload)
        config = RetrievalConfig.from_json(config_path)

        pool = SQLitePoolStore(store_path)
        cache = SQLiteVectorCache(pool)
        index = SQLiteRetrievalIndex(cache, config)

        # 检索不调用 initialize() 或离线构建命令。
        embeddings = EmbeddingService(config, cache)
        result = SemanticPrefilter(index, embeddings).prefilter(request)
        report = result.to_dict()

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
            "status": report["status"],
            "reason": report["reason"],
            "candidate_count": report["candidate_count"],
            "routing": report["routing"],
            "candidates": [
                {
                    "identity": item["identity"],
                    "group_id": item["group_id"],
                    "similarity": item["similarity"],
                    "question": item["content"]["question"],
                }
                for item in report["candidates"]
            ],
            "usage": report["usage"],
            "stage_seconds": report["stage_seconds"],
            "report_file": str(report_path) if report_path else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        return 2 if result.status in {
            "failed", "stale", "index_not_ready", "query_unavailable"
        } else 0

    except (OSError, ValueError, TypeError, sqlite3.Error) as error:
        print(json.dumps({
            "status": "failed",
            "error_type": type(error).__name__,
            "reason": str(error),
        }, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())