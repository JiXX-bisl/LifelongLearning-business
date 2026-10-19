"""新问题初筛与一次 LLM 择优的本地命令。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.project_paths import resolve_project_path
from ..models import RetrievalRequest
from ..pool.pool_store import SQLitePoolStore
from ..retrieval.embedding_service import EmbeddingService
from ..retrieval.index_store import SQLiteRetrievalIndex
from ..retrieval.prefilter import SemanticPrefilter
from ..retrieval.retrieval_models import RetrievalConfig
from ..retrieval.selector import LLMExperienceSelector
from ..retrieval.vector_cache import SQLiteVectorCache


def _create_client(model_config_path, config):
    # 延迟导入和创建：无可用候选时不读取模型凭据。
    from ..retrieval.selector_llm import LocalSelectorClient

    return LocalSelectorClient.from_yaml(
        model_config_path,
        retrieval_config=config,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PSU 一次 LLM 经验择优")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=Path(
            "src/psu_memory/retrieval/retrieval_config.example.json"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/business/query.yaml"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    store_path = resolve_project_path(args.store)
    request_path = resolve_project_path(args.request)
    config_path = resolve_project_path(args.retrieval_config)
    model_config_path = resolve_project_path(args.model_config)
    report_path = (
        resolve_project_path(args.report)
        if args.report is not None else None
    )

    if not store_path.is_file():
        parser.error(f"候选库不存在：{store_path}")

    protected = {
        store_path,
        Path(str(store_path) + "-wal"),
        Path(str(store_path) + "-shm"),
        Path(str(store_path) + "-journal"),
        request_path,
        config_path,
        model_config_path,
    }
    if report_path is not None:
        if report_path in protected:
            parser.error("报告不能覆盖数据库、输入文件或配置文件")
        if report_path.exists():
            parser.error("报告文件已经存在，请指定新的报告路径")

    started = perf_counter()

    try:
        payload = json.loads(
            request_path.read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("请求必须是 JSON 对象")

        request = RetrievalRequest(**payload)
        config = RetrievalConfig.from_json(config_path)

        pool = SQLitePoolStore(store_path)
        cache = SQLiteVectorCache(pool)
        index = SQLiteRetrievalIndex(cache, config)

        # 不初始化表，不重建索引；全流程只创建一个编码服务实例。
        embeddings = EmbeddingService(config, cache)
        prefilter = SemanticPrefilter(
            index,
            embeddings,
        ).prefilter(request)

        with LLMExperienceSelector(
            index,
            client_factory=lambda: _create_client(
                model_config_path,
                config,
            ),
        ) as selector:
            selection = selector.select(request, prefilter)

        if selection.selected is not None:
            status = "ok"
        elif (
            selection.reason_code in {"no_selection", "disabled"}
            or (
                selection.reason_code == "prefilter_unavailable"
                and prefilter.status in {
                    "no_members",
                    "no_encodable_members",
                }
            )
        ):
            status = "empty"
        else:
            status = "unavailable"

        report = {
            "status": status,
            "inputs": {
                "store": str(store_path),
                "request": str(request_path),
                "retrieval_config": str(config_path),
                "model_config": str(model_config_path),
            },
            "prefilter": prefilter.to_dict(),
            "selection": selection.to_dict(),
            "pipeline_elapsed_seconds": perf_counter() - started,
        }

        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            # x 模式也阻止检查后出现的同名文件被覆盖。
            with report_path.open("x", encoding="utf-8") as stream:
                json.dump(
                    report,
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")

        selected = report["selection"]["selected"]
        summary = {
            "status": status,
            "reason_code": selection.reason_code,
            "reason": selection.reason,
            "selected_id": selection.selected_id,
            "selected_identity": (
                selected["identity"] if selected is not None else None
            ),
            "sent_candidate_count": selection.diagnostics[
                "sent_candidate_count"
            ],
            "llm_usage": report["selection"]["usage"],
            "prefilter_usage": report["prefilter"]["usage"],
            "stage_seconds": selection.stage_seconds,
            "pipeline_elapsed_seconds": report["pipeline_elapsed_seconds"],
            "report_file": str(report_path) if report_path else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if status in {"ok", "empty"} else 2

    except Exception as error:
        # 不打印模型错误正文，避免输出连接信息或凭据。
        print(json.dumps({
            "status": "failed",
            "error_type": type(error).__name__,
            "reason": "本地择优命令执行失败。",
        }, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())