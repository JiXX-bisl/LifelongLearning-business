"""本地经验池批量协调；复用现有评价、分类、难度和选择规则。"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from ..evaluation.quality_store import SQLiteQualityStore
from ._common import require_nonnegative_int
from .classification import classify
from .difficulty import DifficultyAssessor
from .pool_models import (
    DifficultyInput,
    DifficultyUsage,
    GroupingInput,
    PoolConfig,
)
from .pool_store import SQLitePoolStore
from .selection import has_sql_result_pass


def _summarize_usage(
    usages: list[DifficultyUsage],
    *,
    elapsed_seconds: float,
) -> dict:
    """汇总本次返回的 usage；不累计缓存的 origin_usage。"""

    def total(name: str):
        values = [getattr(item, name) for item in usages]
        if any(value is None for value in values):
            return None
        return sum(values)

    prompt_tokens = total("prompt_tokens")
    completion_tokens = total("completion_tokens")

    complete = (
        all(item.token_usage_complete for item in usages)
        and prompt_tokens is not None
        and completion_tokens is not None
    )

    # 重试等情况可能只返回部分请求的 Token。
    # 此时不能把已返回部分宣称为完整总量。
    if not complete:
        prompt_tokens = None
        completion_tokens = None

    return {
        "elapsed_seconds": round(elapsed_seconds, 6),
        "model_call_count": total("model_call_count"),
        "http_request_count": total("http_request_count"),
        "retry_count": total("retry_count"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens
            if complete else None
        ),
        "embedding_input_tokens": 0,
        "token_usage_complete": complete,
        "reported_prompt_tokens": sum(
            item.prompt_tokens or 0 for item in usages
        ),
        "reported_completion_tokens": sum(
            item.completion_tokens or 0 for item in usages
        ),
    }


def run_pool_maintenance(
    *, quality_store: SQLiteQualityStore, pool_store: SQLitePoolStore,
    namespace: str, data_version: str, assessor: DifficultyAssessor | None = None,
    config: PoolConfig | None = None, sync_only: bool = False, limit: int | None = None,
    schema_context: str = "", retry_failed: bool = False,
    progress: Callable[[dict], None] | None = None,
    execution_ids: tuple[str, ...] | None = None,
    strict: bool = False, refresh_quality: bool = True,
) -> dict:
    """种子准备和在线更新共用同一标注、选择流程；仅返回简要状态与成本。"""
    if quality_store.path != pool_store.path:
        raise ValueError("质量存储与经验池存储必须使用同一候选库")
    if not sync_only and (assessor is None or assessor.cache is not pool_store):
        raise ValueError("准备标注必须提供以本次 pool_store 为缓存的 DifficultyAssessor")
    if limit is not None:
        require_nonnegative_int(limit, "limit")
        if not limit:
            raise ValueError("limit 必须大于零；仅更新成员请使用 sync_only")
    if execution_ids is not None and len(set(execution_ids)) != len(execution_ids):
        raise ValueError("execution_ids 不能重复")
    started = perf_counter()
    if refresh_quality:
        quality_store.assess_batch(namespace=namespace, data_version=data_version,
                                   execution_ids=execution_ids)
    pool_store.initialize()
    if execution_ids is None:
        entries = pool_store.load_inputs(namespace=namespace, data_version=data_version)
    else:
        entries = []
        for execution_id in execution_ids:
            found = pool_store.load_inputs(namespace=namespace, data_version=data_version,
                                           execution_id=execution_id)
            if not found:
                raise ValueError("指定标注执行不属于当前候选范围")
            entries.extend(found)
    if strict and any(not has_sql_result_pass(entry["quality"]) for entry in entries):
        raise ValueError("经验必须具有绑定实际 SQL 的结果 PASS 证据")
    eligible = [entry for entry in entries if has_sql_result_pass(entry["quality"])]
    chosen = [] if sync_only else eligible[:limit]
    usages = []
    cache_hits = changed_annotations = errors = 0
    prepared_ids = []
    for index, entry in enumerate(chosen, start=1):
        identity = entry["identity"]
        progress_base = {"index": index, "total": len(chosen),
                         "execution_id": identity["execution_id"]}
        if progress is not None:
            progress({**progress_base, "stage": "start"})
        result = None
        operation = "prepared"
        try:
            content = entry["quality"].content
            request = GroupingInput(**identity, question=content["question"],
                                    context_text=content.get("context_text") or "", sql=content["sql"])
            grouping = classify(request)
            result = assessor.assess(
                DifficultyInput.from_grouping(request, schema_context=schema_context),
                admission_state="eligible", retry_failed=retry_failed,
            )
            usages.append(result.usage)
            cache_hits += int(result.cache_hit)
            if result.assessment.status != "assessed":
                raise ValueError("经验难度尚未形成有效的 L1–L4 标注")
            saved = pool_store.save_annotation(entry, grouping=grouping, difficulty=result)
            changed_annotations += int(saved["changed"])
            prepared_ids.append(identity["execution_id"])
        except (ValueError, RuntimeError):
            if strict:
                raise
            operation = "preparation_failed"
            errors += 1
        if progress is not None:
            progress({**progress_base, "stage": "done", "operation": operation,
                      "cache_hit": result.cache_hit if result is not None else False,
                      "level": result.assessment.level if result is not None else None})
    synced = pool_store.sync(namespace=namespace, data_version=data_version, config=config,
                             execution_ids=tuple(prepared_ids))
    report = {
        "operation": "maintain", "namespace": namespace, "data_version": data_version,
        "sync_only": sync_only, "records_in_scope": len(entries),
        "preparation": {"selected_for_preparation": len(chosen), "prepared_count": len(prepared_ids),
                        "changed_annotations": changed_annotations, "cache_hits": cache_hits},
        "error_count": errors,
        "usage": _summarize_usage(usages, elapsed_seconds=perf_counter() - started),
        "sync": synced,
    }
    return pool_store.save_run_report(report) if pool_store.record_runs else report
