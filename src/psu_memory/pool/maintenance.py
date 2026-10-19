"""本地经验池批量协调；复用现有评价、分类、难度和选择规则。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from time import perf_counter

from ..evaluation.quality_store import SQLiteQualityStore
from ._common import require_nonnegative_int
from .classification import classify
from .difficulty import DifficultyAssessor
from .pool_models import (
    GROUP_RULE_VERSION,
    DifficultyInput,
    DifficultyUsage,
    GroupingInput,
    PoolConfig,
)
from .pool_store import SQLitePoolStore


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
    *,
    quality_store: SQLiteQualityStore,
    pool_store: SQLitePoolStore,
    namespace: str,
    data_version: str,
    assessor: DifficultyAssessor | None = None,
    config: PoolConfig | None = None,
    sync_only: bool = False,
    limit: int | None = None,
    schema_context: str = "",
    retry_failed: bool = False,
    progress: Callable[[dict], None] | None = None,
    execution_ids: tuple[str, ...] | None = None,
) -> dict:
    """处理已有候选并更新经验池，不负责导入日志或连接业务数据库。"""

    if quality_store.path != pool_store.path:
        raise ValueError("质量存储与经验池存储必须使用同一候选库")

    if not sync_only:
        if assessor is None:
            raise ValueError("准备标注时必须提供 DifficultyAssessor")
        if assessor.cache is not pool_store:
            raise ValueError("assessor.cache 必须使用本次 pool_store")

    if limit is not None:
        require_nonnegative_int(limit, "limit")
        if limit == 0:
            raise ValueError("limit 必须大于零；仅更新成员请使用 sync_only")

    started = perf_counter()
    stages = {
        "quality_seconds": 0.0,
        "initialize_and_load_seconds": 0.0,
        "classification_seconds": 0.0,
        "difficulty_seconds": 0.0,
        "annotation_save_seconds": 0.0,
        "sync_seconds": 0.0,
    }

    # 使用已有规则刷新资格，也为新导入但尚未评价的候选建立结论。
    phase = perf_counter()
    quality_report = quality_store.assess_batch(
        namespace=namespace,
        data_version=data_version,
    )
    stages["quality_seconds"] = perf_counter() - phase

    phase = perf_counter()
    pool_store.initialize()
    entries = pool_store.load_inputs(
        namespace=namespace,
        data_version=data_version,
    )
    stages["initialize_and_load_seconds"] = perf_counter() - phase

    eligible = [
        entry for entry in entries
        if entry["quality"].admission_state == "eligible"
    ]
    preparation_entries = eligible
    if execution_ids is not None:
        requested_ids = set(execution_ids)
        if not requested_ids <= {
            entry["identity"]["execution_id"] for entry in entries
        }:
            raise ValueError("指定标注执行不属于当前候选范围")
        preparation_entries = [entry for entry in eligible
                               if entry["identity"]["execution_id"] in requested_ids]
    chosen = [] if sync_only else preparation_entries[:limit]
    ordinals = {
        entry["identity"]["execution_id"]: index
        for index, entry in enumerate(chosen, start=1)
    }

    items = []
    usages: list[DifficultyUsage] = []
    cache_hits = 0
    changed_annotations = 0
    removed_by_annotation = 0

    for entry in entries:
        identity = entry["identity"]
        quality = entry["quality"]
        execution_id = identity["execution_id"]

        row = {
            **identity,
            "admission_state": quality.admission_state,
            "quality_sha256": entry["quality_sha256"],
            "quality_reasons": list(quality.reasons),
            "historical_execution_usage": quality.historical_usage,
        }

        if quality.admission_state != "eligible":
            row["operation"] = "skipped_suspended"
            items.append(row)
            continue

        if sync_only:
            row["operation"] = "sync_only"
            items.append(row)
            continue

        if execution_id not in ordinals:
            row["operation"] = "deferred_limit"
            items.append(row)
            continue

        progress_base = {
            "index": ordinals[execution_id],
            "total": len(chosen),
            "execution_id": execution_id,
        }
        if progress is not None:
            progress({**progress_base, "stage": "start"})

        phase = perf_counter()
        try:
            content = quality.content
            request = GroupingInput(
                **identity,
                question=content["question"],
                context_text=content.get("context_text") or "",
            )

            # 人工分类继续保留；普通分类用当前规则重新计算。
            previous = entry["grouping"]
            grouping = (
                previous
                if previous is not None and previous.method == "manual"
                else classify(request)
            )
            difficulty_input = DifficultyInput.from_grouping(
                request,
                schema_context=schema_context,
            )
        except ValueError as error:
            stages["classification_seconds"] += perf_counter() - phase
            row.update(
                operation="invalid_input",
                error=str(error),
            )
            items.append(row)
            if progress is not None:
                progress({
                    **progress_base,
                    "stage": "done",
                    "operation": row["operation"],
                })
            continue

        stages["classification_seconds"] += perf_counter() - phase
        row["grouping"] = grouping.to_dict()

        # 模型调用在写入标注、更新成员的事务之外发生。
        phase = perf_counter()
        result = assessor.assess(
            difficulty_input,
            admission_state=quality.admission_state,
            retry_failed=retry_failed,
        )
        stages["difficulty_seconds"] += perf_counter() - phase

        # 先记录本次成本，随后即使保存遇到版本冲突也不会漏计。
        usages.append(result.usage)
        cache_hits += int(result.cache_hit)
        row["difficulty"] = result.to_dict()

        phase = perf_counter()
        try:
            saved = pool_store.save_annotation(
                entry,
                grouping=grouping,
                difficulty=result,
            )
        except RuntimeError as error:
            row.update(
                operation="write_conflict",
                error=str(error),
            )
        else:
            row.update(
                operation="prepared",
                annotation_save=saved,
            )
            changed_annotations += int(saved["changed"])
            removed_by_annotation += saved["pool_removed_count"]

        stages["annotation_save_seconds"] += perf_counter() - phase
        items.append(row)

        if progress is not None:
            progress({
                **progress_base,
                "stage": "done",
                "operation": row["operation"],
                "cache_hit": result.cache_hit,
                "difficulty_status": result.assessment.status,
                "level": result.assessment.level,
            })

    # 即使本次只处理一部分候选，也使用全部已存标注进行成员更新。
    phase = perf_counter()
    sync_report = pool_store.sync(
        namespace=namespace,
        data_version=data_version,
        config=config,
    )
    stages["sync_seconds"] = perf_counter() - phase

    final_entries = pool_store.load_inputs(
        namespace=namespace,
        data_version=data_version,
    )
    final_eligible = [
        entry for entry in final_entries
        if entry["quality"].admission_state == "eligible"
    ]

    group_counts = Counter(
        entry["grouping"].group_id
        if entry["grouping"] is not None
        and entry["grouping"].group_id is not None
        else "unassigned"
        for entry in final_eligible
    )
    difficulty_counts = Counter(
        entry["difficulty"].assessment.level
        if entry["difficulty"] is not None
        and entry["difficulty"].assessment.status == "assessed"
        else "unavailable"
        for entry in final_eligible
    )

    difficulty_status_counts = Counter(
        row["difficulty"]["assessment"]["status"]
        for row in items if "difficulty" in row
    )
    error_count = sum(
        row["operation"] in ("invalid_input", "write_conflict")
        or row.get("difficulty", {}).get("assessment", {}).get("status")
        == "failed"
        for row in items
    )

    group_summaries = {}
    for plan in sync_report["plans"]:
        group_id = plan["scope"]["group_id"]
        group_summaries[group_id] = {
            "capacity": plan["capacity"],
            "selectable_count": plan["selectable_count"],
            "selected_count": len(plan["selected"]),
            "added_count": len(plan["added"]),
            "removed_count": len(plan["removed"]),
            "retained_count": len(plan["retained"]),
            "difficulty_counts": plan["difficulty_counts"],
            "unfilled_capacity": plan["unfilled_capacity"],
        }

    report = {
        "operation": "maintain",
        "namespace": namespace,
        "data_version": data_version,
        "sync_only": sync_only,
        "limit": limit,
        "records_in_scope": len(entries),
        "eligible_before_preparation": len(eligible),
        "quality": {
            key: quality_report[key]
            for key in (
                "run_id",
                "policy_version",
                "states",
                "reason_counts",
                "source_status_counts",
                "new_quality_versions",
                "refreshed_current_decisions",
                "pool_removed_count",
            )
        },
        "preparation": {
            "selected_for_preparation": len(chosen),
            "deferred_count": (
                len(eligible) - len(chosen) if not sync_only else 0
            ),
            "assessment_result_count": len(usages),
            "cache_hits": cache_hits,
            "cache_misses": len(usages) - cache_hits,
            "changed_annotations": changed_annotations,
            "difficulty_status_counts": dict(
                sorted(difficulty_status_counts.items())
            ),
        },
        "quality_after_processing": dict(sorted(Counter(
            entry["quality"].admission_state for entry in final_entries
        ).items())),
        "eligible_group_counts": dict(sorted(group_counts.items())),
        "eligible_difficulty_counts": dict(
            sorted(difficulty_counts.items())
        ),
        "group_summaries": group_summaries,
        "pool_changes": {
            "removed_by_quality": quality_report["pool_removed_count"],
            "removed_by_annotation": removed_by_annotation,
            "added_by_selection": sync_report["added_count"],
            "removed_by_selection": sync_report["removed_count"],
            "retained_by_selection": sync_report["retained_count"],
        },
        "error_count": error_count,
        "stage_seconds": {
            key: round(value, 6) for key, value in stages.items()
        },
        "usage": _summarize_usage(
            usages,
            elapsed_seconds=perf_counter() - started,
        ),
        "items": items,
        "sync": sync_report,
    }
    return pool_store.save_run_report(report)

def reclassify_existing_annotations(
    *,
    pool_store: SQLitePoolStore,
    namespace: str,
    data_version: str,
    config: PoolConfig | None = None
) -> dict:
    """更新分类并重新选择成员"""
    started = perf_counter()
    entries = pool_store.load_inputs(
        namespace=namespace,
        data_version=data_version
    )
    if not entries:
        raise ValueError("制定范围内没有已评价候选")
    
    def group_counts(values):
        return dict(sorted(Counter(
            entry["grouping"].group_id
            if entry["grouping"] is not None
            and entry["grouping"].group_id is not None
            else "unassigned"
            for entry in values
            if entry["quality"].admission_state == "eligible"
        ).items()))
    
    states_before = dict(sorted(Counter(
        entry["quality"].admission_state for entry in entries
    ).items()))

    groups_before = group_counts(entries)
    rows = []
    changed = 0
    removed = 0

    for entry in entries:
        if entry["quality"].admission_state != "eligible":
            continue
        row = {
            **entry["identity"],
            "previous_group_id": (
                entry["grouping"].group_id 
                if entry["grouping"] is not None else None
            )
        }
        difficulty = entry["difficulty"]
        if difficulty is None:
            row.update(
                operation="missing_difficulty",
                reason="没有保存的难度结果，未修改标注"
            )
            rows.append(row)
            continue

        previous = entry["grouping"]
        if previous is not None and previous.method == "manual":
            grouping = previous
        else:
            content = entry["quality"].content
            grouping = classify(GroupingInput(
                **entry["identity"],
                question=content["question"],
                context_text=content.get("context_text") or ""
            ))

        try:
            saved = pool_store.save_annotation(
                entry,
                grouping=grouping,
                difficulty=difficulty
            )
        except RuntimeError as error:
            row.update(
                operation="write_conflict",
                reason=str(error)
            )
        else:
            changed += int(saved["changed"])
            removed += saved["pool_removed_count"]
            row.update(
                operation="reclassified",
                group_id=grouping.group_id,
                method=grouping.method,
                changed=saved["changed"]
            )
        rows.append(row)

    sync_report = pool_store.sync(
        namespace=namespace,
        data_version=data_version,
        config=config
    )
    final = pool_store.load_inputs(
        namespace=namespace,
        data_version=data_version
    )

    report = {
        "operation": "reclassify",
        "namespace": namespace,
        "data_version": data_version,
        "classification_rule_version": GROUP_RULE_VERSION,
        "records_in_scope": len(entries),
        "quality_states_before": states_before,
        "quality_states_after": dict(sorted(Counter(
            entry["quality"].admission_state for entry in final
        ).items())),
        "eligible_groups_before": groups_before,
        "eligible_groups_after": group_counts(final),
        "changed_annotations": changed,
        "removed_by_annotation": removed,
        "selected_member_count": sum(
            len(plan["selected"]) for plan in sync_report["plans"]
        ),
        "error_count": sum(
            row["operation"] != "reclassified" for row in rows
        ),
        "items": rows,
        "sync": sync_report,
        "usage": _summarize_usage(
            [],
            elapsed_seconds=perf_counter() - started,
        ),
    }
    return pool_store.save_run_report(report)
