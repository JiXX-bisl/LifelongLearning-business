"""确定性的成员选择：adaptive 质量排序与难度平衡、FIFO 插入顺序。"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

from ..evaluation.quality_models import QualityDecision
from ._common import record_fields, require_nonnegative_int
from .pool_models import (
    DIFFICULTY_LEVELS, GROUP_RULE_VERSION, ContentKey, DifficultyResult, GroupingDecision,
    PoolCandidate, PoolSnapshot, RankedCandidate, RankValue, SelectionPlan, require_text,
)


def has_sql_result_pass(quality: QualityDecision) -> bool:
    """SQL 结果核验是入池门槛；不改动原质量评分与排序。"""
    content = quality.content
    operation = content.get("operation", {})
    return bool(quality.admission_state == "eligible" and quality.execution_verified
                and quality.direct_sql_result_verified
                and "reference_result_match" in quality.basis
                and content.get("kind") != "correction"
                and operation.get("success") is True and operation.get("is_complete") is True
                and operation.get("query_id")
                and isinstance(content.get("sql"), str) and content["sql"].strip()
                and isinstance(operation.get("sql"), str)
                and operation["sql"].strip() == content["sql"].strip())


def candidate_from_evaluations(
    quality: QualityDecision,
    grouping: GroupingDecision | None,
    difficulty: DifficultyResult | None,
    *,
    namespace: str,
    data_version: str,
) -> PoolCandidate:
    """构建选择视图；不重新评价质量，也不保存另一份历史记录。"""
    require_text(namespace, "namespace")
    require_text(data_version, "data_version")
    expected = {
        "execution_id": quality.execution_id,
        "content_version": quality.content_version,
        "namespace": namespace,
        "data_version": data_version,
    }
    for name, result in (("grouping", grouping), ("difficulty", difficulty)):
        if result is not None and record_fields(result) != expected:
            raise ValueError(f"{name} 与质量结论的经验、版本或使用范围不一致")

    if not isinstance(quality.completeness, dict) or any(
        type(value) is not bool for value in quality.completeness.values()
    ):
        raise ValueError("completeness 必须是布尔值字典")
    if not isinstance(quality.preview_text, str):
        raise ValueError("preview_text 必须是字符串")

    level = None
    if difficulty is not None:
        assessment = difficulty.assessment
        if assessment.status == "assessed":
            if assessment.level not in DIFFICULTY_LEVELS:
                raise ValueError("assessed 必须具有有效难度等级")
            level = assessment.level
        elif assessment.status not in ("undetermined", "failed"):
            raise ValueError("未知的难度评估状态")

    return PoolCandidate(
        **expected,
        admission_state="eligible" if has_sql_result_pass(quality) else "suspended",
        group_id=(grouping.group_id if grouping and grouping.rule_version == GROUP_RULE_VERSION
                  and grouping.method == "rule" and grouping.text_source == "sql" else None),
        difficulty=level,
        validation_level=quality.validation_level,
        completeness_count=sum(quality.completeness.values()),
        preview_chars=len(quality.preview_text),
        quality_policy_version=quality.policy_version,
    )



class BasicRankingPolicy:
    """同时供池内排序和检索同分排序使用。"""
    version = "base_quality_rank_v1"

    def rank(self, candidate: PoolCandidate) -> RankValue:
        return RankValue(
            key=(-candidate.validation_level, -candidate.completeness_count, candidate.preview_chars),
            reasons=(f"核验支持等级：{candidate.validation_level}",
                     f"完整信息项数：{candidate.completeness_count}",
                     f"经验预览字符数：{candidate.preview_chars}"),
        )


def _current_eligible(candidate: PoolCandidate, current_versions: Mapping[str, str]) -> bool:
    return (current_versions.get(candidate.execution_id) == candidate.content_version
            and candidate.admission_state == "eligible"
            and candidate.group_id is not None and candidate.difficulty is not None)


def _unique_candidates(candidates, *, namespace, data_version):
    unique = {}
    for candidate in candidates:
        if not isinstance(candidate, PoolCandidate):
            raise ValueError("candidates 只能包含 PoolCandidate")
        if (candidate.namespace, candidate.data_version) != (namespace, data_version):
            raise ValueError("候选与经验池的使用范围不一致")
        previous = unique.get(candidate.key)
        if previous is not None and previous != candidate:
            raise ValueError("同一内容版本出现冲突候选")
        unique[candidate.key] = candidate
    return unique


def plan_group_update(*, snapshot: PoolSnapshot, candidates: Iterable[PoolCandidate],
                      current_versions: Mapping[str, str], capacity: int) -> SelectionPlan:
    """adaptive 总是从所有当前候选重选，已淘汰候选仍可再次入池。"""
    require_nonnegative_int(capacity, "capacity")
    scope = snapshot.scope
    unique = _unique_candidates(candidates, namespace=scope.namespace, data_version=scope.data_version)
    incumbents = set(snapshot.members)
    ranker = BasicRankingPolicy()
    ranked = sorted((RankedCandidate(candidate, ranker.rank(candidate), candidate.key in incumbents)
                     for candidate in unique.values()
                     if candidate.group_id == scope.group_id
                     and _current_eligible(candidate, current_versions)),
                    key=RankedCandidate.order_key)
    queues = {level: deque() for level in DIFFICULTY_LEVELS}
    for item in ranked:
        queues[item.candidate.difficulty].append(item)
    counts = {level: 0 for level in DIFFICULTY_LEVELS}
    selected = []
    while len(selected) < capacity:
        available = [level for level, queue in queues.items() if queue]
        if not available:
            break
        level = min(available, key=lambda value: (counts[value], queues[value][0].order_key()))
        selected.append(queues[level].popleft().candidate.key)
        counts[level] += 1
    selected_set = set(selected)
    return SelectionPlan(
        scope=scope, capacity=capacity, selected=tuple(selected),
        added=tuple(key for key in selected if key not in incumbents),
        retained=tuple(key for key in selected if key in incumbents),
        removed=tuple(sorted(incumbents - selected_set)),
        selectable_count=len(ranked), difficulty_counts=counts,
    )


def select_fifo_members(*, incumbents: tuple[ContentKey, ...],
                        candidates: Iterable[PoolCandidate], new_keys: tuple[ContentKey, ...],
                        current_versions: Mapping[str, str], namespace: str,
                        data_version: str, capacity: int) -> tuple[ContentKey, ...]:
    """现有成员保持入队顺序；只追加显式提交的新候选，不扫描历史补位。"""
    require_nonnegative_int(capacity, "capacity")
    if len(set(incumbents)) != len(incumbents):
        raise ValueError("FIFO 原成员不能重复")
    unique = _unique_candidates(candidates, namespace=namespace, data_version=data_version)
    queue = []
    seen = set()
    for key in (*incumbents, *new_keys):
        candidate = unique.get(key)
        if key not in seen and candidate is not None and _current_eligible(candidate, current_versions):
            queue.append(key)
            seen.add(key)
    return tuple(queue[-capacity:]) if capacity else ()
