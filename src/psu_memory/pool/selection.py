"""经验池选择的纯计算策略；输入为完整候选、成员快照及当前版本。

RankingPolicy 可替换质量排序，SelectionPolicy 可替换难度组合/末位淘汰。
两种策略都不能越过外层的资格、内容版本、使用范围和容量约束。
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Mapping
from typing import Protocol

from ..evaluation.quality_models import QualityDecision
from ._common import record_fields, require_nonnegative_int
from .pool_models import (
    DIFFICULTY_LEVELS, ContentKey, DifficultyResult, Exclusion, GroupingDecision,
    PoolCandidate, PoolScope, PoolSnapshot, RankedCandidate, RankValue,
    SelectionPlan, require_text,
)


_REASON_TEXT = {
    "current_version_unknown": "缺少该经验的当前内容版本",
    "superseded_version": "该内容已被其他版本替代",
    "admission_suspended": "当前内容的准入状态为 suspended",
    "group_unassigned": "尚未确定粗分组",
    "outside_group": "当前分类不属于本经验池组",
    "difficulty_unavailable": "尚无有效的 L1–L4 难度标注",
    "group_disabled": "该组容量为零，当前停用",
    "not_selected_by_policy": "本次容量与组合策略未选中",
    "candidate_missing": "输入中缺少该成员的当前候选信息",
}


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
        admission_state=quality.admission_state,
        group_id=grouping.group_id if grouping else None,
        difficulty=level,
        validation_level=quality.validation_level,
        completeness_count=sum(quality.completeness.values()),
        preview_chars=len(quality.preview_text),
        quality_policy_version=quality.policy_version,
    )


class RankingPolicy(Protocol):
    """只返回核心排序值；不混入成员身份，不访问外部服务。"""

    version: str

    def rank(self, candidate: PoolCandidate) -> RankValue:
        ...


class SelectionPolicy(Protocol):
    """从已通过资格检查的候选中，决定容量内保留哪些成员。"""

    version: str

    def select(
        self, candidates: tuple[RankedCandidate, ...], *, capacity: int,
    ) -> tuple[RankedCandidate, ...]:
        ...


class BasicRankingPolicy:
    version = "base_quality_rank_v1"

    def rank(self, candidate: PoolCandidate) -> RankValue:
        return RankValue(
            key=(
                -candidate.validation_level,
                -candidate.completeness_count,
                candidate.preview_chars,
            ),
            reasons=(
                f"核验支持等级：{candidate.validation_level}",
                f"完整信息项数：{candidate.completeness_count}",
                f"经验预览字符数：{candidate.preview_chars}",
            ),
        )


class BalancedDifficultySelection:
    version = "balanced_difficulty_v1"

    def select(
        self, candidates: tuple[RankedCandidate, ...], *, capacity: int,
    ) -> tuple[RankedCandidate, ...]:
        # 一次排序、一次分桶，避免逐等级重复扫描全部候选。
        queues = {level: deque() for level in DIFFICULTY_LEVELS}
        for item in sorted(candidates, key=RankedCandidate.order_key):
            level = item.candidate.difficulty
            if level in queues:
                queues[level].append(item)
        counts = {level: 0 for level in DIFFICULTY_LEVELS}
        selected = []
        while len(selected) < capacity:
            available = [level for level, queue in queues.items() if queue]
            if not available:
                break
            # 先按当前数量，再比较各队列最优候选；不足的等级自然借出容量。
            level = min(
                available,
                key=lambda value: (counts[value], queues[value][0].order_key()),
            )
            selected.append(queues[level].popleft())
            counts[level] += 1
        return tuple(selected)


def _version_reason(key: ContentKey, heads: Mapping[str, str]) -> str | None:
    current = heads.get(key.execution_id)
    if current is None:
        return "current_version_unknown"
    if current != key.content_version:
        return "superseded_version"
    return None


def _blocking_reason(
    candidate: PoolCandidate, *, scope: PoolScope,
    current_versions: Mapping[str, str],
) -> str | None:
    reason = _version_reason(candidate.key, current_versions)
    if reason:
        return reason
    if candidate.admission_state == "suspended":
        return "admission_suspended"
    if candidate.group_id is None:
        return "group_unassigned"
    if candidate.group_id != scope.group_id:
        return "outside_group"
    if candidate.difficulty is None:
        return "difficulty_unavailable"
    return None


def _exclusion(key: ContentKey, code: str) -> Exclusion:
    return Exclusion(key=key, reason_code=code, reason=_REASON_TEXT[code])


def plan_group_update(
    *,
    snapshot: PoolSnapshot,
    candidates: Iterable[PoolCandidate],
    current_versions: Mapping[str, str],
    capacity: int,
    ranking_policy: RankingPolicy | None = None,
    selection_policy: SelectionPolicy | None = None,
) -> SelectionPlan:
    """计算完整更新计划；不能只传新增经验。结果不会写回候选或数据库。"""
    started = time.perf_counter()
    require_nonnegative_int(capacity, "capacity")
    if not isinstance(current_versions, Mapping):
        raise ValueError("current_versions 必须是映射")
    heads = dict(current_versions)
    for execution_id, version in heads.items():
        require_text(execution_id, "当前版本的 execution_id")
        require_text(version, "当前 content_version")

    ranking = ranking_policy if ranking_policy is not None else BasicRankingPolicy()
    selection = selection_policy if selection_policy is not None else BalancedDifficultySelection()
    require_text(ranking.version, "ranking.version")
    require_text(selection.version, "selection.version")

    scope = snapshot.scope
    input_items = tuple(candidates)
    unique: dict[ContentKey, PoolCandidate] = {}
    for candidate in input_items:
        if not isinstance(candidate, PoolCandidate):
            raise ValueError("candidates 只能包含 PoolCandidate")
        if (candidate.namespace, candidate.data_version) != (scope.namespace, scope.data_version):
            raise ValueError("候选与经验池的使用范围不一致")
        previous = unique.get(candidate.key)
        if previous is not None and previous != candidate:
            raise ValueError(f"同一执行记录和内容版本出现了冲突候选：{candidate.key}")
        unique[candidate.key] = candidate

    incumbents = set(snapshot.members)
    rejected: dict[ContentKey, str] = {}
    ranked = []
    for key, candidate in sorted(unique.items()):
        reason = _blocking_reason(candidate, scope=scope, current_versions=heads)
        if reason is None and capacity == 0:
            reason = "group_disabled"
        if reason is not None:
            rejected[key] = reason
            continue
        rank = ranking.rank(candidate)
        if not isinstance(rank, RankValue):
            raise ValueError("RankingPolicy 必须返回 RankValue")
        ranked.append(RankedCandidate(candidate, rank, key in incumbents))

    if len({len(item.rank.key) for item in ranked}) > 1:
        raise ValueError("同一排序策略返回了不同维数的 key")
    ranked_tuple = tuple(sorted(ranked, key=RankedCandidate.order_key))
    chosen = (
        tuple(selection.select(ranked_tuple, capacity=capacity))
        if ranked_tuple and capacity > 0 else ()
    )

    # 自定义策略返回的结果也必须再次检查，不能绕过硬约束。
    allowed = {item.candidate.key: item for item in ranked_tuple}
    selected_keys = []
    selected_set: set[ContentKey] = set()
    if len(chosen) > capacity:
        raise ValueError("SelectionPolicy 返回的成员超过容量")
    for item in chosen:
        if not isinstance(item, RankedCandidate):
            raise ValueError("SelectionPolicy 必须返回 RankedCandidate")
        key = item.candidate.key
        if allowed.get(key) != item:
            raise ValueError("SelectionPolicy 返回了非法候选")
        if key in selected_set:
            raise ValueError("SelectionPolicy 返回了重复成员")
        selected_set.add(key)
        selected_keys.append(key)

    rejected.update({
        key: "not_selected_by_policy" for key in allowed if key not in selected_set
    })
    removed = tuple(
        _exclusion(
            key, rejected.get(key) or _version_reason(key, heads) or "candidate_missing",
        )
        for key in sorted(incumbents - selected_set)
    )
    counts = {level: 0 for level in DIFFICULTY_LEVELS}
    for item in chosen:
        counts[item.candidate.difficulty] += 1

    return SelectionPlan(
        scope=scope, capacity=capacity,
        ranking_policy_version=ranking.version,
        selection_policy_version=selection.version,
        input_count=len(input_items),
        duplicates_collapsed=len(input_items) - len(unique),
        selectable_count=len(ranked_tuple),
        selected=tuple(selected_keys),
        added=tuple(key for key in selected_keys if key not in incumbents),
        retained=tuple(key for key in selected_keys if key in incumbents),
        removed=removed,
        excluded=tuple(_exclusion(key, rejected[key]) for key in sorted(rejected)),
        difficulty_counts=counts,
        unfilled_capacity=capacity - len(selected_keys),
        rankings=ranked_tuple,
        usage={
            "elapsed_seconds": time.perf_counter() - started,
            "model_call_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "embedding_input_tokens": 0,
        },
    )
