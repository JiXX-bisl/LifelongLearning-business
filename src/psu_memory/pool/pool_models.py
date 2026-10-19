"""pool 数据结构的唯一实现位置；旧模块保留转发导入。

分类输入、难度输入、选择候选和成员快照各有用途，保留原字段和构造参数。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Literal

from ._common import (
    RECORD_FIELDS, record_fields, require_nonnegative_int, require_text_fields,
    user_questions,
)
from ._common import require_text as require_text

GroupId = Literal["detail", "aggregate", "compare", "rank"]
DifficultyLevel = Literal["L1", "L2", "L3", "L4"]
GROUP_IDS: tuple[GroupId, ...] = ("detail", "aggregate", "compare", "rank")
DIFFICULTY_LEVELS: tuple[DifficultyLevel, ...] = ("L1", "L2", "L3", "L4")
GROUP_LABELS: dict[GroupId, str] = {
    "detail": "明细获取", "aggregate": "统计汇总",
    "compare": "比较变化", "rank": "排序选取",
}
GROUP_RULE_VERSION = "action_group_v2"
DIFFICULTY_PROMPT_VERSION = "difficulty_v1"


def require_group_id(value: object) -> None:
    if not isinstance(value, str) or value not in GROUP_LABELS:
        raise ValueError("group_id 必须是 detail、aggregate、compare 或 rank")


class _Serializable:
    def to_dict(self) -> dict:
        return asdict(self)



# 分类与容量配置。


@dataclass(frozen=True)
class GroupingInput:
    """对经验内容进行分组"""
    execution_id: str
    content_version: str
    question: str
    namespace: str
    data_version: str

    # 本地历史适配器保存的上下文通常是消息列表的JSON文本
    context_text: str = ""

    # 当前问题的建议 后续由Vanna适配器传入
    data_advice: str = ""

    def __post_init__(self) -> None:
        require_text_fields(self, *RECORD_FIELDS, "question")
        for name in ("context_text", "data_advice"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} 必须是字符串")


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    group_id: GroupId
    matched_text: str


@dataclass(frozen=True)
class GroupOverride:
    """人工更正记录"""
    previous_group_id: GroupId | None
    group_id: GroupId
    reason: str
    actor: str


@dataclass(frozen=True)
class GroupingDecision(_Serializable):
    execution_id: str
    content_version: str
    namespace: str
    data_version: str
    group_id: GroupId | None
    input_sha256: str
    reason: str

    rule_version: str = GROUP_RULE_VERSION
    method: Literal["rule", "manual"] = "rule"
    text_source: Literal[
        "question", "data_advice", "context", "none"
    ] = "none"

    matches: tuple[RuleMatch, ...] = ()
    manual_history: tuple[GroupOverride, ...] = ()


@dataclass(frozen=True)
class PoolConfig:
    """每个粗组的总容量"""
    capacities: Mapping[str, int] = field(
        default_factory=lambda: {
            group_id: 8 for group_id in GROUP_IDS
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.capacities, Mapping):
            raise ValueError("capacities 必须是映射对象")
        if set(self.capacities) != set(GROUP_IDS):
            raise ValueError("capacities 必须且只能包含 detail、aggregate、compare、rank")
        checked = {group_id: self.capacities[group_id] for group_id in GROUP_IDS}
        for group_id, value in checked.items():
            require_nonnegative_int(value, f"{group_id} 的容量")
        object.__setattr__(self, "capacities", MappingProxyType(checked))

    def capacity_for(self, group_id: GroupId) -> int:
        require_group_id(group_id)
        return self.capacities[group_id]

    def to_dict(self) -> dict:
        return {"capacities": dict(self.capacities)}


# 难度评估：保持输入边界，避免将模型回答作为待评估任务。


@dataclass(frozen=True)
class DifficultyInput:
    execution_id: str
    content_version: str
    question: str
    namespace: str
    data_version: str

    context_questions: tuple[str, ...] = ()
    schema_context: str = ""
    data_advice: str = ""

    def __post_init__(self) -> None:
        require_text_fields(self, *RECORD_FIELDS, "question")
        if not isinstance(self.context_questions, tuple):
            raise ValueError("context_questions 必须是字符串元组")
        for question in self.context_questions:
            require_text(question, "context_questions 中的内容")
        for name in ("schema_context", "data_advice"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} 必须是字符串")

    def model_payload(self) -> dict:
        return {
            "question": self.question.strip(),
            "context_questions": [
                item.strip() for item in self.context_questions
            ],
            "schema_context": self.schema_context.strip(),
            "data_advice": self.data_advice.strip(),
        }

    @classmethod
    def from_grouping(cls, request: GroupingInput, *, schema_context: str = "") -> DifficultyInput:
        return cls(
            **record_fields(request), question=request.question,
            context_questions=user_questions(request.context_text),
            schema_context=schema_context, data_advice=request.data_advice,
        )


@dataclass(frozen=True)
class DifficultyUsage:
    elapsed_seconds: float = 0.0

    # 组件发起的generate调用次数
    model_call_count: int = 0

    # SDK 实际发出的HTTP请求及重试次数
    http_request_count: int | None = 0
    retry_count: int | None = 0

    # 服务端返回的用量
    prompt_tokens: int | None = 0
    completion_tokens: int | None = 0

    token_usage_complete: bool = True


@dataclass(frozen=True)
class ModelReply:
    text: str
    usage: DifficultyUsage
    error_type: str | None = None

    # 调用阶段耗时、模型签名等信息；不保存凭据或异常正文。
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DifficultyAssessment:
    cache_key: str
    prompt_version: str
    prompt_sha256: str
    model_signature: str

    status: Literal["assessed", "undetermined", "failed"]
    level: DifficultyLevel | None
    reason: str
    missing_information: tuple[str, ...]

    origin_usage: DifficultyUsage


@dataclass(frozen=True)
class DifficultyResult(_Serializable):
    execution_id: str
    content_version: str
    namespace: str
    data_version: str

    assessment: DifficultyAssessment
    cache_hit: bool

    # 本次调用的成本。命中缓存时调用数及 Token 为零。
    usage: DifficultyUsage


# 选择阶段：PoolCandidate 是评价结果的投影，不是另一份历史记录。


@dataclass(frozen=True, order=True)
class ContentKey:
    """一条执行记录的一个内容版本。"""

    execution_id: str
    content_version: str

    def __post_init__(self) -> None:
        require_text_fields(self, "execution_id", "content_version")


@dataclass(frozen=True)
class PoolScope:
    namespace: str
    data_version: str
    group_id: GroupId

    def __post_init__(self) -> None:
        require_text(self.namespace, "namespace")
        require_text(self.data_version, "data_version")
        require_group_id(self.group_id)


@dataclass(frozen=True)
class PoolCandidate:
    execution_id: str
    content_version: str
    namespace: str
    data_version: str

    admission_state: Literal["eligible", "suspended"]
    group_id: GroupId | None
    difficulty: DifficultyLevel | None

    validation_level: int
    completeness_count: int
    preview_chars: int
    quality_policy_version: str

    def __post_init__(self) -> None:
        require_text_fields(self, *RECORD_FIELDS, "quality_policy_version")
        if self.admission_state not in ("eligible", "suspended"):
            raise ValueError("准入状态只能是 eligible 或 suspended")
        if self.group_id is not None:
            require_group_id(self.group_id)
        if self.difficulty is not None and self.difficulty not in DIFFICULTY_LEVELS:
            raise ValueError("difficulty 必须是 L1–L4 或 None")
        for name in ("validation_level", "completeness_count", "preview_chars"):
            require_nonnegative_int(getattr(self, name), name)

    @property
    def key(self) -> ContentKey:
        return ContentKey(
            self.execution_id,
            self.content_version,
        )


@dataclass(frozen=True)
class PoolSnapshot:
    """某一组原有的成员集合。"""

    scope: PoolScope
    members: tuple[ContentKey, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, PoolScope):
            raise ValueError("scope 必须是 PoolScope")

        if (
            not isinstance(self.members, tuple)
            or any(
                not isinstance(item, ContentKey)
                for item in self.members
            )
        ):
            raise ValueError("members 必须是 ContentKey 元组")

        if len(set(self.members)) != len(self.members):
            raise ValueError("原成员快照不能包含重复内容")


@dataclass(frozen=True)
class RankValue:
    """排序核心值，按元组升序比较，越小越优。"""

    key: tuple[int | float, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, tuple) or not self.key:
            raise ValueError("排序 key 必须是非空数字元组")

        for value in self.key:
            if type(value) not in (int, float):
                raise ValueError("排序值必须是 int 或 float")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("排序值不能包含 NaN 或无穷大")

        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("排序 reasons 必须是非空字符串元组")

        for reason in self.reasons:
            require_text(reason, "排序依据")


@dataclass(frozen=True)
class RankedCandidate:
    candidate: PoolCandidate
    rank: RankValue
    incumbent: bool

    def order_key(self) -> tuple:
        """质量相同之后，才比较原成员身份与稳定标识。"""
        return (
            self.rank.key,
            0 if self.incumbent else 1,
            self.candidate.execution_id,
            self.candidate.content_version,
        )


@dataclass(frozen=True)
class Exclusion:
    key: ContentKey
    reason_code: str
    reason: str


@dataclass(frozen=True)
class SelectionPlan(_Serializable):
    scope: PoolScope
    capacity: int

    ranking_policy_version: str
    selection_policy_version: str

    input_count: int
    duplicates_collapsed: int
    selectable_count: int

    selected: tuple[ContentKey, ...]
    added: tuple[ContentKey, ...]
    retained: tuple[ContentKey, ...]
    removed: tuple[Exclusion, ...]
    excluded: tuple[Exclusion, ...]

    difficulty_counts: dict[str, int]
    unfilled_capacity: int
    rankings: tuple[RankedCandidate, ...]
    usage: dict


    def next_snapshot(self) -> PoolSnapshot:
        """返回建议的新快照，不执行任何保存操作。"""
        return PoolSnapshot(
            scope=self.scope,
            members=self.selected,
        )
