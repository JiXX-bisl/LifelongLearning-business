"""质量结论与反馈输入；准入状态只有 eligible / suspended。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

POLICY_VERSION = "local_sql_quality_v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass
class Feedback:
    execution_id: str
    target_version: str
    rating: Literal["helpful", "problem", "correction"]
    comment: str = ""
    error_reason: str = ""
    corrected_sql: str = ""
    confirmed: bool = False
    notes: str = ""
    applicability: str = ""
    actor: str = "local_user"
    event_id: str = ""  # 相同 ID 重放幂等；内容不同则拒绝。

    def payload(self):
        value = asdict(self)
        value.pop("event_id")
        return value

    def validate(self):
        for key, value in self.payload().items():
            if key == "confirmed":
                if type(value) is not bool:
                    raise ValueError("confirmed 必须是布尔值")
            elif not isinstance(value, str):
                raise ValueError(f"{key} 必须是字符串")
        if not self.execution_id or not self.target_version or not self.actor.strip():
            raise ValueError("必须指定经验、目标版本与反馈来源")
        if self.rating not in ("helpful", "problem", "correction"):
            raise ValueError("rating 必须是 helpful、problem 或 correction")
        if self.rating == "helpful" and (self.corrected_sql or self.confirmed):
            raise ValueError("修正内容请使用 correction 或 problem")
        if self.confirmed and not self.corrected_sql.strip():
            raise ValueError("确认修正版时必须提供 SQL")
        if not isinstance(self.event_id, str):
            raise ValueError("event_id 必须是字符串")


@dataclass
class QualityDecision:
    execution_id: str
    content_version: str
    admission_state: Literal["eligible", "suspended"]
    effective_correctness: bool | None
    reasons: list[str]
    basis: list[str]
    validation_level: int
    completeness: dict[str, bool]
    preview_text: str
    source: dict
    evidence_version: str | None
    content: dict
    historical_usage: dict
    policy_version: str = POLICY_VERSION
    feedback_event_ids: list[str] = field(default_factory=list)
    execution_verified: bool = False
    # 当前直接比较的准入证据，区别于历史 LLM 审核或 legacy 参考命中。
    direct_sql_result_verified: bool = False

    def to_dict(self):
        result = asdict(self)
        result["preview_chars"] = len(self.preview_text)
        result["estimated_injection_tokens"] = None
        return result


def ranking_key(decision):
    """状态过滤由调用者完成；没有时间先后或历史生成成本加分。"""
    return (-decision.validation_level, -sum(decision.completeness.values()),
            len(decision.preview_text), decision.execution_id, decision.content_version)
