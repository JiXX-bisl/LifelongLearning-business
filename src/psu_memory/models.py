from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class UsageMetrics:
    """操作成本"""
    elapsed_seconds: float | None = None
    model_call_count: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    embedding_input_tokens: int | None = None


@dataclass 
class RetrievalRequest:
    """查询前由用户提交的经验检索请求"""

    task_id: str
    question: str

    # 指定经验隔离范围
    namespace: str
    # 当前数据结构版本
    data_version: str
    # 对话上下文
    context_text: str = ""
    # 可使用的经验数量上限
    limit: int = 1
    # 文本长度预算
    max_context_chars: int | None = None

    data_advice: str = ""


@dataclass
class ContextItem:
    """可以提供给 LLM 的内容。"""

    item_id: str
    content: str
    source_ref: str
    version: str
    kind: str = "experience"
    group_id: str | None = None
    scores: dict[str, float] = field(default_factory=dict)

    # 仅供追踪和后续使用前复核，不加入增强文本。
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索结果，包含参考内容、选择原因和本次成本。"""

    items: list[ContextItem] = field(default_factory=list)
    reason: str = ""
    usage: UsageMetrics = field(default_factory=UsageMetrics)

    # 新增字段均有默认值，兼容已有构造方式。
    reason_code: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompletedTask:
    """查询结束后回传的任务记录"""
    task_id: str
    question: str
    answer: str
    namespace: str
    data_version: str

    status: Literal["completed", "failed", "limit_reached"]

    context_text: str = ""

    # 实际执行SQL后返回结果的文本表示
    evidence_text: str = ""

    evaluation_verdict: Literal["pass", "revise", "unknown"] = "unknown"
    quality_scores: dict[str, float] = field(default_factory=dict)

    source_ref: str | None = None
    usage: UsageMetrics = field(default_factory=UsageMetrics)


@dataclass
class AdmissionResult:
    """任务回传后的处理结果"""

    # 是否已经保存该记录
    stored: bool

    # 是否进入当前可供检索的经验池
    admitted: bool

    reason: str
