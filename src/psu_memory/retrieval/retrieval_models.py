"""检索内部结构的唯一定义位置"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.project_paths import resolve_project_path
from ..models import UsageMetrics
from ..pool.pool_models import DifficultyUsage
from ..pool._common import (
    MATCH_TEXT_RULE_VERSION,
    require_nonnegative_int,
    require_text,
)


SELECTOR_PROMPT_VERSION = "selector_v1"


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    embedding_model_path: str = (
        "data/models/huggingface/models--BAAI--bge-m3/snapshots/"
        "5617a9f61b028005a4858fdac845db406aefb181"
    )

    device: str = "mps"
    embedding_dimension: int = 1024
    embedding_dtype: str = "float32"
    pooling_method: str = "cls"
    normalize_embeddings: bool = True
    embedding_max_tokens: int = 1024
    embedding_batch_size: int = 4

    max_context_questions: int = 2
    prefilter_limit: int = 3
    max_context_chars: int = 3000
    selector_timeout_seconds: float = 30.0
    selector_max_retries: int = 0

    selector_max_input_chars: int = 12000
    selector_max_output_tokens: int = 512
    selector_reason_max_chars: int = 160

    selector_model_name: str = "kimi-k2.6"
    selector_temperature: float = 0.6
    selector_extra_body: dict = field(
        default_factory=lambda: {
            "thinking": {"type": "disabled"},
        },
    )

    def __post_init__(self) -> None:
        for name in (
            "embedding_model_name",
            "embedding_revision",
            "embedding_model_path",
            "selector_model_name"
        ):
            require_text(getattr(self, name), name)
        
        for name in (
            "embedding_dimension",
            "embedding_max_tokens",
            "embedding_batch_size",
            "prefilter_limit",
            "max_context_chars",
            "selector_max_input_chars",
            "selector_max_output_tokens",
            "selector_reason_max_chars",
        ):
            value = getattr(self, name)
            require_nonnegative_int(value, name)
            if value == 0:
                raise ValueError(f"{name} 必须大于零")
        
        require_nonnegative_int(self.max_context_questions, "max_context_questions")
        if self.max_context_questions > 2:
            raise ValueError("max_context_questions 最多为 2")
        require_nonnegative_int(self.selector_max_retries, "selector_max_retries")

        if self.selector_max_retries != 0:
            raise ValueError("尚未配置自动重试功能")
        
        timeout = self.selector_timeout_seconds
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("selector_timeout_seconds 必须是正有限数")
        
        temperature = self.selector_temperature
        if (
            type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError(
                "selector_temperature 必须是 0～2 的有限数"
            )

        if not isinstance(self.selector_extra_body, dict):
            raise ValueError("selector_extra_body 必须是 JSON 对象")

        try:
            json.dumps(
                self.selector_extra_body,
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "selector_extra_body 必须能够序列化为合法 JSON"
            ) from error
            
        if self.device not in ("mps", "cpu") and (
            not isinstance(self.device, str)
            or re.fullmatch(r"cuda:(?:0|[1-9][0-9]*)", self.device) is None
        ):
            raise ValueError("device 仅支持 mps、cpu 或 cuda:N（N 为非负整数）")
        if self.embedding_dtype != "float32":
            raise ValueError("当前 embedding_dtype 使用 float32")
        if self.pooling_method != "cls":
            raise ValueError("当前 BGE-M3 使用 cls 池化")
        if self.normalize_embeddings is not True:
            raise ValueError("当前向量必须进行 L2 归一化")

    @property
    def model_path(self) -> Path:
        return resolve_project_path(self.embedding_model_path)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> RetrievalConfig:
        payload = json.loads(
            resolve_project_path(path).read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("检索配置必须是 JSON 对象")
        return cls(**payload)
    
@dataclass
class MatchingText:
    """可编码文本及来源说明"""
    question: str
    context_questions: tuple[str, ...]
    needs_context: bool
    text: str | None
    input_sha256: str | None
    reason_code: str
    rule_version: str = MATCH_TEXT_RULE_VERSION

    @property
    def ready(self) -> bool:
        return self.text is not None
    
    def to_dict(self) -> dict:
        return {**asdict(self), "ready": self.ready}
    

@dataclass(frozen=True)
class PreparedEmbedding:
    """沿用 MatchingText，补充实际编码长度和表示方式"""
    matching: MatchingText
    token_count: int | None
    original_token_count: int | None
    mode: str

    @property
    def ready(self) -> bool:
        return self.matching.ready and self.token_count is not None
    

@dataclass
class EmbeddingUsage(UsageMetrics):
    """基础成本字段沿用 UsageMetrics，模型调用使用 LLM 调用"""
    tokenizer_load_seconds: float = 0.0
    model_load_seconds: float = 0.0
    encoding_seconds: float = 0.0
    cache_read_seconds: float = 0.0
    cache_write_seconds: float = 0.0

    model_load_count: int = 0
    embedding_batch_count: int = 0
    encoded_text_count: int = 0

    cache_hit_count: int = 0
    cache_miss_count: int = 0
    deduplicated_count: int = 0
    corrupted_cache_count: int = 0
    inserted_cache_count: int = 0
    repaired_cache_count: int = 0

    unavailable_count: int = 0


@dataclass
class EmbeddingItem:
    prepared: PreparedEmbedding
    cache_key: str | None = None
    vector: tuple[float, ...] | None = None
    cache_hit: bool = False
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.vector is not None and self.error is None

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "dimension": len(self.vector) if self.vector is not None else None,
            "mode": self.prepared.mode,
            "reason": self.error or self.prepared.matching.reason_code,
            "token_count": self.prepared.token_count,
            "original_token_count": self.prepared.original_token_count,
            "matching": self.prepared.matching.to_dict(),
        }


@dataclass
class EmbeddingBatch:
    items: list[EmbeddingItem]
    usage: EmbeddingUsage

    def to_dict(self) -> dict:
        return {
            "item_count": len(self.items),
            "ready_count": sum(item.ready for item in self.items),
            "items": [item.to_dict() for item in self.items],
            "usage": asdict(self.usage),
        }
    
def validate_dense_vector(
    values,
    dimension: int,
) -> tuple[float, ...]:
    """编码输出与缓存读取共用同一份向量校验。"""

    vector = tuple(float(value) for value in values)

    if len(vector) != dimension:
        raise ValueError("向量维度不匹配")

    if not all(math.isfinite(value) for value in vector):
        raise ValueError("向量包含非有限数")

    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError("向量未完成 L2 归一化")

    return vector

@dataclass
class PrefilterResult:
    """内部初筛结果；候选尚未经过 LLM 择优。"""

    status: str
    reason: str
    candidates: list[dict] = field(default_factory=list)
    routing: dict = field(default_factory=dict)
    query: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    usage: EmbeddingUsage = field(default_factory=lambda: EmbeddingUsage(
        elapsed_seconds=0.0,
        model_call_count=0,
        prompt_tokens=0,
        completion_tokens=0,
        embedding_input_tokens=0,
    ))

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "candidate_count": len(self.candidates),
            "candidates": [
                candidate_to_dict(item)
                for item in self.candidates
            ],
            "routing": self.routing,
            "query": self.query,
            "diagnostics": self.diagnostics,
            "stage_seconds": self.stage_seconds,
            "usage": asdict(self.usage),
        }
    
def candidate_to_dict(item: dict) -> dict:
    """初筛和择优结果共用候选序列化。"""

    member = item["member"]
    candidate = member["candidate"]
    quality = member["quality"]
    binding = item["binding"]
    rank = item["quality_rank"]

    return {
        "identity": member["identity"],
        "group_id": candidate.group_id,
        "difficulty": candidate.difficulty,
        "similarity": item["similarity"],
        "quality_rank_key": list(rank.key),
        "quality_rank_reasons": list(rank.reasons),
        "index_key": binding["index_key"],
        "cache_key": binding["cache_key"],
        "representation": binding["representation"],
        "annotation_sha256": member["annotation_sha256"],
        "quality_sha256": member["quality_sha256"],
        "admission_state": quality.admission_state,
        "effective_correctness": quality.effective_correctness,
        "admission_basis": quality.basis,
        "content": quality.content,
        "source": quality.source,
    }

@dataclass(frozen=True)
class SelectorPrompt:
    """准备好的选择输入及发送时的候选映射"""

    request_fingerprint: str
    reason_code: str
    system_prompt: str = ""
    user_text: str = ""
    alias_map: dict[str, dict] = field(default_factory=dict)
    input_chars: int = 0
    prompt_version: str = SELECTOR_PROMPT_VERSION
    prompt_sha256: str | None = None
    excluded_candidates: tuple[dict, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.reason_code == "ready" and bool(self.alias_map)

    def to_dict(self, *, include_text: bool = False) -> dict:
        result = {
            "ready": self.ready,
            "reason_code": self.reason_code,
            "request_fingerprint": self.request_fingerprint,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "input_chars": self.input_chars,
            "candidate_count": len(self.alias_map),
            "aliases": {
                alias: {
                    key: value
                    for key, value in snapshot.items()
                    if key != "candidate"
                }
                for alias, snapshot in self.alias_map.items()
            },
            "excluded_candidates": list(self.excluded_candidates),
            "diagnostics": self.diagnostics,
        }
        if include_text:
            result["system_prompt"] = self.system_prompt
            result["user_text"] = self.user_text
        return result

@dataclass
class SelectorResult:
    """选择协调器的结果；selected 为现有候选对象或 None。"""

    reason_code: str
    reason: str
    selected: dict | None = None
    selected_id: str | None = None
    request_fingerprint: str = ""
    prompt_version: str = SELECTOR_PROMPT_VERSION
    prompt_sha256: str | None = None
    diagnostics: dict = field(default_factory=dict)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    usage: DifficultyUsage = field(default_factory=DifficultyUsage)
    prefilter_usage: EmbeddingUsage | None = None

    def to_dict(self) -> dict:
        return {
            "selected": (
                candidate_to_dict(self.selected)
                if self.selected is not None else None
            ),
            "selected_id": self.selected_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "request_fingerprint": self.request_fingerprint,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "diagnostics": self.diagnostics,
            "stage_seconds": self.stage_seconds,
            "usage": asdict(self.usage),
            "prefilter_usage": (
                asdict(self.prefilter_usage)
                if self.prefilter_usage is not None else None
            ),
        }
