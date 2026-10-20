"""难度评估流程与缓存接口；数据结构统一从 pool_models 导入。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol, cast

from ._common import canonical_json as _canonical, sha256_text as _sha256, record_fields
from ._common import user_questions as extract_user_questions
from .pool_models import (
    DIFFICULTY_LEVELS, DIFFICULTY_PROMPT_VERSION, DifficultyAssessment,
    DifficultyInput, DifficultyLevel, DifficultyResult, DifficultyUsage,
    GroupingInput, ModelReply, require_text,
)


class DifficultyClient(Protocol):
    signature: str

    def generate(
        self, 
        *, 
        system_prompt: str,
        user_text: str
    ) -> ModelReply:
        ...


class DifficultyCache(Protocol):
    def get(self, key: str) -> DifficultyAssessment | None:
        ...

    def put(self, assessment: DifficultyAssessment) -> None:
        ...


class MemoryDifficultyCache:
    def __init__(self) -> None:
        self._items: dict[str, DifficultyAssessment] = {}
    
    def get(self, key: str) -> DifficultyAssessment | None:
        return self._items.get(key)
    
    def put(self, assessment: DifficultyAssessment) -> None:
        self._items[assessment.cache_key] = assessment

    
def parse_difficulty_output(text: str) -> tuple[DifficultyLevel | None, str, tuple[str, ...]]:
    """校验输出"""
    if not isinstance(text, str) or len(text) > 12000:
        raise ValueError("模型输出必须是不超过 12000 字符的 JSON")
    
    try:
        payload = json.loads(text)
    except ValueError as error:
        raise ValueError("模型输出不是合法 JSON") from error

    expected_fields = {
        "level",
        "reason",
        "missing_information",
    }

    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
    ):
        raise ValueError(
            "必须且只能包含 level、reason、missing_information"
        )

    level = payload["level"]
    reason = payload["reason"]
    missing = payload["missing_information"]

    if level not in (*DIFFICULTY_LEVELS, "undetermined"):
        raise ValueError("level 必须是 L1–L4 或 undetermined")

    if (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason.strip()) > 300
    ):
        raise ValueError("reason 必须是 1–300 字的字符串")

    if not isinstance(missing, list) or len(missing) > 8:
        raise ValueError("missing_information 必须是最多 8 项的数组")

    for item in missing:
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item.strip()) > 160
        ):
            raise ValueError("缺失信息每项必须是 1–160 字的字符串")

    if level == "undetermined":
        if not missing:
            raise ValueError("undetermined 必须说明缺失信息")
        parsed_level = None
    else:
        if missing:
            raise ValueError("已确定等级时，缺失信息必须为空")
        parsed_level = cast(DifficultyLevel, level)

    return (
        parsed_level,
        reason.strip(),
        tuple(item.strip() for item in missing),
    )   


class DifficultyAssessor:
    def __init__(
        self,
        client: DifficultyClient,
        cache: DifficultyCache | None = None,
        *,
        prompt_text: str | None = None,
        prompt_version: str = DIFFICULTY_PROMPT_VERSION
    ) -> None:
        require_text(client.signature, "client.signature")
        require_text(prompt_version, "prompt_version")

        if prompt_text is None:
            prompt_path = (
                Path(__file__).parent
                / "prompts"
                / "difficulty_v2.txt"
            )
            prompt_text = prompt_path.read_text(encoding="utf-8")
        
        require_text(prompt_text, "prompt_text")

        self.client = client
        self.cache = (
            cache if cache is not None else MemoryDifficultyCache()
        )
        self.prompt_text = prompt_text
        self.prompt_version = prompt_version
        self.prompt_sha256 = _sha256(prompt_text)
        # 避免同一实例同时重复请求
        self._lock = threading.RLock()

    def cache_key(self, request: DifficultyInput) -> str:
        return _sha256(
            _canonical(
                {
                    "namespace": request.namespace,
                    "data_version": request.data_version,
                    "input": request.model_payload(),
                    "model_signature": self.client.signature,
                    "prompt_version": self.prompt_version,
                    "prompt_sha256": self.prompt_sha256, 
                }
            )
        )

    def assess(
        self, 
        request: DifficultyInput,
        *,
        admission_state: str,
        retry_failed: bool = False
    ) -> DifficultyResult:
        """评估 eligible 候选；不执行入池，也不修改准入状态。"""
        if admission_state != "eligible":
            raise ValueError("难度评估入口只接受eligible内容")
        if type(retry_failed) is not bool:
            raise ValueError("retry_failed 必须是 bool")
        with self._lock:
            started = time.perf_counter()
            key = self.cache_key(request)
            cached = self.cache.get(key)
            should_retry = (
                cached is not None
                and retry_failed
                and cached.status in ("failed", "undetermined")
            )

            if cached is not None and not should_retry:
                return self._result(
                    request, cached, cache_hit=True, started=started
                )
            
            try:
                reply = self.client.generate(
                    system_prompt=self.prompt_text,
                    user_text=_canonical(request.model_payload())
                )
            except Exception as error:
                # 不保存可能包含服务端信息的异常正文
                reply = ModelReply(
                    text="",
                    error_type=type(error).__name__,
                    usage=DifficultyUsage(
                        model_call_count=1,
                        http_request_count=None,
                        retry_count=None,
                        prompt_tokens=None,
                        completion_tokens=None,
                        token_usage_complete=False,
                    ),
                )

            level: DifficultyLevel | None = None
            missing: tuple[str, ...] = ()

            if reply.error_type:
                status = "failed"
                reason = f"模型调用失败: {reply.error_type}"
            else:
                try:
                    level, reason, missing = (
                        parse_difficulty_output(reply.text)
                    )
                    status = (
                        "assessed" if level is not None
                        else "undetermined"
                    )
                except ValueError as error:
                    status = "failed"
                    reason = f"模型输出校验失败: {error}"

            assessment = DifficultyAssessment(
                cache_key=key,
                prompt_version=self.prompt_version,
                prompt_sha256=self.prompt_sha256,
                model_signature=self.client.signature,
                status=cast(
                    Literal["assessed", "undetermined", "failed"],
                    status
                ),
                level=level,
                reason=reason,
                missing_information=missing,
                origin_usage=replace(
                    reply.usage, 
                    elapsed_seconds=time.perf_counter() - started
                )
            )

            attempt_usage = assessment.origin_usage
            self.cache.put(assessment=assessment)

            effective = self.cache.get(key)
            if effective is None:
                effective = assessment

            return self._result(
                request,
                effective,
                cache_hit=False,
                started=started,
                attempt_usage=attempt_usage
            )

    @staticmethod
    def _result(
        request: DifficultyInput,
        assessment: DifficultyAssessment,
        *,
        cache_hit: bool,
        started: float,
        attempt_usage: DifficultyUsage | None = None,
    ) -> DifficultyResult:
        elapsed = time.perf_counter() - started

        if attempt_usage is not None:
            current_usage = attempt_usage
        elif cache_hit:
            current_usage = DifficultyUsage()
        else:
            current_usage = assessment.origin_usage

        return DifficultyResult(
            **record_fields(request),
            assessment=assessment,
            cache_hit=cache_hit,
            usage=replace(
                current_usage,
                elapsed_seconds=elapsed,
            ),
        )
