"""选择输入准备与输出解析；不调用模型、不访问数据库。"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

from ..models import RetrievalRequest
from ..pool._common import canonical_json, require_text, sha256_text
from ..pool.pool_models import ContentKey
from .retrieval_models import (
    PrefilterResult,
    RetrievalConfig,
    SELECTOR_PROMPT_VERSION,
    SelectorPrompt,
)
from .text import (
    build_matching_text,
    retrieval_request_fingerprint,
    validate_request,
)


def prepare_selector_prompt(
    request: RetrievalRequest,
    prefilter_result: PrefilterResult,
    config: RetrievalConfig,
    *,
    prompt_text: str | None = None,
) -> SelectorPrompt:
    """组织完整输入，按初筛顺序保留预算内的候选前缀。"""

    validate_request(request)
    if not isinstance(prefilter_result, PrefilterResult):
        raise TypeError("prefilter_result 必须是 PrefilterResult")

    fingerprint = retrieval_request_fingerprint(request)

    def unavailable(reason: str, **diagnostics) -> SelectorPrompt:
        return SelectorPrompt(
            request_fingerprint=fingerprint,
            reason_code=reason,
            diagnostics=diagnostics,
        )

    if request.limit == 0 or request.max_context_chars == 0:
        return unavailable("disabled")

    actual_fingerprint = prefilter_result.diagnostics.get(
        "request_fingerprint"
    )
    if actual_fingerprint is None:
        return unavailable("request_fingerprint_missing")
    if actual_fingerprint != fingerprint:
        return unavailable("request_mismatch")

    if prefilter_result.status != "ok":
        return unavailable(
            "prefilter_unavailable",
            prefilter_status=prefilter_result.status,
            prefilter_reason=prefilter_result.reason,
        )
    if not prefilter_result.candidates:
        return unavailable("no_candidates")

    # 从原请求恢复上下文，不使用向量降级后的文本。
    current = build_matching_text(
        request.question,
        request.context_text,
        max_context_questions=config.max_context_questions,
    )
    if not current.ready:
        return unavailable(current.reason_code)

    if prompt_text is None:
        path = (
            Path(__file__).parent
            / "prompts"
            / f"{SELECTOR_PROMPT_VERSION}.txt"
        )
        prompt_text = path.read_text(encoding="utf-8")
    require_text(prompt_text, "选择提示词")

    system_prompt = Template(prompt_text).substitute(
        reason_max_chars=config.selector_reason_max_chars,
    )
    prompt_sha256 = sha256_text(system_prompt)

    payload = {
        "current": {
            "question": current.question,
            "context_questions": list(current.context_questions),
            "data_advice": request.data_advice,
            "namespace": request.namespace,
            "data_version": request.data_version,
        },
        "candidates": [],
    }

    alias_map = {}
    excluded = []
    seen = set()
    candidates = prefilter_result.candidates[:config.prefilter_limit]

    for position, item in enumerate(candidates):
        member = item["member"]
        identity = member["identity"]

        if (
            identity["namespace"] != request.namespace
            or identity["data_version"] != request.data_version
        ):
            return unavailable("candidate_scope_mismatch")

        key = ContentKey(
            identity["execution_id"],
            identity["content_version"],
        )
        if key in seen:
            return unavailable("duplicate_candidate_identity")
        seen.add(key)

        content = member["quality"].content
        historical = build_matching_text(
            content["question"],
            content.get("context_text", ""),
            max_context_questions=config.max_context_questions,
        )
        if not historical.ready:
            return unavailable(
                "candidate_context_unavailable",
                identity=dict(identity),
                context_reason=historical.reason_code,
            )
        require_text(content["sql"], "候选 SQL")

        alias = f"c{len(alias_map) + 1}"
        candidate_payload = {
            "id": alias,
            "question": historical.question,
            "context_questions": list(historical.context_questions),
            "sql": content["sql"],
            "notes": content.get("notes", ""),
            "applicability": content.get("applicability", ""),
        }

        trial = {
            **payload,
            "candidates": [
                *payload["candidates"],
                candidate_payload,
            ],
        }
        trial_text = canonical_json(trial)
        trial_chars = len(system_prompt) + len(trial_text)

        if trial_chars > config.selector_max_input_chars:
            excluded.extend({
                "identity": dict(value["member"]["identity"]),
                "reason": "input_budget_exceeded",
            } for value in candidates[position:])
            break

        payload = trial
        alias_map[alias] = {
            "candidate": item,
            "identity": dict(identity),
            "annotation_sha256": member["annotation_sha256"],
            "quality_sha256": member["quality_sha256"],
            "index_key": item["binding"]["index_key"],
        }

    excluded.extend({
        "identity": dict(item["member"]["identity"]),
        "reason": "prefilter_limit",
    } for item in prefilter_result.candidates[config.prefilter_limit:])

    user_text = canonical_json(payload)
    input_chars = len(system_prompt) + len(user_text)

    return SelectorPrompt(
        request_fingerprint=fingerprint,
        reason_code="ready" if alias_map else "input_budget_exceeded",
        system_prompt=system_prompt,
        user_text=user_text,
        alias_map=alias_map,
        input_chars=input_chars,
        prompt_sha256=prompt_sha256,
        excluded_candidates=tuple(excluded),
        diagnostics={
            "prompt_template_sha256": sha256_text(prompt_text),
            "user_text_sha256": sha256_text(user_text),
            "input_char_budget": config.selector_max_input_chars,
            "original_candidate_count": len(prefilter_result.candidates),
        },
    )


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 字段重复：{key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"JSON 包含非法常量：{value}")


def parse_selection_output(
    text: str,
    prepared: SelectorPrompt,
    config: RetrievalConfig,
) -> tuple[str | None, str]:
    """返回模型选择的短标识及原因；不解析为实际可用经验。"""

    if not prepared.ready:
        raise ValueError("没有发送候选，不能接受选择结果")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("选择输出必须是非空字符串")
    if len(text) > 4096:
        raise ValueError("选择输出超过解析长度上限")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except ValueError as error:
        raise ValueError("选择输出不是合规 JSON") from error

    if (
        not isinstance(payload, dict)
        or set(payload) != {"selected_id", "reason"}
    ):
        raise ValueError("必须且只能包含 selected_id 和 reason")

    selected_id = payload["selected_id"]
    reason = payload["reason"]

    if selected_id is not None:
        if not isinstance(selected_id, str):
            raise ValueError("selected_id 必须是候选标识或 null")
        if selected_id not in prepared.alias_map:
            raise ValueError("selected_id 不属于发送的候选")

    require_text(reason, "选择原因")
    reason = reason.strip()
    if len(reason) > config.selector_reason_max_chars:
        raise ValueError("选择原因超过长度上限")

    return selected_id, reason