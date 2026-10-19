"""整理语义初筛文本"""
from __future__ import annotations
from dataclasses import asdict

from ..models import RetrievalRequest
from ..pool._common import (
    MATCH_TEXT_RULE_VERSION,
    canonical_json,
    necessary_user_questions,
    needs_user_context,
    normalize_question_text,
    require_nonnegative_int,
    require_text,
    sha256_text,
)
from .retrieval_models import MatchingText, RetrievalConfig


def matching_text_sha256(
    text: str,
    *,
    rule_version: str = MATCH_TEXT_RULE_VERSION,
) -> str:
    """完整匹配文本与超限降级文本使用同一套指纹算法。"""

    return sha256_text(canonical_json({
        "rule_version": rule_version,
        "text": text,
    }))

def validate_request(request: RetrievalRequest) -> None:
    if not isinstance(request, RetrievalRequest):
        raise TypeError("request格式不匹配")
    for name in ("task_id", "question", "namespace", "data_version"):
        require_text(getattr(request, name), name=name)
    
    for name in ("context_text", "data_advice"):
        if not isinstance(getattr(request, name), str):
            raise ValueError(f"{name}必须是字符串")
        
    require_nonnegative_int(request.limit, "limit")
    if request.max_context_chars is not None:
        require_nonnegative_int(
            request.max_context_chars, "max_context_chars"
        )

def build_matching_text(question: str, context_text: str = "", *, max_context_questions: int = 2) -> MatchingText:
    """将当前问题与历史问题使用该函数生成初筛样本"""
    current = normalize_question_text(question)
    if not isinstance(context_text, str):
        raise ValueError("context_text 必须是字符串")
    require_nonnegative_int(max_context_questions, "max_context_quesitons")
    required = needs_user_context(current)
    selected: tuple[str, ...] = ()

    def unavailable(reason: str) -> MatchingText:
        return MatchingText(
            question=current,
            context_questions=selected,
            needs_context=required,
            text=None,
            input_sha256=None,
            reason_code=reason
        )
    
    if required:
        try:
            selected = necessary_user_questions(
                current, context_text, max_questions=max_context_questions, strict=True
            )
        except ValueError:
            return unavailable("invalid_context")
        
        if not selected:
            return unavailable("missing_context")
    
        if needs_user_context(selected[0]):
            return unavailable("incomplete_context_chain")
    
    if selected:
        text = "\n".join([
            "必要历史用户问题: ",
            *selected,
            "当前用户问题: " + current
        ])
        reason = "with_user_context"
    else:
        text = current
        reason = "question_only"
        
    fingerprint = matching_text_sha256(text)
    return MatchingText(
        question=current,
        context_questions=selected,
        needs_context=required,
        text=text,
        input_sha256=fingerprint,
        reason_code=reason
    )

def build_request_text(request: RetrievalRequest, config: RetrievalConfig) -> MatchingText:
    validate_request(request=request)
    return build_matching_text(
        request.question,
        request.context_text,
        max_context_questions=config.max_context_questions
    )

def retrieval_request_fingerprint(request: RetrievalRequest) -> str:
    """调用方完成请求校验后，生成完整请求指纹。"""

    return sha256_text(canonical_json(asdict(request)))

    