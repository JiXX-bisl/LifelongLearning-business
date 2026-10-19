"""历史经验文本组织与字段级预算删减。

输入使用现有 QualityDecision.content 字典。
本模块不检索、不评价、不执行 SQL，也不调用模型。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from string import Template
from time import perf_counter

from ..models import UsageMetrics
from ..pool._common import (
    require_nonnegative_int,
    require_text,
    sha256_text,
)
from .retrieval_models import RetrievalConfig
from .text import build_matching_text


RENDER_RULE_VERSION = "experience_context_v1"

# 下一阶段拼接原提示词时，必须复用这个分隔符。
PROMPT_SEPARATOR = "\n\n"


def _optional_text(content: dict, name: str) -> str:
    value = content.get(name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是字符串")
    return value


def _sections(
    *,
    question: str | None,
    context_questions: tuple[str, ...],
    sql: str,
    notes: str,
    applicability: str,
) -> str:
    """只组织字段；SQL 和人工约束正文保持原样。"""
    sections = []

    if question is not None:
        sections.append("历史问题：\n" + question)

    if context_questions:
        sections.append(
            "历史用户上下文：\n"
            + "\n".join(context_questions)
        )

    sections.append("历史 SQL：\n" + sql)

    if notes.strip():
        sections.append("人工注意事项：\n" + notes)

    if applicability.strip():
        sections.append("适用范围：\n" + applicability)

    return "\n\n".join(sections)


def render_experience(
    content: dict,
    config: RetrievalConfig,
    *,
    max_context_chars: int | None = None,
    template_text: str | None = None,
) -> tuple[str | None, dict]:
    """返回经验块及诊断。

    文本不包含前置拼接分隔符，但预算包含该分隔符。
    max_context_chars=None 使用配置默认预算。
    内容不可用或预算不足时返回 None。
    非法配置对象或非法预算属于调用错误，抛出异常。
    """
    started = perf_counter()

    if not isinstance(config, RetrievalConfig):
        raise TypeError("config 必须是 RetrievalConfig")

    budget = (
        config.max_context_chars
        if max_context_chars is None
        else max_context_chars
    )
    require_nonnegative_int(budget, "max_context_chars")

    report = {
        "render_rule_version": RENDER_RULE_VERSION,
        "reason_code": "not_started",
        "render_level": "omitted",
        "effective_budget_chars": budget,
        "separator_chars": len(PROMPT_SEPARATOR),
        "content_chars": 0,
        "added_chars": 0,
        "full_required_chars": None,
        "minimum_required_chars": None,
        "omitted_fields": [],
        "template_sha256": None,
        "content_sha256": None,
        "sql_sha256": None,
        "attempts": [],
    }

    def finish(
        text: str | None,
        reason_code: str,
    ) -> tuple[str | None, dict]:
        report["reason_code"] = reason_code

        if text is not None:
            report["content_chars"] = len(text)
            report["added_chars"] = len(PROMPT_SEPARATOR) + len(text)
            report["content_sha256"] = sha256_text(text)

        # 只记录本次文本组织成本，不混入此前的择优成本。
        report["usage"] = {
            **asdict(UsageMetrics(
                elapsed_seconds=perf_counter() - started,
                model_call_count=0,
                prompt_tokens=0,
                completion_tokens=0,
                embedding_input_tokens=0,
            )),
            "http_request_count": 0,
            "retry_count": 0,
            "token_usage_complete": True,
        }
        return text, report

    # 停用时不读取经验内容或模板。
    if budget == 0:
        return finish(None, "disabled")

    try:
        if not isinstance(content, dict):
            raise ValueError("content 必须是已有经验内容字典")

        sql = content.get("sql")
        require_text(sql, "sql")

        notes = _optional_text(content, "notes")
        applicability = _optional_text(content, "applicability")
        context_text = _optional_text(content, "context_text")

        # 索引、查询、选择与注入共用同一显式会话规则。
        matching = build_matching_text(
            content.get("question"),
            context_text,
            max_context_questions=config.max_context_questions,
        )
    except (ValueError, TypeError, KeyError) as error:
        report["error_type"] = type(error).__name__
        return finish(None, "invalid_content")

    report["sql_sha256"] = sha256_text(sql)

    if not matching.ready:
        report["context_reason_code"] = matching.reason_code
        return finish(None, matching.reason_code)

    try:
        if template_text is None:
            template_path = (
                Path(__file__).parent
                / "prompts"
                / f"{RENDER_RULE_VERSION}.txt"
            )
            template_text = template_path.read_text(encoding="utf-8")

        require_text(template_text, "增强模板")
        template = Template(template_text)

        # 模板必须且只能包含一次 $sections，防止漏掉或重复 SQL。
        if (
            not template.is_valid()
            or template.get_identifiers() != ["sections"]
            or template_text.count("$sections") != 1
        ):
            raise ValueError("增强模板必须且只能包含一次 $sections")

        report["template_sha256"] = sha256_text(template_text)
    except OSError as error:
        report["error_type"] = type(error).__name__
        return finish(None, "template_unavailable")
    except (ValueError, TypeError) as error:
        report["error_type"] = type(error).__name__
        return finish(None, "template_invalid")

    original_question = matching.question
    original_context = matching.context_questions

    plans = [
        ("full", original_question, original_context),
        (
            "without_context",
            None if matching.needs_context else original_question,
            (),
        ),
        ("sql_only", None, ()),
    ]

    seen_texts = set()

    for level, question, context_questions in plans:
        # 历史上下文被移除时，一并省略问题，避免改变会话含义。
        if question is None:
            level = "sql_only"

        sections = _sections(
            question=question,
            context_questions=context_questions,
            sql=sql,
            notes=notes,
            applicability=applicability,
        )
        text = template.substitute(sections=sections)

        # 相同文本不重复尝试。
        if text in seen_texts:
            continue
        seen_texts.add(text)

        required_chars = len(PROMPT_SEPARATOR) + len(text)
        omitted_fields = []
        if original_context and not context_questions:
            omitted_fields.append("context_questions")
        if question is None:
            omitted_fields.append("question")

        if level == "full":
            report["full_required_chars"] = required_chars
        if level == "sql_only":
            report["minimum_required_chars"] = required_chars

        fits = required_chars <= budget
        report["attempts"].append({
            "render_level": level,
            "required_chars": required_chars,
            "omitted_fields": omitted_fields,
            "fits": fits,
        })

        if fits:
            report["render_level"] = level
            report["omitted_fields"] = omitted_fields
            return finish(text, "rendered")

    # 最低档也放不下，不截断 SQL 或人工约束。
    report["omitted_fields"] = ["experience"]
    return finish(None, "context_budget_exceeded")