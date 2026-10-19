"""pool 内部共用工具；不依赖业务模块，也不执行 I/O。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

RECORD_FIELDS = ("execution_id", "content_version", "namespace", "data_version")


def require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def require_text_fields(value: object, *names: str) -> None:
    for name in names:
        require_text(getattr(value, name), name)


def nonnegative_int_or_none(value: object) -> int | None:
    # bool 是 int 的子类，但不应被当作容量、计数或用量。
    return value if type(value) is int and value >= 0 else None


def require_nonnegative_int(value: object, name: str) -> None:
    if nonnegative_int_or_none(value) is None:
        raise ValueError(f"{name} 必须是非负整数")


def record_fields(value: object) -> dict[str, str]:
    """复制版本绑定信息；不将这些字段加入模型输入。"""
    return {name: getattr(value, name) for name in RECORD_FIELDS}


def canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def user_questions(context_text: str, *, strict: bool = True) -> tuple[str, ...]:
    """解析调用方显式提供的会话消息，只读取用户问题。"""
    if not isinstance(context_text, str):
        if strict:
            raise ValueError("context_text 必须是字符串")
        return ()
    if not context_text.strip():
        return ()
    try:
        messages = json.loads(context_text)
    except ValueError as error:
        if strict:
            raise ValueError("上下文必须是消息列表 JSON") from error
        return ()
    if not isinstance(messages, list):
        if strict:
            raise ValueError("上下文必须是消息列表 JSON")
        return ()

    questions = []
    for message in messages:
        if not isinstance(message, dict):
            if strict:
                raise ValueError("上下文中的消息必须是对象")
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if strict:
            require_text(content, "用户上下文内容")
        if isinstance(content, str) and content.strip():
            questions.append(content.strip() if strict else content)
    return tuple(questions)

MATCH_TEXT_RULE_VERSION = "question_context_v2"


def normalize_question_text(text: str) -> str:
    """统一 Unicode 与空白；不改写问题含义。"""
    require_text(text, "question")
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFC", text),
    ).strip()


def necessary_user_questions(
    question: str,
    context_text: str,
    *,
    max_questions: int = 2,
    strict: bool = True,
) -> tuple[str, ...]:
    """取显式会话中最近最多两个用户问题，不推断追问或独立问句。"""
    require_nonnegative_int(max_questions, "max_questions")
    require_text(question, "question")
    questions = [
        normalize_question_text(item)
        for item in user_questions(context_text, strict=strict)
    ]

    # 宿主提供当前轮之前的会话；相同文字也可能是再次提问，不能按文字删去。
    limit = min(max_questions, 2)
    return tuple(questions[-limit:]) if limit else ()
