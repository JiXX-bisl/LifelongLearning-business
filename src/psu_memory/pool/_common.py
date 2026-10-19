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
    """难度评估严格校验；分类跳过坏消息，并保留原始空白用于命中证据。"""
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

MATCH_TEXT_RULE_VERSION = "question_context_v1"

_CONTEXT_EXPLICIT = re.compile(
    r"(?:同样|相同|一样)(?:的)?(?:条件|方法|方式|要求)"
    r"|照着?(?:刚才|上次)"
    r"|沿用(?:刚才|上次|上述)"
    r"|继续(?:刚才|上次|上述)"
    r"|同上"
)

_CONTEXT_REFERENCE_START = re.compile(
    r"^(?:请帮我|帮我|麻烦|请)?\s*"
    r"(?:(?:再|继续|那就|那么)\s*)?"
    r"(?:(?:查询|查看|看看|看一下|我想看|想看|统计|比较|对比"
    r"|列出|列一下|给出|计算|汇总|筛选|找出|把|将|对于|对)\s*)*"
    r"(?:(?:这些|那些|这几|那几|这家|那家|上述|该)"
    r"(?:的|家)?(?:企业|公司|条件|结果|记录|数据|地区|月份)"
    r"|它们|他们)"
)

_CONTEXT_CHANGE_START = re.compile(
    r"^(?:那就)?(?:换成|改成|换为|改为"
    r"|再看|再查|继续查|继续看"
    r"|只看|仅看|只保留|仅保留)"
)

_CONTEXT_FRAGMENT = re.compile(
    r"^(?:那(?:么|就)?[^。！？?\n]{0,40}"
    r"|去年同期|去年|今年|前年|本月|上月|上个月|\d{4}年)"
    r"(?:的)?(?:呢|怎么样|如何)[？?。！!]*$"
)


def normalize_question_text(text: str) -> str:
    """统一 Unicode 与空白；不改写问题含义。"""
    require_text(text, "question")
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFC", text),
    ).strip()


def needs_user_context(question: str) -> bool:
    """规则识别明确的追问、外部指代和条件调整。"""
    text = normalize_question_text(question)
    return any(
        pattern.search(text)
        for pattern in (
            _CONTEXT_EXPLICIT,
            _CONTEXT_REFERENCE_START,
            _CONTEXT_CHANGE_START,
            _CONTEXT_FRAGMENT,
        )
    )


def necessary_user_questions(
    question: str,
    context_text: str,
    *,
    max_questions: int = 2,
    strict: bool = True,
) -> tuple[str, ...]:
    """从最近用户问题回溯，遇到独立问题或达到上限即停止。"""
    require_nonnegative_int(max_questions, "max_questions")
    current = normalize_question_text(question)

    if max_questions == 0 or not needs_user_context(current):
        return ()

    # 复用已有消息解析，不改变 difficulty 使用的 user_questions()。
    questions = [
        normalize_question_text(item)
        for item in user_questions(context_text, strict=strict)
    ]

    # 调用方误将当前问题放在消息末尾时，避免重复加入。
    while questions and questions[-1] == current:
        questions.pop()

    selected = []
    for previous in reversed(questions):
        selected.append(previous)
        if (
            not needs_user_context(previous)
            or len(selected) >= max_questions
        ):
            break

    return tuple(reversed(selected))
