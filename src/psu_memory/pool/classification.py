"""基于数据获取动作的规则分类"""

from __future__ import annotations

import re

from dataclasses import replace
from typing import Literal, cast


from ._common import (
    canonical_json,
    necessary_user_questions,
    needs_user_context,
    record_fields,
    require_nonnegative_int,
    sha256_text,
)
from .pool_models import (
    GROUP_LABELS,
    GROUP_RULE_VERSION,
    GroupId,
    GroupOverride,
    GroupingDecision,
    GroupingInput,
    RuleMatch,
    require_group_id,
    require_text,
)

# 复合动作的确定性优先顺序
_PRIORITY: tuple[GroupId, ...] = (
    "rank", "compare", "aggregate", "detail"
)


# 识别明确的排名、Top-K、极值选取
_RULES = (
    (
        "rank_output", 
        "rank",
        re.compile(
            r"排名|排行|名次"
            r"|\btop\s*\d+\b"
            r"|前\s*[\d一二两三四五六七八九十百]+\s*"
            r"(?:名|家|位|条|项)"
            r"|最高|最低|最大值|最小值"
            r"|(?:最近|最早|最晚)(?:的)?"
            r"[\d一二两三四五六七八九十]+\s*(?:笔|条|次)",
            re.IGNORECASE,
        ),
    ),
    (
        "compare_output",
        "compare",
        re.compile(
            r"同比|环比|趋势|变化|增长率|增幅|降幅"
            r"|差值|对比|比较"
            r"|连续.{0,12}(?:上升|下降)"           
        )
    ),
    (
        "aggregate_action",
        "aggregate",
        re.compile(
            r"统计|计数|汇总|合计|累计|总额|总数|总量"
            r"|总收入|总营收|平均值|均值|平均|求和"
            r"|有多少|多少(?:家|个|条|名|笔)"
            r"|几(?:家|个|条|名|笔)"
        )
    ),
    (
        "quantity_request",
        "aggregate",
        re.compile(
            r"(?:查询|查看|查一下|查|返回|给出|计算)"
            r".{0,24}?(?:数量|总数|总额|总量)"     
        )
    ),
    (
        "detail_action",
        "detail",
        re.compile(
            r"查询|查一下|查看|查找|找出|筛选|列出"
            r"|返回|名单|明细|详情|获取|检索|给出|提供"  
        )
    )
)

_RULES += (
    (
        "compare_colloquial",
        "compare",
        re.compile(r"走势"),
    ),
    (
        "quantity_colloquial",
        "aggregate",
        re.compile(
            r"(?:看看|看一下|我想看|想看|帮我看|反查)"
            r".{0,48}?(?:数量|总数|总额|总量|有多少|多少家|几家)"
            r"|数一数|数一下"
        ),
    ),
    (
        "detail_colloquial",
        "detail",
        re.compile(
            r"看看|看一下|我想看|想看"
            r"|列一下|列一列"
            r"|筛出来|筛出|挑出|反查"
            r"|给我(?:一下|看看)|一起给我"
        ),
    ),
)

# 语句边界
_CLAUSE_SPLIT = re.compile(
    r"[，,。；;！？!?\n]+"
    r"|但是|而是|但|只需|仅需|只要"
    r"|只(?=统计|查询|列出|列一下|返回|筛选|筛出"
    r"|比较|对比|汇总|排名|计算|看看|查看|反查)"
    r"|仅(?=统计|查询|列出|列一下|返回|筛选|筛出"
    r"|比较|对比|汇总|排名|计算|看看|查看|反查)"
)


# 否定要求作用于数据获取动作
# “未缴社保”“不缴社保”这样的对象条件不会因此被排除。
_NEGATED_ACTION = re.compile(
    r"(?:不要|不用|无需|无须|不必|不需要|不要求|不再|不)"
    r"(?:(?:先|再|额外|另外|进行|做|提供|给出|输出"
    r"|显示|展示|计算|知道|查看)\s*)*"
    r"(?:排名|排行|排序|统计|计数|汇总|求和|计算"
    r"|分析|比较|对比|查询|查找"
    r"|列出|列一下|列一列|返回"
    r"|筛选|筛出来|筛出|挑出|反查"
    r"|看看|看一下|想看|看"
    r"|展示|数量|有多少)"
)

_EXAMPLE_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?"
    r"(?:历史示例|参考示例|示例|样例|例如|举例)"
    r"(?:\s*[:：]|$)"
)

LiteralSource = Literal[
    "question", "data_advice", "context", "none"
]


def _positive_clauses(text: str) -> tuple[str, ...]:
    """取得用于规则匹配的正向语句片段"""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # 这里只处理明确标记的示例行。
    # 调用者仍应只传入当前问题建议，不能传整个系统提示词。
    lines = [
        line
        for line in text.splitlines()
        if not _EXAMPLE_LINE.match(line)
    ]

    clauses: list[str] = []
    
    for part in _CLAUSE_SPLIT.split("\n".join(lines)):
        clause = part.strip()
        if not clause:
            continue
        negative = _NEGATED_ACTION.search(clause)
        if negative:
            # 保留否定要求之前的有效请求
            clause = clause[:negative.start()].strip()
        if clause:
            clauses.append(clause)
    return tuple(clauses)


def _collect_matches(text: str) -> tuple[RuleMatch, ...]:
    clauses = _positive_clauses(text)
    matches: list[RuleMatch] = []

    for rule_id, group_id, pattern in _RULES:
        for clause in clauses:
            found = pattern.search(clause)
            if found:
                matches.append(
                    RuleMatch(
                        rule_id=rule_id,
                        group_id=cast(GroupId, group_id),
                        matched_text=found.group(0)
                    )
                )
                break
    return tuple(matches)


def _input_fingerprint(request: GroupingInput, max_context_questions: int) -> str:
    # 只归并算法相同的输入，不合并执行记录或内容版本。
    return sha256_text(canonical_json({
        "namespace": request.namespace,
        "data_version": request.data_version,
        "question": request.question.strip(),
        "context_text": request.context_text.strip(),
        "data_advice": request.data_advice.strip(),
        "rule_version": GROUP_RULE_VERSION,
        "max_context_questions": max_context_questions,
    }))


def classify(request: GroupingInput, *, max_context_questions: int = 2) -> GroupingDecision:
    require_nonnegative_int(
        max_context_questions,
        "max_context_questions",
    )
    fingerprint = _input_fingerprint(request=request, max_context_questions=max_context_questions)

    sources: list[tuple[str, str]] = []

    if request.data_advice.strip():
        sources.append(("data_advice", request.data_advice))
    
    sources.append(("question", request.question))

    if needs_user_context(request.question):
        previous = necessary_user_questions(request.question, request.context_text, 
                                            max_questions=max_context_questions, strict=False)
        if previous:
            sources.append(("context", "\n".join(previous)))
        
    for source_name, text in sources:
        matches = _collect_matches(text)
        if not matches:
            continue
        matches_groups = {item.group_id for item in matches}
        chosen = next(
            group_id 
            for group_id in _PRIORITY
            if group_id in matches_groups
        )

        return GroupingDecision(
            **record_fields(request),
            group_id=chosen,
            input_sha256=fingerprint,
            reason=(
                f"依据 {source_name} 命中动作规则，"
                f"按优先顺序选择“{GROUP_LABELS[chosen]}”"
            ),
            text_source=cast(
                LiteralSource,
                source_name,
            ),
            matches=matches,
        )

    return GroupingDecision(
        **record_fields(request),
        group_id=None,
        input_sha256=fingerprint,
        reason="未识别到明确的数据获取动作，保留为未归组候选",
    )

def apply_group_override(
    decision: GroupingDecision,
    group_id: GroupId,
    *,
    reason: str,
    actor: str = "local_user"
) -> GroupingDecision:
    """返回人工更新后的结果"""
    require_group_id(group_id)
    require_text(reason, "reason")
    require_text(actor, "actor")

    reason = reason.strip()
    actor = actor.strip()

    # 相同人工操作重放，不重复增加更正记录。
    if decision.manual_history:
        last = decision.manual_history[-1]
        if (
            decision.method == "manual"
            and decision.group_id == group_id
            and last.reason == reason
            and last.actor == actor
        ):
            return decision

    event = GroupOverride(
        previous_group_id=decision.group_id,
        group_id=group_id,
        reason=reason,
        actor=actor,
    )

    return replace(
        decision,
        group_id=group_id,
        method="manual",
        reason=f"人工更正：{reason}",
        manual_history=decision.manual_history + (event,),
    )
