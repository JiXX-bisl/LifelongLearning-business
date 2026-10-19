"""按已核验 SQL 的最终输出动作归组；问题和上下文仅随经验保存。"""
from __future__ import annotations

import re
from dataclasses import replace
from collections.abc import Iterator

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from ._common import canonical_json, record_fields, require_nonnegative_int, sha256_text
from .pool_models import (
    GROUP_LABELS, GROUP_RULE_VERSION, GroupId, GroupOverride, GroupingDecision,
    GroupingInput, RuleMatch, require_group_id, require_text,
)

_PRIORITY: tuple[GroupId, ...] = ("rank", "compare", "aggregate", "detail")
_RANK_FUNCTIONS = {"rownumber", "rank", "denserank", "percentrank", "cumedist", "ntile"}
# SQL 中的技术键和时间字段用于版本选择/展示次序，不能据此认定业务排名。
_TECHNICAL_COLUMN = re.compile(
    r"(?:^|_)(?:id|ref|revision|version|date|time|timestamp|month|year|period|"
    r"created|updated|ingested|effective|reported)(?:_|$)|(?:_at|_ts)$", re.I,
)
_COMPARISON_AXIS = re.compile(
    r"(?:^|_)(?:date|month|year|period|enterprise|company|entity)(?:_|$)", re.I,
)

# 每个输出表达式携带所属 SELECT，避免把内层去重等辅助动作算作最终动作。
_Output = tuple[exp.Expression, exp.Select, dict[str, exp.Expression]]


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, (exp.Alias, exp.Paren, exp.Subquery)):
        node = node.this
    return node


def _environment(query: exp.Expression, inherited: dict) -> dict:
    result = dict(inherited)
    with_clause = query.args.get("with")
    if with_clause is not None:
        for cte in with_clause.expressions:
            result[cte.alias_or_name.lower()] = cte.this
    return result


def _sources(select: exp.Select, ctes: dict) -> dict[str, exp.Expression | None]:
    sources: dict[str, exp.Expression | None] = {}
    from_clause = select.args.get("from")
    nodes = ([from_clause.this] if from_clause is not None else [])
    nodes += [join.this for join in select.args.get("joins", [])]
    for node in nodes:
        if isinstance(node, exp.Subquery):
            sources[node.alias_or_name.lower()] = node.this
        elif isinstance(node, exp.Table):
            sources[node.alias_or_name.lower()] = ctes.get(node.name.lower())
    return sources


def _resolve(node: exp.Expression, select: exp.Select, ctes: dict,
             seen: frozenset = frozenset()) -> _Output:
    node = _unwrap(node)
    if not isinstance(node, exp.Column) or node.is_star:
        return node, select, ctes
    key = (id(select), node.sql())
    if key in seen:
        return node, select, ctes
    sources = _sources(select, ctes)
    source = sources.get(node.table.lower()) if node.table else (
        next(iter(sources.values())) if len(sources) == 1 else None
    )
    source = _unwrap(source) if source is not None else None
    if isinstance(source, exp.Select):
        environment = _environment(source, ctes)
        for projection in source.expressions:
            if projection.alias_or_name.lower() == node.name.lower():
                return _resolve(projection, source, environment, seen | {key})
        # SELECT * 的包装可以继续解析同名列，但不推断多来源列归属。
        if any(item.is_star for item in source.expressions):
            column = exp.column(node.name)
            return _resolve(column, source, environment, seen | {key})
    return node, select, ctes


def _outputs(query: exp.Expression, inherited: dict,
             seen: frozenset = frozenset()) -> Iterator[_Output]:
    query = _unwrap(query)
    if id(query) in seen:
        return
    seen = seen | {id(query)}
    ctes = _environment(query, inherited)
    if isinstance(query, exp.SetOperation):
        yield from _outputs(query.this, ctes, seen)
        yield from _outputs(query.expression, ctes, seen)
    elif isinstance(query, exp.Select):
        for projection in query.expressions:
            if projection.is_star:
                sources = _sources(query, ctes)
                table = projection.table.lower() if isinstance(projection, exp.Column) else ""
                values = [sources.get(table)] if table else list(sources.values())
                for source in values:
                    if source is not None:
                        yield from _outputs(source, ctes, seen)
            else:
                yield projection, query, ctes


def _nodes(output: _Output, seen: frozenset = frozenset(), *,
           stop_aggregate: bool = False) -> Iterator[_Output]:
    node, select, ctes = _resolve(*output)
    key = (id(node), id(select))
    if key in seen:
        return
    yield node, select, ctes
    seen = seen | {key}
    if stop_aggregate and isinstance(node, exp.AggFunc):
        return
    if isinstance(node, exp.Query):
        for item in _outputs(node, ctes):
            yield from _nodes(item, seen, stop_aggregate=stop_aggregate)
        return
    for child in node.iter_expressions():
        yield from _nodes((child, select, ctes), seen, stop_aggregate=stop_aggregate)


def _metric_columns(output: _Output) -> set[str]:
    return {
        node.name.lower() for node, _, _ in _nodes(output)
        if isinstance(node, exp.Column) and not node.is_star
        and not _TECHNICAL_COLUMN.search(node.name)
    }


def _business_order(order: exp.Expression | None, select: exp.Select, ctes: dict) -> bool:
    if order is None:
        return False
    for item in order.expressions:
        value = item.this
        # ORDER BY 可以引用当前输出别名或序号。
        if isinstance(value, exp.Literal) and value.is_int:
            index = int(value.this) - 1
            if 0 <= index < len(select.expressions):
                value = select.expressions[index]
        elif isinstance(value, exp.Column) and not value.table:
            value = next((p for p in select.expressions
                          if p.alias_or_name.lower() == value.name.lower()), value)
        if _metric_columns((value, select, ctes)):
            return True
        if any(isinstance(node, exp.Count) for node, _, _ in _nodes((value, select, ctes))):
            return True
        # 只有首要排序键决定选取动作，日期优先、金额次序仍是最新日期选取。
        return False
    return False


def _window_rank(output: _Output) -> bool:
    return any(
        isinstance(node, exp.Window)
        and (node.this.name if isinstance(node.this, exp.Anonymous) else node.this.key)
        .lower().replace("_", "") in _RANK_FUNCTIONS
        and _business_order(node.args.get("order"), owner, ctes)
        for node, owner, ctes in _nodes(output, stop_aggregate=True)
    )


def _rank_query(query: exp.Expression, inherited: dict,
                seen: frozenset = frozenset(), *, include_projected_rank: bool = True) -> bool:
    query = _unwrap(query)
    if id(query) in seen:
        return False
    seen = seen | {id(query)}
    ctes = _environment(query, inherited)
    if isinstance(query, exp.SetOperation):
        # UNION 的全局 ORDER/LIMIT 列解析存在歧义，仅识别各最终分支。
        return any(_rank_query(branch, ctes, seen, include_projected_rank=include_projected_rank)
                   for branch in (query.this, query.expression))
    if not isinstance(query, exp.Select):
        return False
    if query.args.get("limit") is not None and _business_order(query.args.get("order"), query, ctes):
        return True
    outputs = list(_outputs(query, ctes))
    if include_projected_rank and any(_window_rank(output) for output in outputs):
        return True
    # 新增聚合会把内部排名结果变成统计输出；透传聚合别名不属于新增聚合。
    adds_aggregate = any(
        isinstance(node, exp.AggFunc)
        for projection in query.expressions
        for node in projection.walk(prune=lambda item: isinstance(item, exp.Query))
    )
    if adds_aggregate:
        return False
    # 排名可以只用于最终 WHERE/QUALIFY 选取，而不作为输出列。
    for key in ("where", "qualify"):
        clause = query.args.get(key)
        if clause is not None and _window_rank((clause, query, ctes)):
            return True
    # 仅穿过不新增聚合的单来源输出包装，不能把被 COUNT 的内部 Top-N 算作最终排名。
    sources = _sources(query, ctes)
    if len(sources) == 1:
        source = next(iter(sources.values()))
        if source is not None:
            return _rank_query(source, ctes, seen, include_projected_rank=False)
    return False


def _comparison(outputs: list[_Output]) -> bool:
    # 相同指标从两个明确的表别名输出，或者用于差值/比值。
    qualified: dict[tuple[int, str], set[str]] = {}
    conditional: dict[tuple[int, str, tuple[str, ...]], set[str]] = {}
    for output in outputs:
        # 保留外层关系别名：同一 CTE 的两次引用可能代表两个时期或主体。
        expression, owner, ctes = output
        for column in expression.walk(prune=lambda item: isinstance(item, exp.Query)):
            if not isinstance(column, exp.Column):
                continue
            metrics = _metric_columns((column, owner, ctes))
            if column.table and metrics:
                for metric in metrics:
                    qualified.setdefault((id(owner), metric), set()).add(column.table.lower())
        for node, owner, ctes in _nodes(output, stop_aggregate=True):
            if (isinstance(node, exp.Window) and isinstance(node.this, (exp.Lag, exp.Lead))
                    and _metric_columns((node.this.this, owner, ctes))):
                return True
            if isinstance(node, exp.Column) and node.table and not _TECHNICAL_COLUMN.search(node.name):
                qualified.setdefault((id(owner), node.name.lower()), set()).add(node.table.lower())
            if not isinstance(node, exp.AggFunc):
                continue
            for case in node.find_all(exp.Case):
                for branch in case.args.get("ifs", []):
                    condition, value = branch.this, branch.args.get("true")
                    if value is None or not any(
                        _COMPARISON_AXIS.search(column.name) for column in condition.find_all(exp.Column)
                    ):
                        continue
                    metrics = tuple(sorted(_metric_columns((value, owner, ctes))))
                    if metrics:
                        conditional.setdefault((id(owner), node.key, metrics), set()).add(condition.sql())
    return any(len(tables) > 1 for tables in qualified.values()) or any(
        len(conditions) > 1 for conditions in conditional.values()
    )


def _sql_group(sql: str) -> tuple[GroupId, str]:
    if not sql.strip():
        return "detail", "缺少 SQL，使用明细组"
    try:
        statements = sqlglot.parse(sql, read="mysql")
    except SqlglotError:
        return "detail", "SQL 结构无法可靠解析，使用明细组"
    if len(statements) != 1 or statements[0] is None:
        return "detail", "无法确定唯一最终查询，使用明细组"
    query = _unwrap(statements[0])
    if not isinstance(query, (exp.Select, exp.SetOperation)):
        return "detail", "未识别到最终 SELECT，使用明细组"
    outputs = list(_outputs(query, {}))
    groups: set[GroupId] = {"detail"}
    if _rank_query(query, {}):
        groups.add("rank")
    if _comparison(outputs):
        groups.add("compare")
    for output in outputs:
        # 最终新增聚合遮蔽作为输入的内层极值；例如 SUM(已求出的各组 MAX)。
        for node, owner, ctes in _nodes(output, stop_aggregate=True):
            if isinstance(node, (exp.Min, exp.Max)) and _metric_columns((node, owner, ctes)):
                # 条件聚合常用于透视输出，不把它当成业务极值选取。
                if not any(isinstance(child, (exp.Case, exp.If)) for child in node.walk()):
                    groups.add("rank")
            if isinstance(node, exp.AggFunc) and not isinstance(node, (exp.Lag, exp.Lead)):
                groups.add("aggregate")
    if isinstance(query, exp.Select) and query.args.get("group") is not None:
        groups.add("aggregate")
    chosen = next(group for group in _PRIORITY if group in groups)
    return chosen, f"依据最终 SQL 输出结构选择“{GROUP_LABELS[chosen]}”"


def classify(request: GroupingInput, *, max_context_questions: int = 2) -> GroupingDecision:
    """调用方负责先核验 PASS；归组只读取实际提交 SQL，不读取标准答案。"""
    require_nonnegative_int(max_context_questions, "max_context_questions")
    group_id, reason = _sql_group(request.sql)
    fingerprint = sha256_text(canonical_json({
        "namespace": request.namespace,
        "data_version": request.data_version,
        "sql": request.sql.strip(),
        "rule_version": GROUP_RULE_VERSION,
    }))
    return GroupingDecision(
        **record_fields(request), group_id=group_id,
        input_sha256=fingerprint, reason=reason, text_source="sql",
        matches=(RuleMatch(rule_id=f"sql_{group_id}", group_id=group_id, matched_text=reason),),
    )


def apply_group_override(decision: GroupingDecision, group_id: GroupId, *,
                         reason: str, actor: str = "local_user") -> GroupingDecision:
    """兼容旧调用；自动维护入口不使用人工覆盖。"""
    require_group_id(group_id)
    require_text(reason, "reason")
    require_text(actor, "actor")
    reason, actor = reason.strip(), actor.strip()
    if decision.manual_history:
        last = decision.manual_history[-1]
        if (decision.method == "manual" and decision.group_id == group_id
                and last.reason == reason and last.actor == actor):
            return decision
    event = GroupOverride(previous_group_id=decision.group_id, group_id=group_id,
                          reason=reason, actor=actor)
    return replace(decision, group_id=group_id, method="manual", reason=f"人工更正：{reason}",
                   manual_history=decision.manual_history + (event,))
