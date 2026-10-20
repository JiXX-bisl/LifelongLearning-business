"""Freeze private, independently sourced evidence for optional output columns.

This module reads approved local JSON only. It never executes reference SQL or
uses a submitted answer to select evidence. Unsupported or ambiguous scopes keep
their original expected fields and entity labels, without guessed supplements.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import json
from pathlib import Path
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from .comparison import _column_key


EVIDENCE_VERSION = "expected_coverage_v3"
_KNOWLEDGE_CUTOFF = "2026-09-15 23:59:59"
_MONTHLY_MONEY = (
    "operating_revenue", "main_business_revenue", "total_profit", "net_profit",
    "tax_amount", "invoice_purchase_amount", "invoice_sales_amount",
    "social_security_amount", "travel_hospitality_invoice_amount",
    "utility_invoice_amount",
)
_MONTHLY_FACTS = (
    "stat_month", "period_start", "period_end", "employee_count",
    "social_security_count", *_MONTHLY_MONEY,
)
_PERSON_FACTS = (
    "enterprise_id", "person_entity_id", "person_name", "assignment_id", "role",
    "start_date", "end_date", "is_current", "age", "household_region",
)
_EVENT_FACTS = (
    "enterprise_id", "business_event_ref", "business_object_ref", "event_date",
    "state_effective_at", "event_type", "handling_result", "related_person_id",
    "case_category", "description", "event_status", "balance_as_of_date",
    "financing_stage", "quantity", "term_months",
)
_PROVENANCE_FIELDS = (
    "report_revision", "report_status", "reported_at", "ingested_at",
    "source_system", "source_record_ref",
)
_PERIOD_KEYS = {
    "period_revenue_yuan": "operating_revenue_yuan",
    "period_total_profit_yuan": "total_profit",
    "period_net_profit_yuan": "net_profit_yuan",
    "period_main_business_revenue_yuan": "main_business_revenue",
    "period_tax_amount_yuan": "tax_amount_yuan",
    "period_invoice_sales_amount_yuan": "invoice_sales_amount",
    "period_invoice_purchase_amount_yuan": "invoice_purchase_amount",
}
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_MONTHLY_VALUE_KEYS = frozenset(_column_key(key) for key in _MONTHLY_MONEY)


@dataclass(frozen=True)
class _MonthlyEvidence:
    # Internal exact values never enter the serialized acceptable output fields.
    fields: dict
    exact_money: dict
    source_count: int = 1


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _equal(key, left, right):
    """Compare only a known field, preserving NULL and the business scope."""
    if left is None or right is None:
        return left is right
    if key == "stat_month":
        left, right = str(left), str(right)
        return left == right or (
            re.fullmatch(r"\d{4}-\d{2}(?:-01)?", left) is not None
            and re.fullmatch(r"\d{4}-\d{2}(?:-01)?", right) is not None
            and left[:7] == right[:7]
        )
    if key.endswith("_at"):
        return str(left).replace(" ", "T", 1) == str(right).replace(" ", "T", 1)
    if (not isinstance(left, bool) and not isinstance(right, bool)
            and _NUMBER.fullmatch(str(left)) and _NUMBER.fullmatch(str(right))):
        return Decimal(str(left)) == Decimal(str(right))
    return type(left) is type(right) and left == right


def _merge(core, extra):
    """A disagreement with approved truth invalidates the entire supplement."""
    if any(key in core and not _equal(key, core[key], value)
           for key, value in extra.items()):
        return dict(core)
    return {**extra, **core}


def _intersection(projections):
    if not projections:
        return {}
    return {key: value for key, value in projections[0].items()
            if all(key in item and _equal(key, value, item[key])
                   for item in projections[1:])}


def _timestamp(value):
    return value.replace(" ", "T", 1) if isinstance(value, str) else value


def _project(record, fields):
    return {_column_key(key): (_timestamp(record[key]) if key.endswith("_at")
                               else record[key])
            for key in fields if key in record}


def _exact_money(value, unit):
    if unit not in ("元", "万元"):
        raise ValueError("补充证据包含不支持的金额单位")
    if value is None:
        return None
    with localcontext() as context:
        context.prec = 50
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("补充证据包含非有限金额")
        if unit == "万元":
            number *= 10000
        return number


def _money(value, unit):
    number = _exact_money(value, unit)
    if number is None:
        return None
    with localcontext() as context:
        context.prec = 50
        return str(number.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _rate(numerator, denominator):
    if numerator is None or denominator is None or Decimal(denominator) == 0:
        return None
    with localcontext() as context:
        context.prec = 50
        return str((100 * Decimal(numerator) / Decimal(denominator)).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _latest(records, primary_key):
    """Apply the dataset's source-revision order before filtering withdrawals."""
    groups = defaultdict(list)
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("ingested_at"), str)
                or record["ingested_at"].replace("T", " ", 1) > _KNOWLEDGE_CUTOFF):
            continue
        keys = ("enterprise_id", "source_system", "source_record_ref", primary_key)
        if any(not record.get(key) for key in keys):
            raise ValueError("补充证据缺少来源身份字段")
        if type(record.get("report_revision")) is not int:
            raise ValueError("补充证据缺少有效修订号")
        groups[tuple(record[key] for key in keys[:3])].append(record)
    latest = []
    for records_in_group in groups.values():
        chosen = max(records_in_group, key=lambda row: (
            row["report_revision"], row["ingested_at"].replace("T", " ", 1),
            row[primary_key]))
        if chosen.get("report_status") == "正式":
            latest.append(chosen)
    return latest


def _monthly_projection(record):
    result = _project(record, _MONTHLY_FACTS)
    exact = {}
    for key in _MONTHLY_MONEY:
        if key not in record:
            raise ValueError("月度补充证据缺少金额字段：" + key)
        canonical = _column_key(key)
        exact[canonical] = _exact_money(record[key], record.get("amount_unit"))
        result[canonical] = _money(record[key], record.get("amount_unit"))
    result["amount_unit"] = "元"
    result["operating_profit_margin_pct"] = _rate(
        exact["total_profit"], exact["operating_revenue_yuan"])
    result["net_profit_margin_pct"] = _rate(
        exact["net_profit_yuan"], exact["operating_revenue_yuan"])
    tax = exact["tax_amount_yuan"]
    result["value_status"] = ("未知" if tax is None else "确认零" if Decimal(tax) == 0
                              else "已知正数" if Decimal(tax) > 0 else "已知负数")
    if tax is None or Decimal(tax) >= 0:
        result["paid_flag"] = None if tax is None else int(Decimal(tax) > 0)
    return _MonthlyEvidence(result, exact)


def _monthly_index(records):
    groups = defaultdict(list)
    for record in _latest(records, "monthly_record_id"):
        groups[(record["enterprise_id"], record["stat_month"][:7])].append(record)
    result = {}
    for key, rows in groups.items():
        facts = [_monthly_projection(row) for row in rows]
        # Different business facts for the same month have no source priority.
        # Metadata differences alone are harmless but cannot be arbitrarily picked.
        if any(fact.exact_money != facts[0].exact_money
               or set(fact.fields) != set(facts[0].fields) or any(
                   not _equal(name, value, fact.fields[name])
                   for name, value in facts[0].fields.items()) for fact in facts[1:]):
            continue
        metadata = _intersection([
            _project(row, (*_PROVENANCE_FIELDS, "monthly_record_id")) for row in rows])
        result[key] = _MonthlyEvidence(
            {**facts[0].fields, **metadata}, facts[0].exact_money, len(rows))
    return result


def _entity(core, context):
    candidates = context.get("enterprises", [])
    for key in ("enterprise_id", "enterprise_name"):
        if key in core:
            candidates = [item for item in candidates if item[key] == core[key]]
    return dict(candidates[0]) if len(candidates) == 1 else {}


def _month_literal(node):
    if not isinstance(node, exp.Literal) or not node.is_string:
        return None
    value = node.this
    if re.fullmatch(r"\d{4}-\d{2}-01", value) is None:
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        return None
    return value[:7]


def _months_between(start, end):
    first = int(start[:4]) * 12 + int(start[5:]) - 1
    last = int(end[:4]) * 12 + int(end[5:]) - 1
    if first > last or last - first > 120:
        return None
    return {f"{value // 12:04d}-{value % 12 + 1:02d}"
            for value in range(first, last + 1)}


def _monthly_scope(sql, enterprise_id):
    """Recognize only a direct source WHERE with explicit entity and months.

    This is a narrow AST reader, not a SQL evaluator. OR, joins at the source,
    computed dates, implicit time ranges and additional predicates fail closed.
    """
    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except (SqlglotError, TypeError):
        return None
    if (not isinstance(tree, exp.Select)
            or next(tree.find_all(exp.Join), None)
            or next(tree.find_all(exp.SetOperation), None)):
        return None
    sources = [table for table in tree.find_all(exp.Table)
               if table.name.lower() == "enterprise_operation_monthly"]
    if len(sources) != 1:
        return None
    select = sources[0].find_ancestor(exp.Select)
    if (select is None or select.args.get("joins")
            or select.args.get("where") is None):
        return None
    def conjuncts(node):
        if isinstance(node, exp.Paren):
            return conjuncts(node.this)
        if isinstance(node, exp.And):
            return conjuncts(node.this) + conjuncts(node.expression)
        return [node]

    # The source month filter is insufficient if a later CTE or final SELECT
    # narrows the rows again. Only the reference's revision/status filters are
    # understood downstream; every other restriction disables period evidence.
    for layer in tree.find_all(exp.Select):
        if any(layer.args.get(key) is not None for key in ("having", "limit", "offset")):
            return None
        where = layer.args.get("where")
        if layer is select or where is None:
            continue
        for predicate in conjuncts(where.this):
            if (not isinstance(predicate, exp.EQ)
                    or not isinstance(predicate.this, exp.Column)
                    or not isinstance(predicate.expression, exp.Literal)):
                return None
            name, value = predicate.this.name.lower(), predicate.expression
            rank_filter = (name in ("rn", "report_rank")
                           and not value.is_string and value.this == "1")
            status_filter = (name == "report_status"
                             and value.is_string and value.this == "正式")
            if not (rank_filter or status_filter):
                return None

    entities, months, cutoff_seen = None, None, False
    for predicate in conjuncts(select.args["where"].this):
        column = predicate.this
        if not isinstance(column, exp.Column):
            return None
        name = column.name.lower()
        if name == "enterprise_id":
            values = ([predicate.expression] if isinstance(predicate, exp.EQ)
                      else predicate.expressions if isinstance(predicate, exp.In) else [])
            if not values or any(not isinstance(value, exp.Literal)
                                 or not value.is_string for value in values):
                return None
            selected = {value.this for value in values}
            entities = selected if entities is None else entities & selected
        elif name == "ingested_at":
            value = predicate.expression
            if (not isinstance(predicate, exp.LTE) or not isinstance(value, exp.Literal)
                    or value.this != _KNOWLEDGE_CUTOFF):
                return None
            cutoff_seen = True
        elif name == "stat_month":
            if isinstance(predicate, exp.Between):
                start = _month_literal(predicate.args.get("low"))
                end = _month_literal(predicate.args.get("high"))
                selected = _months_between(start, end) if start and end else None
            else:
                values = ([predicate.expression] if isinstance(predicate, exp.EQ)
                          else predicate.expressions if isinstance(predicate, exp.In) else [])
                selected = {_month_literal(value) for value in values}
            if not selected or None in selected:
                return None
            months = selected if months is None else months & selected
        else:
            return None
    if entities != {enterprise_id} or not months or not cutoff_seen:
        return None
    return months


def _period_evidence(core, fact, entity, monthly):
    counted_revenue = set(core) == {"business_month_count", "operating_revenue_yuan"}
    if (not entity or (not counted_revenue and not any(key in core for key in (
            "period_revenue_yuan", "period_total_profit_yuan", "overall_profit_rate_pct")))):
        return {}
    enterprise_id = entity["enterprise_id"]
    months = _monthly_scope(fact.get("reference_sql"), enterprise_id)
    if months is None:
        return {}
    rows = [monthly.get((enterprise_id, month)) for month in sorted(months)]
    # Reference SUM preserves effective source rows, whereas monthly projection
    # can collapse identical business facts. Do not infer an aggregate from that
    # collapsed view, even if its core totals happen to coincide (e.g. all zero).
    if not rows or any(row is None or row.source_count != 1 for row in rows):
        return {}
    result, exact_totals = {}, {}
    # A complete period total is not inferred from partly unknown amounts.
    for target, source in _PERIOD_KEYS.items():
        values = [row.exact_money[source] for row in rows]
        if any(value is None for value in values):
            continue
        with localcontext() as context:
            context.prec = 50
            exact_totals[target] = sum(values, Decimal(0))
            result[target] = str(exact_totals[target].quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP))
    if "period_revenue_yuan" in result and "period_total_profit_yuan" in result:
        result["overall_profit_rate_pct"] = _rate(
            exact_totals["period_total_profit_yuan"], exact_totals["period_revenue_yuan"])
    if "period_revenue_yuan" in result and "period_net_profit_yuan" in result:
        result["overall_net_profit_rate_pct"] = _rate(
            exact_totals["period_net_profit_yuan"], exact_totals["period_revenue_yuan"])
    result["business_month_count"] = len(rows)
    if counted_revenue:
        if "period_revenue_yuan" not in result:
            return {}
        # This approved COUNT + SUM schema names the sum operating_revenue_yuan.
        # Include it as an independent overlap check against the unchanged core.
        result["operating_revenue_yuan"] = result["period_revenue_yuan"]
        if len(rows) == 1:
            # A one-month sum has the same grain as that month's detail. A
            # multi-month sum must retain period-qualified supplemental fields.
            result = {**rows[0].fields, **result}
    return result


def _bound_records(core, entity, records, *, person=False):
    if not entity:
        return {}
    if person:
        keys = ("person_entity_id", "assignment_id")
        if not all(key in core for key in keys):
            return {}
        fields = _PERSON_FACTS
    else:
        # Object identity is necessary; dates/statuses alone do not identify an event.
        keys = tuple(key for key in ("business_object_ref", "business_event_ref",
                                    "state_effective_at", "event_date")
                     if key in core)
        if not {"business_object_ref", "business_event_ref"} & set(keys):
            return {}
        fields = _EVENT_FACTS
    candidates = [row for row in records if row["enterprise_id"] == entity["enterprise_id"]
                  and all(key in row and _equal(key, core[key], row[key]) for key in keys)]
    projections = []
    for row in candidates:
        projection = _project(row, (*fields, *_PROVENANCE_FIELDS))
        if not person:
            for source, target in (("amount", "amount"),
                                   ("outstanding_principal", "outstanding_principal_yuan"),
                                   ("overdue_principal", "overdue_principal_yuan"),
                                   ("valuation", "valuation")):
                if source in row:
                    projection[target] = _money(row[source], row.get("amount_unit"))
            if "amount" in row:
                projection["amount_unit"] = "元"
        # No source may contradict a field from the original expected row.
        if any(key in core and not _equal(key, core[key], value)
               for key, value in projection.items()):
            return {}
        projections.append(projection)
    return _intersection(projections)


def build_coverage_contexts(cases, standards, ground_truth_path, baseline_dir, contexts):
    """Return a copy of contexts with one frozen field dictionary per expected row.

    Missing/malformed static tables fail during preparation, before any model run.
    Every original expected field/value is retained. Extra facts are conservative:
    monthly business projection conflicts, unsupported periods and unbound
    aggregate/event rows get no speculative business fields.
    """
    path = Path(baseline_dir) / "tables.json"
    if not path.is_file():
        raise ValueError("expected_coverage_v3缺少静态补充证据：tables.json")
    tables = _read(path)
    required = (
        "enterprise_basic", "enterprise_operation_monthly", "enterprise_person",
        "enterprise_negative_event", "enterprise_operation_event",
    )
    if (not isinstance(tables, dict)
            or any(not isinstance(tables.get(table), list)
                   or any(not isinstance(row, dict) for row in tables[table])
                   for table in required)):
        raise ValueError("expected_coverage_v3静态补充证据的表结构无效")
    facts = {fact.get("id", fact.get("question_id")): fact
             for fact in _read(ground_truth_path)["cases"]}
    try:
        monthly = _monthly_index(tables["enterprise_operation_monthly"])
        people = _latest(tables["enterprise_person"], "person_id")
        events = [
            *_latest(tables["enterprise_negative_event"], "negative_event_id"),
            *_latest(tables["enterprise_operation_event"], "operation_event_id"),
        ]
        result = deepcopy(contexts)
        for case in cases:
            case_id = case["id"]
            expected, context = standards[case_id], result[case_id]
            keys = [_column_key(name) for name in expected["columns"]]
            if None in keys or len(set(keys)) != len(keys):
                raise ValueError("核心标准列无法唯一映射：" + case_id)
            evidence_rows = []
            for row in expected["rows"]:
                core = dict(zip(keys, deepcopy(row)))
                entity = _entity(core, context)
                base_extra = dict(entity)
                if any(key.endswith("_yuan") or key in _MONTHLY_MONEY for key in core):
                    base_extra["amount_unit"] = "元"
                base = _merge(core, base_extra)
                extra = {}
                if (entity and "stat_month" in core
                        and _MONTHLY_VALUE_KEYS.intersection(core)):
                    month_fact = monthly.get(
                        (entity["enterprise_id"], str(core["stat_month"])[:7]))
                    if month_fact is not None:
                        extra.update(month_fact.fields)
                extra.update(_period_evidence(core, facts[case_id], entity, monthly))
                extra.update(_bound_records(core, entity, people, person=True))
                extra.update(_bound_records(core, entity, events))
                evidence_rows.append(_merge(base, extra))
            context["coverage_evidence"] = {
                "version": EVIDENCE_VERSION, "rows": evidence_rows,
            }
        return result
    except (InvalidOperation, OverflowError) as error:
        raise ValueError("静态补充证据包含无效数值") from error
