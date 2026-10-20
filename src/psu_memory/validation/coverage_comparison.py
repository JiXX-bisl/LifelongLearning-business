"""Complete required results, plus independently verified supplementary fields.

No database, model, execution of submitted SQL, or reconstruction of missing
answers occurs here. Private evidence is frozen by prepare, never supplied by
the query model. The v2 comparator is deliberately left unchanged.
"""
from __future__ import annotations

from collections import Counter
from decimal import InvalidOperation
import re

from .comparison import (
    _AUXILIARY_FIELDS, _MONETARY_FIELDS, _auxiliary_matches, _column_key,
    _month_scalar, _rows, _scalar, _valid_auxiliary_context,
)
from .comparison_versions import COVERAGE_COMPARISON_VERSION as COMPARISON_VERSION


# Additional explicit business labels, not value-based or positional guesses.
_LABELS = {
    "月份标签": "stat_month", "收到条数": "submitted_record_count",
    "report_count": "submitted_record_count",
    "销项发票额": "invoice_sales_amount", "进项发票额": "invoice_purchase_amount",
    "社保缴款": "social_security_amount", "社保缴费金额": "social_security_amount",
    "参保人数": "social_security_count", "员工人数": "employee_count",
    "运维发票额": "utility_invoice_amount",
    "差旅餐饮住宿发票额": "travel_hospitality_invoice_amount",
    "是否缴税": "paid_flag", "是否付税": "paid_flag",
    "任职开始日": "start_date", "任职结束日": "end_date",
    "自然人id": "person_entity_id", "稳定自然人id": "person_entity_id",
    "任职事实id": "assignment_id", "年龄": "age",
    "户籍地区": "household_region", "户籍省份": "household_region",
    "户籍省级地区": "household_region",
    "是否现任": "is_current", "是否在任": "is_current",
    "是否当前在任": "is_current", "在任标志": "is_current",
    "事件发生日": "event_date", "事件类型": "event_type",
    "案件类别": "case_category", "处理结果": "handling_result",
    "状态生效时刻": "state_effective_at", "报送时刻": "reported_at",
    "入库时刻": "ingested_at", "事件描述": "description",
    "营业利润率": "operating_profit_margin_pct",
    "利润总额利润率": "operating_profit_margin_pct",
    "profit_margin_pct": "operating_profit_margin_pct",
    "operating_profit_margin_pct": "operating_profit_margin_pct",
    "total_profit_margin_pct": "operating_profit_margin_pct",
    "net_profit_margin_pct": "net_profit_margin_pct",
    "净利率": "net_profit_margin_pct", "净利润率": "net_profit_margin_pct",
    "净利润利润率": "net_profit_margin_pct",
}
_MONEY_LABELS = {
    "invoice_sales_amount", "invoice_purchase_amount", "social_security_amount",
    "utility_invoice_amount", "travel_hospitality_invoice_amount",
}
_UNIT_SUFFIX = re.compile(r"(?:_(元|yuan)|\((元|yuan)\)|（(元|yuan)）)$")

# These ambiguous labels acquire a PERIOD meaning only when the final SQL
# expression proves SUM(metric) or SUM(profit)/SUM(revenue)*100.
_PERIOD_LABELS = {
    "total_operating_revenue_yuan": ("sum", "operating_revenue_yuan", "period_revenue_yuan"),
    "total_revenue_yuan": ("sum", "operating_revenue_yuan", "period_revenue_yuan"),
    "total_profit_yuan": ("sum", "total_profit", "period_total_profit_yuan"),
    "total_net_profit_yuan": ("sum", "net_profit_yuan", "period_net_profit_yuan"),
    "total_main_business_revenue_yuan": ("sum", "main_business_revenue", "period_main_business_revenue_yuan"),
    "profit_margin_pct": ("rate", "total_profit", "overall_profit_rate_pct"),
    "total_profit_margin_pct": ("rate", "total_profit", "overall_profit_rate_pct"),
    "net_profit_margin_pct": ("rate", "net_profit_yuan", "overall_net_profit_rate_pct"),
}


def _key(name):
    normalized = name.strip().casefold()
    if normalized in _LABELS:
        return _LABELS[normalized]
    for suffix in ("_百分数", "_百分比"):
        if normalized.endswith(suffix):
            key = _LABELS.get(normalized[:-len(suffix)])
            if key is not None and key.endswith("_pct"):
                return key
    suffix = _UNIT_SUFFIX.search(normalized)
    if suffix:
        base = normalized[:suffix.start()].strip()
        if _LABELS.get(base) in _MONEY_LABELS:
            return _LABELS[base]
    return _column_key(name)


def _period_keys(sql, columns, allowed_keys):
    """Prove scope of a few common aggregate aliases using the final projection.

    Unsupported SQL shapes simply cannot authorize an ambiguous period alias.
    SQL is parsed, never executed. Exact output values are checked separately.
    """
    if not isinstance(sql, str) or not sql.strip():
        return {}
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    def unwrap(node):
        while isinstance(node, (exp.Alias, exp.Paren, exp.Round)):
            node = node.this
        return node

    def literal(node, value):
        return isinstance(node, exp.Literal) and not node.is_string and node.this == str(value)

    def sum_measure(node):
        node = unwrap(node)
        if not isinstance(node, exp.Sum) or isinstance(node.this, exp.Distinct):
            return None
        if any(isinstance(child, (exp.Subquery, exp.Window, exp.AggFunc))
               for child in node.this.walk()):
            return None
        # An amount_unit CASE factor is allowed; values are independently
        # verified in yuan, so an incorrect factor still fails the value check.
        keys = {_key(column.name) for column in node.this.find_all(exp.Column)}
        keys.discard("amount_unit")
        return next(iter(keys)) if len(keys) == 1 else None

    def ratio_measure(node):
        node = unwrap(node)
        if isinstance(node, exp.Mul):
            if literal(node.this, 100):
                node = unwrap(node.expression)
            elif literal(node.expression, 100):
                node = unwrap(node.this)
            else:
                return None
        elif isinstance(node, exp.Div) and isinstance(unwrap(node.this), exp.Mul):
            numerator = unwrap(node.this)
            if literal(numerator.this, 100):
                node = exp.Div(this=numerator.expression, expression=node.expression)
            elif literal(numerator.expression, 100):
                node = exp.Div(this=numerator.this, expression=node.expression)
            else:
                return None
        else:
            return None
        if not isinstance(node, exp.Div):
            return None
        denominator = unwrap(node.expression)
        if isinstance(denominator, exp.Nullif) and literal(denominator.expression, 0):
            denominator = denominator.this
        if sum_measure(denominator) != "operating_revenue_yuan":
            return None
        return sum_measure(node.this)

    try:
        statements = sqlglot.parse(sql, read="mysql")
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return {}
        projections = statements[0].expressions
        if len(projections) != len(columns):
            return {}
        mapped = {}
        for index, (name, expression) in enumerate(zip(columns, projections)):
            label = name.strip().casefold()
            rule = _PERIOD_LABELS.get(label)
            if (rule is None or rule[2] not in allowed_keys
                    or expression.alias_or_name.casefold() != label):
                continue
            kind, measure, canonical = rule
            proven = sum_measure(expression) if kind == "sum" else ratio_measure(expression)
            if proven == measure:
                mapped[index] = canonical
        return mapped
    except (SqlglotError, ValueError, TypeError, AttributeError, RecursionError):
        return {}


def _value(key, value):
    return _month_scalar(value) if key == "stat_month" else _scalar(value)


def valid_coverage_evidence(expected, context):
    """Validate frozen core bindings before starting services or store writes."""
    if _rows(expected) is None or not isinstance(context, dict):
        return False
    keys = [_key(name) for name in expected["columns"]]
    evidence = context.get("coverage_evidence")
    if (None in keys or len(set(keys)) != len(keys)
            or not isinstance(evidence, dict)
            or evidence.get("version") != COMPARISON_VERSION
            or not isinstance(evidence.get("rows"), list)
            or len(evidence["rows"]) != len(expected["rows"])):
        return False
    try:
        return all(isinstance(fact, dict)
                   and all(key in fact and _value(key, fact[key]) == _value(key, value)
                           for key, value in zip(keys, row))
                   for row, fact in zip(expected["rows"], evidence["rows"]))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return False


def _assess(actual, expected, *, context=None, sql=None):
    if _rows(actual) is None or _rows(expected) is None:
        return "invalid_result"
    expected_keys = [_key(name) for name in expected["columns"]]
    if None in expected_keys or len(set(expected_keys)) != len(expected_keys):
        return "ambiguous_expected_columns"
    evidence = context.get("coverage_evidence") if isinstance(context, dict) else None
    if (not isinstance(evidence, dict) or evidence.get("version") != COMPARISON_VERSION
            or not isinstance(evidence.get("rows"), list)
            or len(evidence["rows"]) != len(expected["rows"])
            or any(not isinstance(row, dict) for row in evidence["rows"])):
        return "missing_private_evidence"
    facts = evidence["rows"]
    allowed = set(expected_keys)
    for row in facts:
        allowed.update(row)
    period_keys = _period_keys(sql, actual["columns"], allowed)
    actual_keys = [period_keys.get(i, _key(name)) for i, name in enumerate(actual["columns"])]
    if None in actual_keys or len(set(actual_keys)) != len(actual_keys):
        return "ambiguous_or_conflicting_columns"
    positions = {name: i for i, name in enumerate(actual_keys)}
    if any(key not in positions for key in expected_keys):
        return "missing_required_columns"
    try:
        # Reject corrupted/misaligned evidence before it can authorize any extra.
        if not valid_coverage_evidence(expected, context):
            return "invalid_private_evidence"
        required_actual = Counter(
            tuple(_value(key, row[positions[key]]) for key in expected_keys)
            for row in actual["rows"])
        required_expected = Counter(
            tuple(_value(key, value) for key, value in zip(expected_keys, row))
            for row in expected["rows"])
        if required_actual != required_expected:
            return "required_values_or_rows_differ"
        extras = set(actual_keys) - set(expected_keys)
        auxiliary = extras & _AUXILIARY_FIELDS
        identity_extras = auxiliary - {"amount_unit"}
        if (not _valid_auxiliary_context(identity_extras, expected_keys, context)
                or any(not _auxiliary_matches(row, actual_keys, identity_extras, context)
                       for row in actual["rows"])):
            return "supplementary_identity_differs"
        if "amount_unit" in auxiliary:
            if (context.get("amount_unit") != "元"
                    or not any(key in _MONETARY_FIELDS or key.endswith("_yuan") for key in actual_keys)
                    or any(row[positions["amount_unit"]] != "元" for row in actual["rows"])):
                return "supplementary_unit_differs"
        business_extras = extras - auxiliary
        if business_extras and (not facts or any(not business_extras.issubset(fact) for fact in facts)):
            return "unverified_supplementary_columns"
        # Compare the entire submitted row against independently frozen facts.
        # This binds extra fields to the same company/month/assignment as core
        # values, preserving multiplicity without dropping rows or calculating
        # missing outputs for the model.
        checked_keys = [key for key in actual_keys if key not in auxiliary]
        actual_rows = Counter(tuple(_value(key, row[positions[key]]) for key in checked_keys)
                              for row in actual["rows"])
        verified_rows = Counter(tuple(_value(key, fact[key]) for key in checked_keys) for fact in facts)
        if actual_rows != verified_rows:
            return "supplementary_values_differ"
    except (InvalidOperation, ValueError, TypeError, OverflowError, KeyError):
        return "invalid_value"
    return "complete_and_verified"


def compare_coverage_results(actual, expected, *, context=None, sql=None):
    return _assess(actual, expected, context=context, sql=sql) == "complete_and_verified"


def evaluate_coverage_result(record, expected, *, context=None):
    """Keep the same submission gates as v2, then apply v3 coverage checks."""
    result = {"verdict": "fail", "query_id": None, "sql": None, "actual": None,
              "reason": "invalid_submission"}
    if not isinstance(record, dict):
        return result
    query_id = record.get("result_query_id")
    if (not isinstance(query_id, str) or query_id == "null"
            or re.fullmatch(r"[A-Za-z0-9_-]+", query_id) is None):
        return result
    result["query_id"] = query_id
    operations = record.get("operations")
    if not isinstance(operations, list):
        return result
    matching = [operation for operation in operations
                if isinstance(operation, dict) and operation.get("query_id") == query_id]
    if len(matching) != 1:
        return result
    operation = matching[0]
    sql = operation.get("sql")
    if operation.get("kind") != "sql" or not isinstance(sql, str) or not sql.strip():
        return result
    result["sql"] = sql
    if isinstance(operation.get("columns"), list) and isinstance(operation.get("rows"), list):
        result["actual"] = {"columns": operation["columns"], "rows": operation["rows"]}
    if (record.get("status") != "completed" or operation.get("success") is not True
            or operation.get("is_complete") is not True):
        result["reason"] = "submission_not_completed"
        return result
    result["reason"] = _assess(result["actual"], expected, context=context, sql=sql)
    if result["reason"] == "complete_and_verified":
        result["verdict"] = "pass"
    return result
