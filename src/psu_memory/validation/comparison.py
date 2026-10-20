"""Deterministic SQL comparisons; reference verification keeps its strict contract."""
from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Any


COMPARISON_VERSION = "explicit_columns_aux_v2"

# Explicit names only: never choose a column because its values fit the reference.
# Scope-specific measures (quarter/month/combined totals) remain separate measures.
_COLUMN_NAMES = {
    "stat_month": ("month", "month_label", "stat_month_label", "月份", "统计月份", "业务月份"),
    "enterprise_id": ("企业编号", "公司编号", "企业id"),
    "enterprise_name": ("企业名称", "公司名称"),
    "amount_unit": ("unit", "单位", "金额单位"),
    "operating_revenue_yuan": ("operating_revenue", "revenue", "monthly_revenue_yuan",
                              "营业收入", "营收", "该月收入"),
    "net_profit_yuan": ("net_profit", "净利润", "该月净利润", "当月净利润"),
    "tax_amount_yuan": ("tax_amount", "税额", "实际缴税金额"),
    "outstanding_principal_yuan": ("outstanding_principal", "未偿本金"),
    "overdue_principal_yuan": ("overdue_principal", "逾期本金"),
    "contract_principal_yuan": ("contract_principal", "合同本金"),
    "period_revenue_yuan": ("period_revenue", "期间收入", "期间收入合计"),
    "period_total_profit_yuan": ("period_total_profit", "期间利润总额"),
    "overall_profit_rate_pct": ("overall_profit_rate", "期间利润率"),
    "mean_monthly_profit_rate_pct": ("mean_monthly_profit_rate", "月利润率平均值"),
    "quarter_net_profit_yuan": ("quarter_net_profit", "季度累计净利润"),
    "business_month_count": ("实际月份数", "实际经营月份数"),
    "observed_month_count": ("实际月份数_季度",),
    "reported_month_count": ("已报月数",),
    "known_tax_month_count": ("已知税额月数",),
    "known_tax_subtotal_yuan": ("已知税额小计",),
    "person_entity_id": ("稳定人员身份", "自然人编号"),
    "person_name": ("姓名", "人员姓名"),
    "assignment_id": ("任职身份", "任职编号"),
    "role": ("职务",),
    "start_date": ("开始日期",),
    "end_date": ("结束日期",),
    "person_count": ("自然人数",),
    "assignment_count": ("任职段数",),
    "loss_month_count": ("亏损月数",),
    "any_loss": ("是否出现亏损",),
    "all_three_loss": ("是否全部亏损",),
    "consecutive_two_loss": ("是否连续两个月亏损",),
    "value_status": ("金额状态", "税额状态"),
    "main_business_revenue": ("主营业务收入", "主营收入"),
    "total_profit": ("利润总额",),
    "total_revenue": ("收入合计", "累计收入"),
}
_ALIASES = {alias: canonical for canonical, aliases in _COLUMN_NAMES.items()
            for alias in (canonical, *aliases)}
# These known fields must not masquerade as a different measure by position.
_KNOWN_FIELDS = set(_ALIASES.values()) | set("""
all_three_loss april_overdue_event_amount_yuan august_overdue_principal_yuan
business_as_of_date business_object_ref combined_assignment_count combined_person_count
comparison_scope event_date first_enterprise_person_count handling_result ingested_at
named_assignment_count named_person_count related_case_count related_person_count
report_revision reported_at second_enterprise_person_count state_effective_at
submitted_amount_sum_yuan submitted_record_count amount balance_as_of_date
event_status event_type main_business_revenue total_profit monthly_record_id
report_status source_system source_record_ref employee_count invoice_purchase_amount
invoice_sales_amount social_security_amount social_security_count
travel_hospitality_invoice_amount utility_invoice_amount total_assets total_liabilities
person_id is_current period_start period_end
quantity term_months valuation operation_event_id negative_event_id age
as_of_date enterprise_type established_date industry main_business_description
registered_address registered_business_scope registered_capital registered_email
registered_phone registration_status registration_status_effective_date
source_updated_at synthetic_credit_code coverage_end coverage_id coverage_start
coverage_status last_synced_at table_name business_event_ref case_category description
related_person_id financing_stage has_wechat_account profile_record_id
wechat_content_summary household_region
""".split())
for _month in ("january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november", "december"):
    for _measure in ("net_profit_yuan", "revenue_yuan", "profit_rate_pct"):
        _KNOWN_FIELDS.add(f"{_month}_{_measure}")
_AUXILIARY_FIELDS = {"enterprise_id", "enterprise_name", "amount_unit"}
_MONETARY_FIELDS = {key for key in _KNOWN_FIELDS if key.endswith("_yuan")} | set("""
amount main_business_revenue total_profit total_revenue invoice_purchase_amount
invoice_sales_amount social_security_amount travel_hospitality_invoice_amount
utility_invoice_amount total_assets total_liabilities valuation registered_capital
""".split())
_FIELD_NAMES = {**{key: key for key in _KNOWN_FIELDS}, **_ALIASES}
_UNIT_BASE_NAMES = sorted(_FIELD_NAMES, key=len, reverse=True)
_UNIT_SUFFIX = re.compile(r"(?:_(.+)|\(([^()]*)\)|（([^（）]*)）)")


def _column_key(name: str) -> str | None:
    name = name.strip().casefold()
    if name in _FIELD_NAMES:
        return _FIELD_NAMES[name]
    # Strip only an explicit yuan label on a recognized monetary field. Match
    # the longest field name first, preserving month/quarter/period qualifiers.
    for base in _UNIT_BASE_NAMES:
        if not name.startswith(base):
            continue
        suffix = _UNIT_SUFFIX.fullmatch(name[len(base):].strip())
        if suffix is None:
            continue
        unit = next(value for value in suffix.groups() if value is not None).strip()
        key = _FIELD_NAMES[base]
        if key in _MONETARY_FIELDS and unit in ("元", "yuan"):
            return key
        # An incompatible unit or extra qualifier must not enter the legacy
        # unknown-alias positional fallback, even when numeric values coincide.
        return None
    return name


def _scalar(value: Any) -> tuple[str, Any]:
    if value is None:
        return ("null", None)
    if type(value) is bool:
        return ("boolean", value)
    if isinstance(value, str):
        if re.fullmatch(r"[+-]?(?:s?nan[0-9]*|inf(?:inity)?)", value, re.IGNORECASE):
            raise ValueError("Non-finite number")
        if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
            return ("text", value)
    if isinstance(value, (int, float, Decimal, str)):
        try:
            # Constructing Decimal and comparing/hashing it are exact; normalize()
            # would round long values according to the active decimal context.
            number = Decimal(str(value))
        except InvalidOperation:
            raise ValueError("Invalid number") from None
        if not number.is_finite():
            raise ValueError("Non-finite number")
        return ("number", number)
    raise ValueError("Unsupported SQL value")


def _rows(result: Any) -> tuple[int, Counter] | None:
    if not isinstance(result, dict):
        return None
    columns, rows = result.get("columns"), result.get("rows")
    if (not isinstance(columns, list) or not columns
            or not all(isinstance(column, str) for column in columns)
            or not isinstance(rows, list)):
        return None
    width = len(columns)
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in rows):
        return None
    try:
        return width, Counter(tuple(_scalar(value) for value in row) for row in rows)
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None


def compare_results(actual: Any, expected: Any) -> bool:
    """Strict reference/formation verification; no projection or month normalization."""
    actual_rows, expected_rows = _rows(actual), _rows(expected)
    return (actual_rows is not None and expected_rows is not None
            and actual_rows == expected_rows)


def _column_mapping(actual_keys, expected_keys):
    if (None in actual_keys or None in expected_keys
            or len(set(actual_keys)) != len(actual_keys)
            or len(set(expected_keys)) != len(expected_keys)):
        return None
    positions = {key: index for index, key in enumerate(actual_keys)}
    mapping = [positions.get(key) for key in expected_keys]
    if None in mapping and any(position is not None and position != index
                               for index, position in enumerate(mapping)):
        return None
    for index, position in enumerate(mapping):
        if position is not None:
            continue
        # Keep old, same-position unnamed aliases. This never authorizes a reorder,
        # projection, or replacement by a known but different business measure.
        if (len(actual_keys) != len(expected_keys) or index in mapping
                or actual_keys[index] in _KNOWN_FIELDS):
            return None
        mapping[index] = index
    return mapping


def _month_scalar(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}(?:-01)?", value):
        # Invalid months/years fail closed; no timestamp or other day is truncated.
        date.fromisoformat(value if len(value) == 10 else value + "-01")
        return ("month", value[:7])
    return _scalar(value)


def _valid_auxiliary_context(extras, expected_keys, context):
    if not extras:
        return True
    if not isinstance(context, dict):
        return False
    if "amount_unit" in extras:
        # The common public contract uses yuan. No unit conversion is performed.
        if (context.get("amount_unit") != "元"
                or not any(key.endswith("_yuan") for key in expected_keys)):
            return False
    if not ({"enterprise_id", "enterprise_name"} & extras):
        return True
    enterprises = context.get("enterprises")
    if (not isinstance(enterprises, list) or not enterprises
            or any(not isinstance(item, dict)
                   or any(not isinstance(item.get(key), str) or not item[key]
                          for key in ("enterprise_id", "enterprise_name"))
                   for item in enterprises)):
        return False
    return len({item["enterprise_id"] for item in enterprises}) == len(enterprises)


def _auxiliary_matches(row, keys, extras, context):
    if not extras:
        return True
    values = dict(zip(keys, row))
    if "amount_unit" in extras and values["amount_unit"] != "元":
        return False
    if not ({"enterprise_id", "enterprise_name"} & extras):
        return True
    enterprises = context["enterprises"]
    # For multi-company results a REQUIRED entity column must bind each row;
    # an extra label alone cannot establish who owns a row's business values.
    candidates = enterprises
    for key in ("enterprise_id", "enterprise_name"):
        if key in values and key not in extras:
            candidates = [item for item in candidates if item[key] == values[key]]
    if len(candidates) != 1:
        return False
    return all(values[key] == candidates[0][key]
               for key in ("enterprise_id", "enterprise_name") if key in values)


def compare_submission_results(actual: Any, expected: Any, *, context=None) -> bool:
    """Explicit column correspondence, bounded auxiliary fields, exact row multiset.

    context is private verifier input, never model output. Values are compared only
    AFTER the column correspondence is fixed, without aggregation or row removal.
    """
    if _rows(actual) is None or _rows(expected) is None:
        return False
    actual_keys = [_column_key(name) for name in actual["columns"]]
    expected_keys = [_column_key(name) for name in expected["columns"]]
    mapping = _column_mapping(actual_keys, expected_keys)
    if mapping is None:
        return False
    extras = {key for index, key in enumerate(actual_keys) if index not in mapping}
    if (not extras.issubset(_AUXILIARY_FIELDS)
            or not _valid_auxiliary_context(extras, expected_keys, context)):
        return False
    # Verify unused fields too, rather than silently accepting arbitrary metadata.
    if any(not _auxiliary_matches(row, actual_keys, extras, context) for row in actual["rows"]):
        return False
    try:
        def rows(values, indices):
            return Counter(tuple(
                (_month_scalar(row[position]) if key == "stat_month" else _scalar(row[position]))
                for key, position in zip(expected_keys, indices)
            ) for row in values)

        return rows(actual["rows"], mapping) == rows(expected["rows"], range(len(expected_keys)))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return False


def evaluate_result(record: Any, expected: Any, *, context=None) -> dict[str, Any]:
    """Only the model's explicit, unique current-task SQL submission is scored."""
    result = {"verdict": "fail", "query_id": None, "sql": None, "actual": None}
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
        result["actual"] = {
            "columns": operation["columns"],
            "rows": operation["rows"],
        }
    if (record.get("status") != "completed" or operation.get("success") is not True
            or operation.get("is_complete") is not True):
        return result
    if compare_submission_results(result["actual"], expected, context=context):
        result["verdict"] = "pass"
    return result
