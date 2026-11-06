"""Score submissions by expected-value inclusion; verify reference files strictly."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Any


COMPARISON_VERSION = "expected_subset_v2"
_MONTH_COLUMNS = {
    "stat_month", "month", "month_label", "stat_month_label",
    "月份", "统计月份", "业务月份",
}
_CN_MONTHS = {word: index for index, word in enumerate(
    ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"), 1)}
_CALENDAR_TOKEN = re.compile(
    r"(?<![0-9一二三四五六七八九十])(?:(?P<year>[0-9]{2,4})年)?(?:"
    r"(?P<month>1[0-2]|0?[1-9]|十一|十二|十|[一二三四五六七八九])月"
    r"(?![0-9一二三四五六七八九十]+[日号])"
    r"|第?(?P<quarter>[一二三四1-4])季度)"
    r"|(?<![0-9])(?P<iso_year>[0-9]{4})-(?P<iso_month>0[1-9]|1[0-2])(?:-01)?(?![0-9-])")
_DYNAMIC_MONTH_SCOPE = re.compile(
    r"(?:多少|几|哪些)个?月|(?:月份|月)(?:的)?(?:数(?:量)?|个数)|"
    r"实际|有效|已知|有数据|有收入|有营收|有营业收入|有业务|经营期间|营业期间|"
    r"开展业务|报送|缺报|缺失|未知|非空|剔除|排除|"
    r"其中|筛选|满足|符合|为正|为负|大于|小于|不为零|不为0|"
    r"以来|之前|之后|截至|截止|以前|以后|至今|过去|历史|全年|年度")
_MONTH_REFERENCE = re.compile(r"(?:上述|这些|这几|前述|刚才|相同|同一).{0,8}(?:月|期间|时段)")
_TAX_STATUS_ALIASES = {"未知": None, "已知正数": 1, "确认零": 0}


def _calendar_month_count(text: str) -> int | None:
    """Count explicitly named calendar months, never observed database coverage."""
    matches = list(_CALENDAR_TOKEN.finditer(text))
    if not matches:
        return None
    months, year, previous = set(), 0, None
    for match in matches:
        raw_year = match["year"] or match["iso_year"]
        if raw_year is not None:
            year = int(raw_year)
            if year < 100:
                year += 2000
        raw = match["month"] or match["iso_month"] or match["quarter"]
        number = _CN_MONTHS[raw] if raw in _CN_MONTHS else int(raw)
        first = (number - 1) * 3 + 1 if match["quarter"] else number
        start = year * 12 + first - 1
        end = start + (2 if match["quarter"] else 0)
        if previous is not None:
            prior_match, prior_end = previous
            gap = text[prior_match.end():match.start()].strip()
            if re.fullmatch(r"(?:至|到|[-—–~～])", gap):
                if start < prior_end and raw_year is None:
                    year += 1
                    start, end = start + 12, end + 12
                if end < prior_end or end - prior_end > 1200:
                    return None
                months.update(range(prior_end, end + 1))
        months.update(range(start, end + 1))
        previous = (match, end)
    return len(months)


def _given_month_count(question: Any, history: Any) -> int | None:
    # Missing/ambiguous question context never relaxes a required count. In
    # particular, asking for coverage/counts or data-dependent filtering keeps it.
    if not isinstance(question, str) or _DYNAMIC_MONTH_SCOPE.search(question):
        return None
    count = _calendar_month_count(question)
    if count is not None or not _MONTH_REFERENCE.search(question):
        return count
    if isinstance(history, list):
        for message in reversed(history):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return _given_month_count(content, None)
    return None


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
            # Decimal construction and equality preserve every supplied digit,
            # independent of the active decimal context's rounding precision.
            number = Decimal(str(value))
        except InvalidOperation:
            raise ValueError("Invalid number") from None
        if not number.is_finite():
            raise ValueError("Non-finite number")
        return ("number", number)
    raise ValueError("Unsupported SQL value")


def _valid_shape(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    columns, rows = result.get("columns"), result.get("rows")
    return (isinstance(columns, list) and bool(columns)
            and all(isinstance(column, str) for column in columns)
            and isinstance(rows, list)
            and all(isinstance(row, (list, tuple)) and len(row) == len(columns)
                    for row in rows))


def _rows(result: Any) -> tuple[int, Counter] | None:
    if not _valid_shape(result):
        return None
    try:
        return len(result["columns"]), Counter(
            tuple(_scalar(value) for value in row) for row in result["rows"])
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return None


def compare_results(actual: Any, expected: Any) -> bool:
    """Verify reference/seed material; this strict check is not online grading."""
    actual_rows, expected_rows = _rows(actual), _rows(expected)
    return (actual_rows is not None and expected_rows is not None
            and actual_rows == expected_rows)


def _month_scalar(value: Any) -> tuple[str, Any]:
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}(?:-01)?", value):
        date.fromisoformat(value if len(value) == 10 else value + "-01")
        return ("month", value[:7])
    return _scalar(value)


def assess_result(actual: Any, expected: Any, *, question=None, history=None) -> dict[str, Any]:
    """Require necessary expected values, ignoring layout and all extra values.

    Each required occurrence consumes a distinct actual cell. Submitted column
    names, row associations, labels and supplementary values have no effect.
    Given calendar counts may be omitted; observed coverage counts stay required.
    Expected month dimensions and tax status values permit bounded equivalence.
    """
    failed = {"verdict": "fail", "reason": "invalid_result", "alignment": None}
    if not _valid_shape(actual) or not _valid_shape(expected):
        return failed
    try:
        columns = [column.strip().casefold() for column in expected["columns"]]
        given_count = _given_month_count(question, history)
        tax_result = bool({"tax_amount_yuan", "tax_amount"}.intersection(columns))
        required, ignored = [], []
        for row_number, row in enumerate(expected["rows"]):
            for column, value in enumerate(row):
                name, coordinate = columns[column], (row_number, column)
                key = _month_scalar(value) if name in _MONTH_COLUMNS else _scalar(value)
                if (name == "business_month_count" and given_count is not None
                        and key == _scalar(given_count)):
                    ignored.append({"expected": list(coordinate), "value": value,
                                    "reason": "month_count_given_in_question"})
                    continue
                alias = None
                if (tax_result and name == "value_status" and isinstance(value, str)
                        and value in _TAX_STATUS_ALIASES):
                    alias = _scalar(_TAX_STATUS_ALIASES[value])
                required.append((key, coordinate, alias))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return failed

    available = defaultdict(deque)
    for row_number, row in enumerate(actual["rows"]):
        for column, value in enumerate(row):
            try:
                key = _scalar(value)
            except (InvalidOperation, ValueError, TypeError, OverflowError):
                # Extras have no correctness gate, including non-finite or
                # unsupported values. They cannot satisfy a required value.
                continue
            available[key].append((row_number, column))

    pairs, deferred = [], []
    # Ordinary exact values consume first, then months, then status values with
    # alternatives. A status must not steal text required by an ordinary label.
    for value, coordinate, alias in sorted(
            required, key=lambda cell: 2 if cell[2] is not None else int(cell[0][0] == "month")):
        candidates = ([("text", value[1]), ("text", value[1] + "-01")]
                      if value[0] == "month" else [value])
        chosen = next((key for key in candidates if available[key]), None)
        if chosen is None:
            if alias is not None:
                deferred.append((alias, coordinate))
                continue
            return {"verdict": "fail", "reason": "missing_expected_values",
                    "alignment": None}
        pairs.append({"expected": list(coordinate),
                      "actual": list(available[chosen].popleft())})
    # Consume all exact requirements before aliases. A tax NULL must not also
    # serve as the separate unknown-status answer, nor 1 as two required values.
    for alias, coordinate in deferred:
        if not available[alias]:
            return {"verdict": "fail", "reason": "missing_expected_values", "alignment": None}
        pairs.append({"expected": list(coordinate),
                      "actual": list(available[alias].popleft()),
                      "match": "tax_status_equivalent"})
    alignment = {"layout": "value_subset",
                 "cells": sorted(pairs, key=lambda pair: pair["expected"])}
    if ignored:
        alignment["omitted_known_inputs"] = ignored
    return {
        "verdict": "pass", "reason": "expected_values_included",
        "alignment": alignment,
    }


def compare_submission_results(actual: Any, expected: Any, *, question=None, history=None) -> bool:
    """True when a valid submitted result contains all necessary expected values."""
    return assess_result(actual, expected, question=question, history=history)["verdict"] == "pass"


def evaluate_result(record: Any, expected: Any, *, question=None, history=None) -> dict[str, Any]:
    """Score the explicit completed SQL submission by result-value inclusion."""
    result = {"verdict": "fail", "query_id": None, "sql": None, "actual": None,
              "reason": "invalid_submission", "alignment": None}
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
    result.update(assess_result(result["actual"], expected, question=question, history=history))
    return result
