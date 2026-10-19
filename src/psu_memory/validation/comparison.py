"""Compare one explicitly submitted SQL result without semantic transformations."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re
from typing import Any


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
    """Compare column positions and unordered rows, preserving every duplicate."""
    actual_rows, expected_rows = _rows(actual), _rows(expected)
    return (actual_rows is not None and expected_rows is not None
            and actual_rows == expected_rows)


def evaluate_result(record: Any, expected: Any) -> dict[str, Any]:
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
    if compare_results(result["actual"], expected):
        result["verdict"] = "pass"
    return result
