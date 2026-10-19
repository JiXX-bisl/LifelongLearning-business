from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from src.lifelong_learning.tasks.enterprise_query_bench.dataset import (
    JSONScalar,
    RowOrder,
)


def normalize_mysql_value(value: Any) -> JSONScalar:
    """Convert connector-specific values to stable JSON-compatible values."""
    if value is None or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8")
    return str(value)


def normalize_mysql_rows(
    rows: list[tuple[Any, ...]],
) -> list[list[JSONScalar]]:
    return [
        [normalize_mysql_value(value) for value in row]
        for row in rows
    ]


def _canonical_value(value: JSONScalar) -> tuple[str, str]:
    if value is None:
        return ("null", "")

    # Treat numeric values with different harmless representations as equal,
    # e.g. Decimal('1.00'), 1 and "1.0000".
    if isinstance(value, (int, float)):
        return ("number", str(Decimal(str(value)).normalize()))

    if isinstance(value, str):
        try:
            number = Decimal(value)
        except InvalidOperation:
            return ("text", value)
        return ("number", str(number.normalize()))

    return ("text", str(value))


def _canonical_row(row: list[JSONScalar]) -> tuple[tuple[str, str], ...]:
    return tuple(_canonical_value(value) for value in row)


def compare_query_results(
    actual: list[list[JSONScalar]],
    expected: list[list[JSONScalar]],
    row_order: RowOrder,
) -> bool:
    """Apply the dataset ordered/unordered comparison policy."""
    actual_rows = [_canonical_row(row) for row in actual]
    expected_rows = [_canonical_row(row) for row in expected]

    if row_order == RowOrder.ORDERED:
        return actual_rows == expected_rows

    return Counter(actual_rows) == Counter(expected_rows)
