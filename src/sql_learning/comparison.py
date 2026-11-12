"""Dataset-neutral, row-preserving SQL execution-result comparison.

Aliases do not define correspondence: positions are compared unless an explicit
expected-index -> actual-index permutation is supplied. Duplicates always count.
Dataset-specific transformations belong to the evaluator before this function.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Literal

COMPARISON_VERSION = "row_multiset_v1"


@dataclass(frozen=True)
class ComparisonPolicy:
    ordered: bool = False
    column_map: tuple[int, ...] | None = None
    absolute_tolerance: Decimal | float | int = 0
    relative_tolerance: Decimal | float | int = 0
    numeric_strings: bool = False

    def __post_init__(self):
        if type(self.ordered) is not bool or type(self.numeric_strings) is not bool:
            raise ValueError("ordered and numeric_strings must be booleans")
        for name in ("absolute_tolerance", "relative_tolerance"):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.column_map is not None and (
            not isinstance(self.column_map, tuple)
            or any(type(value) is not int or value < 0 for value in self.column_map)
            or len(set(self.column_map)) != len(self.column_map)
        ):
            raise ValueError("column_map must be a tuple of distinct nonnegative indices")


@dataclass(frozen=True)
class ComparisonResult:
    status: Literal["PASS", "FAIL", "UNKNOWN"]
    reason: str
    version: str = COMPARISON_VERSION

    def to_dict(self):
        return asdict(self)


def _scalar(value: Any, numeric_strings: bool):
    if value is None:
        return "null", None
    if type(value) is bool:
        return "boolean", value
    if isinstance(value, str) and not numeric_strings:
        return "text", value
    if isinstance(value, (int, float, Decimal, str)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            if isinstance(value, str):
                return "text", value
            raise ValueError("invalid numeric value") from None
        if not number.is_finite():
            raise ValueError("nonfinite numeric value")
        return "number", number
    if isinstance(value, bytes):
        return "binary", value
    raise ValueError("unsupported SQL scalar")


def _normalize(result, policy):
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    columns, rows = result.get("columns"), result.get("rows")
    if (not isinstance(columns, (list, tuple)) or not columns
            or not all(isinstance(column, str) for column in columns)
            or not isinstance(rows, (list, tuple))):
        raise ValueError("invalid result shape")
    if "is_complete" in result and result["is_complete"] is not True:
        raise ValueError("incomplete result")
    output = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(columns):
            raise ValueError("invalid row width")
        output.append(tuple(_scalar(value, policy.numeric_strings) for value in row))
    return len(columns), output


def _equal_scalar(actual, expected, policy):
    if actual == expected:
        return True
    if actual[0] != "number" or expected[0] != "number":
        return False
    left, right = actual[1], expected[1]
    # Avoid ambient Decimal precision silently accepting distinct long numbers.
    if not policy.absolute_tolerance and not policy.relative_tolerance:
        return False
    with localcontext() as context:
        values = (left, right, policy.absolute_tolerance, policy.relative_tolerance)
        context.prec = max(28, sum(len(v.as_tuple().digits) + abs(v.as_tuple().exponent)
                                   for v in values) + 8)
        return abs(left - right) <= max(policy.absolute_tolerance,
                                       policy.relative_tolerance * abs(right))


def compare_results(actual: Any, expected: Any, *,
                    policy: ComparisonPolicy | None = None) -> ComparisonResult:
    """Compare full results; malformed/incomplete evidence yields UNKNOWN.

    Tolerant multiset comparison uses one-to-one matching, so a row cannot satisfy
    several expected occurrences. No string normalization or business coercion
    is enabled implicitly. Empty results still require the same column count.
    """
    policy = ComparisonPolicy() if policy is None else policy
    if not isinstance(policy, ComparisonPolicy):
        raise TypeError("policy must be ComparisonPolicy")
    try:
        actual_width, actual_rows = _normalize(actual, policy)
        expected_width, expected_rows = _normalize(expected, policy)
    except (ValueError, TypeError, InvalidOperation, OverflowError) as error:
        return ComparisonResult("UNKNOWN", str(error))
    if actual_width != expected_width:
        return ComparisonResult("FAIL", "column_count_mismatch")
    mapping = (tuple(range(expected_width)) if policy.column_map is None
               else policy.column_map)
    if len(mapping) != expected_width or set(mapping) != set(range(actual_width)):
        return ComparisonResult("UNKNOWN", "invalid_column_mapping")
    actual_rows = [tuple(row[index] for index in mapping) for row in actual_rows]
    if len(actual_rows) != len(expected_rows):
        return ComparisonResult("FAIL", "row_count_mismatch")
    if not policy.absolute_tolerance and not policy.relative_tolerance:
        equal = (actual_rows == expected_rows if policy.ordered else
                 Counter(actual_rows) == Counter(expected_rows))
    else:
        def row_equal(left, right):
            return all(_equal_scalar(a, e, policy) for a, e in zip(left, right))

        if policy.ordered:
            equal = all(row_equal(a, e) for a, e in zip(actual_rows, expected_rows))
        else:
            # Iterative augmenting paths avoid greedy ambiguity and recursion
            # limits for large duplicate/tolerance-connected result sets.
            edges = [[j for j, expected_row in enumerate(expected_rows)
                      if row_equal(actual_row, expected_row)] for actual_row in actual_rows]
            matched = {}
            equal = True
            for start in range(len(actual_rows)):
                pending, seen, predecessor = [start], {start}, {}
                terminal = None
                while pending and terminal is None:
                    left = pending.pop()
                    for right in edges[left]:
                        if right not in matched:
                            terminal = (left, right)
                            break
                        next_left = matched[right]
                        if next_left not in seen:
                            seen.add(next_left)
                            predecessor[next_left] = (left, right)
                            pending.append(next_left)
                if terminal is None:
                    equal = False
                    break
                left, right = terminal
                matched[right] = left
                while left in predecessor:
                    left, right = predecessor[left]
                    matched[right] = left
    return ComparisonResult("PASS" if equal else "FAIL",
                            "rows_match" if equal else "row_values_mismatch")
