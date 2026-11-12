"""Dataset-neutral, row-preserving SQL execution-result comparison.

An explicit expected-index -> actual-index projection takes precedence. Otherwise
unique matching column names identify required columns; equal-width results keep
positional alias compatibility. With extra columns, unresolved aliases can use
one global injective value mapping, checked against complete rows. Extra columns
are ignored, never extra rows.
Dataset-specific transformations belong to the evaluator before this function.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Literal

COMPARISON_VERSION = "row_projection_v2"
_MAX_MAPPING_STEPS = 10_000


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
    # One mapping for every row, in expected-column order. Retained for audit.
    column_map: tuple[int, ...] | None = None

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


def _validate_shape(result):
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    columns, rows = result.get("columns"), result.get("rows")
    if (not isinstance(columns, (list, tuple)) or not columns
            or not all(isinstance(column, str) for column in columns)
            or not isinstance(rows, (list, tuple))):
        raise ValueError("invalid result shape")
    if "is_complete" in result and result["is_complete"] is not True:
        raise ValueError("incomplete result")
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(columns):
            raise ValueError("invalid row width")
    return columns, rows


def _column_mapping(actual_columns, expected_columns, policy):
    """Resolve explicit/schema correspondence; None requests global projection."""
    actual_width, expected_width = len(actual_columns), len(expected_columns)
    if policy.column_map is not None:
        mapping = policy.column_map
        if len(mapping) != expected_width or any(index >= actual_width for index in mapping):
            raise ValueError("invalid_column_mapping")
        return mapping

    actual_names = [name.strip().casefold() for name in actual_columns]
    expected_names = [name.strip().casefold() for name in expected_columns]
    if (all(expected_names) and len(set(expected_names)) == expected_width
            and all(actual_names.count(name) == 1 for name in expected_names)):
        return tuple(actual_names.index(name) for name in expected_names)
    if actual_width == expected_width:
        # Existing callers may use different aliases for the same column order.
        return tuple(range(expected_width))
    return None


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


def _rows_equal(actual_rows, expected_rows, policy):
    if len(actual_rows) != len(expected_rows):
        return False
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
    return equal


def _infer_projection(actual_columns, expected_columns, raw_actual, expected_rows, policy):
    """Bounded evaluator-side search; the same column map must satisfy every row.

    Unique shared names stay bound, so a wrong area cannot be replaced with an
    unrelated population value. Whole-row checks prune ambiguous alias choices.
    No reference values or alignment information are sent to the generator.
    """
    actual_names = [name.strip().casefold() for name in actual_columns]
    expected_names = [name.strip().casefold() for name in expected_columns]
    locked_columns = {index: actual_names.index(name)
                      for index, name in enumerate(expected_names)
                      if name and expected_names.count(name) == 1 and actual_names.count(name) == 1}
    # Validate known required columns before value-mismatch early returns, so
    # malformed required evidence stays UNKNOWN regardless of column order.
    normalized = {index: tuple(_scalar(row[index], policy.numeric_strings) for row in raw_actual)
                  for index in locked_columns.values()}
    candidates = []
    for expected_index in range(len(expected_names)):
        choices = ([locked_columns[expected_index]] if expected_index in locked_columns
                   else range(len(actual_columns)))
        matches = []
        expected_column = [(row[expected_index],) for row in expected_rows]
        for actual_index in choices:
            if actual_index not in normalized:
                try:
                    normalized[actual_index] = tuple(_scalar(row[actual_index], policy.numeric_strings)
                                                     for row in raw_actual)
                except (ValueError, TypeError, InvalidOperation, OverflowError):
                    normalized[actual_index] = None  # An invalid auxiliary column cannot satisfy a value.
            values = normalized[actual_index]
            if values is not None and _rows_equal([(value,) for value in values], expected_column, policy):
                matches.append(actual_index)
        if not matches:
            return None
        candidates.append(matches)

    order = sorted(range(len(expected_columns)), key=lambda index: (len(candidates[index]), index))
    chosen, used, steps, depth = {}, set(), 0, 0
    options = [iter(candidates[order[0]])]
    while options:
        try:
            actual_index = next(options[depth])
        except StopIteration:
            options.pop()
            if depth == 0:
                break
            depth -= 1
            used.remove(chosen.pop(order[depth]))
            continue
        steps += 1
        if steps > _MAX_MAPPING_STEPS:
            raise ValueError("column_mapping_search_limit")
        if actual_index in used:
            continue
        prefix = order[:depth + 1]
        projection = [chosen[index] for index in order[:depth]] + [actual_index]
        actual_prefix = [tuple(normalized[index][row] for index in projection)
                         for row in range(len(raw_actual))]
        expected_prefix = [tuple(row[index] for index in prefix) for row in expected_rows]
        if not _rows_equal(actual_prefix, expected_prefix, policy):
            continue
        chosen[order[depth]] = actual_index
        used.add(actual_index)
        depth += 1
        if depth == len(order):
            return tuple(chosen[index] for index in range(len(expected_columns)))
        options.append(iter(candidates[order[depth]]))
    return None


def compare_results(actual: Any, expected: Any, *,
                    policy: ComparisonPolicy | None = None) -> ComparisonResult:
    """Compare complete required rows, permitting additional output columns.

    Tolerant multiset comparison uses one-to-one matching, so a row cannot satisfy
    several expected occurrences. No string normalization or business coercion
    is enabled implicitly. Unresolved aliases with extra columns require one
    globally consistent projection; an exhausted search budget yields UNKNOWN.
    """
    policy = ComparisonPolicy() if policy is None else policy
    if not isinstance(policy, ComparisonPolicy):
        raise TypeError("policy must be ComparisonPolicy")
    try:
        actual_columns, raw_actual = _validate_shape(actual)
        expected_columns, raw_expected = _validate_shape(expected)
    except (ValueError, TypeError) as error:
        return ComparisonResult("UNKNOWN", str(error))
    if len(actual_columns) < len(expected_columns):
        return ComparisonResult("FAIL", "column_count_mismatch")
    try:
        mapping = _column_mapping(actual_columns, expected_columns, policy)
    except ValueError as error:
        return ComparisonResult("UNKNOWN", str(error))
    if len(raw_actual) != len(raw_expected):
        return ComparisonResult("FAIL", "row_count_mismatch", column_map=mapping)
    try:
        expected_rows = [tuple(_scalar(value, policy.numeric_strings) for value in row)
                         for row in raw_expected]
        if mapping is None:
            mapping = _infer_projection(actual_columns, expected_columns, raw_actual, expected_rows, policy)
            if mapping is None:
                return ComparisonResult("FAIL", "row_values_mismatch")
        # Normalize only required cells; supplementary values do not affect scoring.
        actual_rows = [tuple(_scalar(row[index], policy.numeric_strings) for index in mapping)
                       for row in raw_actual]
    except (ValueError, TypeError, InvalidOperation, OverflowError) as error:
        return ComparisonResult("UNKNOWN", str(error), column_map=mapping)
    equal = _rows_equal(actual_rows, expected_rows, policy)
    return ComparisonResult("PASS" if equal else "FAIL",
                            "rows_match" if equal else "row_values_mismatch",
                            column_map=mapping)
