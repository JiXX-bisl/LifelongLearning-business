from decimal import Decimal, localcontext

import pytest

from src.sql_learning.comparison import ComparisonPolicy, compare_results


def table(rows, columns=("a", "b"), **extras):
    return {"columns": list(columns), "rows": rows, **extras}


def test_row_association_is_required_and_row_order_is_optional():
    expected = table([["June", 2], ["July", 7]])
    assert compare_results(table([["July", 7], ["June", 2]]), expected).status == "PASS"
    assert compare_results(table([["June", 7], ["July", 2]]), expected).status == "FAIL"
    assert compare_results(table([["July", 7], ["June", 2]]), expected,
                           policy=ComparisonPolicy(ordered=True)).status == "FAIL"


def test_column_mapping_is_explicit_and_bijective():
    actual = table([[7, "July"]], ("amount", "month"))
    expected = table([["July", 7]], ("date", "total"))
    assert compare_results(actual, expected).status == "FAIL"
    assert compare_results(actual, expected, policy=ComparisonPolicy(column_map=(1, 0))).status == "PASS"
    assert compare_results(actual, expected, policy=ComparisonPolicy(column_map=(2, 0))).status == "UNKNOWN"
    with pytest.raises(ValueError):
        ComparisonPolicy(column_map=(0, 0))


def test_duplicates_extras_and_empty_results():
    expected = table([[1, 2], [1, 2]])
    assert compare_results(table([[1, 2], [2, 1]]), expected).status == "FAIL"
    assert compare_results(table([[1, 2]]), expected).status == "FAIL"
    assert compare_results(table([[1, 2], [1, 2], [1, 2]]), expected).status == "FAIL"
    assert compare_results(table([]), table([])).status == "PASS"
    assert compare_results(table([[1, 2]]), table([])).status == "FAIL"
    assert compare_results(table([], ("a",)), table([])).status == "FAIL"


def test_null_boolean_text_and_number_are_distinct():
    for left, right in [(None, 0), (False, 0), ("1", 1), ("2026-06", "2026-06-01")]:
        assert compare_results(table([[left]], ("x",)), table([[right]], ("x",))).status == "FAIL"
    assert compare_results(table([["1.00"]], ("x",)), table([[1]], ("x",)),
                           policy=ComparisonPolicy(numeric_strings=True)).status == "PASS"


def test_decimal_precision_and_tolerant_matching_not_greedy():
    with localcontext() as context:
        context.prec = 3
        actual = table([[Decimal("12345678901234567890.001")]], ("x",))
        expected = table([[Decimal("12345678901234567890.002")]], ("x",))
        assert compare_results(actual, expected).status == "FAIL"
        assert compare_results(actual, expected,
                               policy=ComparisonPolicy(absolute_tolerance=Decimal(".001"))).status == "PASS"
    # First actual can match either expected; second can only match first.
    assert compare_results(table([[1], [0]], ("x",)), table([[0], [2]], ("x",)),
                           policy=ComparisonPolicy(absolute_tolerance=1)).status == "PASS"
    assert compare_results(table([[1], [1]], ("x",)), table([[0], [3]], ("x",)),
                           policy=ComparisonPolicy(absolute_tolerance=1)).status == "FAIL"


@pytest.mark.parametrize("actual", [None, table([[1]]), table([[1, float("nan")]]),
                                    table([[1, 2]], is_complete=False)])
def test_invalid_or_partial_evidence_is_unknown(actual):
    assert compare_results(actual, table([[1, 2]])).status == "UNKNOWN"
