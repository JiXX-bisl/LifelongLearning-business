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


def test_equal_width_aliases_keep_positional_compatibility_and_explicit_mapping():
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


@pytest.mark.parametrize("columns,rows,expected_columns,expected_rows,mapping", [
    (("state_name", "area"), [["texas", 266807.0]], ("area",), [[266807]], (1,)),
    (("city_name", "population"), [["phoenix", 789704]], ("city_name",), [["phoenix"]], (0,)),
    (("state_name", "size"), [["texas", 266807.0]], ("area",), [[266807]], (1,)),
    ((" AREA ", "state_name"), [[266807, "texas"]], ("area", "state_name"),
     [[266807, "texas"]], (0, 1)),
    (("revenue", "company", "note"), [[200, "B", "x"], [100, "A", "y"]],
     ("company", "revenue"), [["A", 100], ["B", 200]], (1, 0)),
    (("amount", "company", "note"), [[200, "B", "x"], [100, "A", "y"]],
     ("company", "revenue"), [["A", 100], ["B", 200]], (1, 0)),
    (("x", "x", "note"), [[1, 2, "a"]], ("first", "second"), [[2, 1]], (1, 0)),
])
def test_required_projection_accepts_auxiliary_columns_and_preserves_mapping(
        columns, rows, expected_columns, expected_rows, mapping):
    result = compare_results(table(rows, columns), table(expected_rows, expected_columns))
    assert result.status == "PASS"
    assert result.version == "row_projection_v2"
    assert result.column_map == mapping
    assert result.to_dict()["column_map"] == mapping


def test_explicit_projection_can_select_subset_and_overrides_names():
    actual = table([[0, 266807, "texas"]], ("area", "size", "state"))
    expected = table([[266807]], ("area",))
    assert compare_results(actual, expected).status == "FAIL"
    result = compare_results(actual, expected, policy=ComparisonPolicy(column_map=(1,)))
    assert result.status == "PASS" and result.column_map == (1,)
    for mapping in ((), (3,), (0, 1)):
        result = compare_results(actual, expected, policy=ComparisonPolicy(column_map=mapping))
        assert result.status == "UNKNOWN" and result.reason == "invalid_column_mapping"


@pytest.mark.parametrize("columns,rows,expected_columns,expected_rows", [
    # Correct column marginals cannot compensate for wrong row associations.
    (("entity", "amount", "note"), [["A", 200, "x"], ["B", 100, "y"]],
     ("company", "revenue"), [["A", 100], ["B", 200]]),
    # The mapping cannot change per row to pick whichever amount is correct.
    (("company", "x", "y"), [["A", 100, 0], ["B", 0, 200]],
     ("company", "revenue"), [["A", 100], ["B", 200]]),
    # One actual column cannot satisfy two required columns, even if equal.
    (("x", "y", "z"), [[1, 0, 2]], ("a", "b"), [[1, 1]]),
    # A same-name column stays bound despite other columns containing the gold value.
    (("area", "population", "note"), [[0, 266807, "texas"]], ("area",), [[266807]]),
    (("area", "x", "y"), [[0, 266807, "texas"]], ("area", "state"), [[266807, "texas"]]),
    # Projected duplicates remain significant; auxiliary values do not deduplicate rows.
    (("x", "note"), [[1, "a"], [1, "b"]], ("value",), [[1], [2]]),
])
def test_projection_rejects_inclusion_without_correct_whole_rows(
        columns, rows, expected_columns, expected_rows):
    assert compare_results(table(rows, columns), table(expected_rows, expected_columns)).status == "FAIL"


def test_projection_preserves_order_duplicates_and_empty_result_rules():
    actual = table([[2, "B"], [1, "A"], [1, "C"]], ("x", "note"))
    expected = table([[1], [1], [2]], ("value",))
    assert compare_results(actual, expected).status == "PASS"
    assert compare_results(actual, expected, policy=ComparisonPolicy(ordered=True)).status == "FAIL"
    assert compare_results(actual, table([[1], [2]], ("value",))).reason == "row_count_mismatch"
    assert compare_results(table([], ("x", "note")), table([], ("value",))).status == "PASS"
    assert compare_results(actual, table([], ("value",))).status == "FAIL"


def test_alias_projection_uses_numeric_policy_during_alignment():
    actual = table([[1, "x"], [0, "y"]], ("amount", "note"))
    expected = table([[0], [2]], ("value",))
    assert compare_results(actual, expected).status == "FAIL"
    assert compare_results(actual, expected, policy=ComparisonPolicy(absolute_tolerance=1)).status == "PASS"
    numeric_text = table([["1.00", "x"]], ("amount", "note"))
    assert compare_results(numeric_text, table([[1]], ("value",))).status == "FAIL"
    assert compare_results(numeric_text, table([[1]], ("value",)),
                           policy=ComparisonPolicy(numeric_strings=True)).status == "PASS"


def test_invalid_auxiliary_cells_are_ignored_but_required_and_partial_evidence_are_not():
    expected = table([[1]], ("value",))
    assert compare_results(table([[1, float("nan")]], ("value", "note")), expected).status == "PASS"
    assert compare_results(table([[1, float("nan")]], ("alias", "note")), expected).status == "PASS"
    assert compare_results(table([[float("nan"), 1]], ("value", "note")), expected).status == "UNKNOWN"
    # The invalid named column is visited first as an alias candidate, then as required.
    invalid = table([[1, float("nan"), "note"]], ("x", "value", "note"))
    assert compare_results(invalid, table([[1, 2]], ("alias", "value"))).status == "UNKNOWN"
    assert compare_results(table([[1, "x"]], ("value", "note"), is_complete=False), expected).status == "UNKNOWN"


def test_ambiguous_projection_search_limit_is_unknown(monkeypatch):
    monkeypatch.setattr("src.sql_learning.comparison._MAX_MAPPING_STEPS", 0)
    result = compare_results(table([[1, 2]], ("x", "y")), table([[1]], ("value",)))
    assert result.status == "UNKNOWN" and result.reason == "column_mapping_search_limit"
