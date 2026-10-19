"""Direct SQL result contract checks; execution is reserved for the user."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext

import pytest

from src.psu_memory.validation.comparison import compare_results, evaluate_result


def result(rows, columns=None):
    return {"columns": ["value"] if columns is None else columns, "rows": rows}


def operation(query_id="submitted", rows=None, **changes):
    value = {
        "kind": "sql", "query_id": query_id, "sql": "SELECT 1",
        "success": True, "is_complete": True,
        "columns": ["actual_name"], "rows": [[1]] if rows is None else rows,
    }
    value.update(changes)
    return value


def record(*operations, query_id="submitted", status="completed"):
    return {"status": status, "result_query_id": query_id, "operations": list(operations)}


def test_direct_comparison_ignores_names_and_row_order_but_preserves_duplicates():
    actual = result([[2], [1], [1]], ["模型列名"])
    expected = result([["1.00"], [2], [Decimal("1")]], ["标准列名"])
    assert compare_results(actual, expected)
    assert not compare_results(result([[1], [2]]), expected)
    assert not compare_results(result([[1], [2], [2]]), expected)


def test_direct_comparison_uses_column_positions():
    actual = result([[1, 2]], ["second", "first"])
    assert compare_results(actual, result([[1, 2]], ["first", "second"]))
    assert not compare_results(actual, result([[2, 1]], ["first", "second"]))


def test_decimal_comparison_does_not_round_to_active_context():
    long_number = "1234567890123456789012345678901234567890.12345678901234567890"
    different = "1234567890123456789012345678901234567890.12345678901234567891"
    with localcontext() as context:
        context.prec = 3
        assert compare_results(result([[long_number]]), result([[Decimal(long_number)]]))
        assert not compare_results(result([[long_number]]), result([[different]]))
        assert compare_results(result([["1.2300"]]), result([["123e-2"]]))


def test_text_comparison_does_not_depend_on_decimal_error_traps():
    with localcontext() as context:
        context.traps[InvalidOperation] = False
        assert compare_results(result([["企业名称"]]), result([["企业名称"]]))
        assert not compare_results(result([["企业名称"]]), result([["另一家企业"]]))


@pytest.mark.parametrize("value", [0, "", "NULL", False])
def test_null_does_not_equal_zero_missing_or_text(value):
    assert compare_results(result([[None]]), result([[None]]))
    assert not compare_results(result([[None]]), result([[value]]))


def test_direct_comparison_does_not_transform_results():
    assert not compare_results(result([[1], [2]]), result([[3]]))
    assert not compare_results(result([[1], [2]]), result([[1]]))
    assert not compare_results(result([[1]]), result([[10000]]))
    assert not compare_results(result([[1, "元"]], ["n", "unit"]), result([[1]]))
    assert not compare_results(result([[False]]), result([[0]]))
    assert not compare_results(result([["2026-01-01"]]), result([["2026-01-01T00:00:00"]]))


@pytest.mark.parametrize("malformed", [
    None, {}, {"columns": [], "rows": []}, {"columns": ["a"]},
    {"columns": ["a"], "rows": None}, {"columns": ["a"], "rows": [[1, 2]]},
    {"columns": ["a", "b"], "rows": [[1]]}, {"columns": [None], "rows": [[1]]},
    {"columns": ["a"], "rows": [1]}, result([[{"nested": 1}]]),
])
def test_malformed_results_never_match(malformed):
    assert not compare_results(malformed, malformed)
    assert not compare_results(malformed, result([[1]]))


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), Decimal("NaN"), "Infinity"])
def test_nonfinite_numbers_are_rejected(nonfinite):
    assert not compare_results(result([[nonfinite]]), result([[nonfinite]]))


def test_empty_results_require_same_column_width():
    assert compare_results(result([], ["a"]), result([], ["b"]))
    assert not compare_results(result([], ["a"]), result([], ["b", "c"]))


def test_evaluation_only_scores_the_explicit_submission_and_does_not_mutate_input():
    first = operation("first", rows=[[1]])
    last = operation("last", rows=[[2]])
    submitted = record(first, last, query_id="first")
    submitted["answer"] = "回复正文不参与SQL评分。"
    before = deepcopy(submitted)
    evaluated = evaluate_result(submitted, result([[1]]))
    assert evaluated == {
        "verdict": "pass", "query_id": "first", "sql": "SELECT 1",
        "actual": {"columns": ["actual_name"], "rows": [[1]]},
    }
    assert submitted == before
    assert evaluate_result(record(first, last, query_id="last"), result([[1]]))["verdict"] == "fail"
    assert evaluate_result({"status": "completed", "operations": [first]}, result([[1]]))["verdict"] == "fail"


@pytest.mark.parametrize("query_id", [
    None, "", " submitted", "submitted ", "sub\nmitted", 1, "missing", "null",
])
def test_missing_or_invalid_submission_fails(query_id):
    evaluated = evaluate_result(record(operation(), query_id=query_id), result([[1]]))
    assert evaluated["verdict"] == "fail"
    assert evaluated["sql"] is None
    assert evaluated["actual"] is None
    assert set(evaluated) == {"verdict", "query_id", "sql", "actual"}


def test_duplicate_or_non_sql_query_identity_fails():
    for operations in (
        [operation(), operation()],
        [operation(kind="schema")],
        [operation(), operation(kind="schema")],
    ):
        evaluated = evaluate_result(record(*operations), result([[1]]))
        assert evaluated["verdict"] == "fail"
        assert evaluated["sql"] is None
        assert evaluated["actual"] is None


@pytest.mark.parametrize("changes", [
    {"success": False}, {"success": 1}, {"is_complete": False}, {"is_complete": None},
    {"columns": ["a"], "rows": [[1, 2]]},
])
def test_failed_incomplete_or_malformed_sql_result_fails(changes):
    evaluated = evaluate_result(record(operation(**changes)), result([[1]]))
    assert evaluated["verdict"] == "fail"
    assert evaluated["query_id"] == "submitted"
    assert evaluated["sql"] == "SELECT 1"


@pytest.mark.parametrize("status", ["failed", "limit_reached", "running", None])
def test_unfinished_query_never_passes_even_with_matching_sql(status):
    assert evaluate_result(record(operation(), status=status), result([[1]]))["verdict"] == "fail"


def test_no_sql_or_no_reference_result_fails():
    assert evaluate_result(record(), result([[1]]))["verdict"] == "fail"
    assert evaluate_result(record(operation()), None)["verdict"] == "fail"
    assert evaluate_result(None, result([[1]]))["verdict"] == "fail"
