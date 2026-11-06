"""Expected-value inclusion and the completed-submission protocol."""
from copy import deepcopy
from decimal import Decimal, localcontext

import pytest

from src.psu_memory.validation.comparison import (
    COMPARISON_VERSION, assess_result, compare_results,
    compare_submission_results, evaluate_result,
)


def result(rows, columns=None):
    return {"columns": ["value"] if columns is None else columns, "rows": rows}


def operation(query_id="submitted", rows=None, **changes):
    value = {
        "kind": "sql", "query_id": query_id, "sql": "SELECT 1",
        "success": True, "is_complete": True,
        "columns": ["arbitrary_name"], "rows": [[1]] if rows is None else rows,
    }
    value.update(changes)
    return value


def record(*operations, query_id="submitted", status="completed"):
    return {"status": status, "result_query_id": query_id, "operations": list(operations)}


def test_only_expected_subset_version_is_active():
    assert COMPARISON_VERSION == "expected_subset_v2"


def test_column_names_positions_and_rows_do_not_gate_inclusion():
    expected = result([[1, 2], [3, 4]], ["first", "second"])
    for actual in (
        result([[4, 1], [2, 3]], ["unrelated", "unrelated"]),
        result([[4], [1], [3], [2]]),
        result([[3, 1, 4, 2]], ["a", "b", "c", "d"]),
    ):
        assert compare_submission_results(actual, expected)
    assert not compare_results(result([[4, 1], [2, 3]], ["a", "b"]), expected)


@pytest.mark.parametrize("extra", [
    999999, -123, "incorrect company", "wrong unit", "unrecognized label", None,
    float("nan"), float("inf"), Decimal("NaN"), "Infinity", {"unknown": 1},
])
def test_any_extra_value_is_ignored_even_if_wrong_or_not_a_supported_scalar(extra):
    actual = result([[12, extra]], ["anything", "anything"])
    assert compare_submission_results(actual, result([[12]], ["net_profit_yuan"]))


def test_extra_duplicate_rows_and_business_columns_do_not_fail():
    expected = result([[12]])
    actual = result([[12, 999], [12, 999], [88, 77]], ["a", "b"])
    assert compare_submission_results(actual, expected)


def test_required_occurrences_each_need_a_separate_actual_cell():
    expected = result([[0, 0, 2]], ["a", "b", "c"])
    assert not compare_submission_results(result([[0, 2]], ["a", "b"]), expected)
    assert compare_submission_results(result([[0], [2], [0], [0]]), expected)
    evaluated = assess_result(result([[0], [2], [0], [0]]), expected)
    cells = evaluated["alignment"]["cells"]
    assert len(cells) == 3
    assert len({tuple(pair["actual"]) for pair in cells}) == 3


@pytest.mark.parametrize("actual", [result([]), result([[11]]), result([[1], [2]])])
def test_missing_or_different_expected_value_fails_without_computing_it(actual):
    assert assess_result(actual, result([[12]])) == {
        "verdict": "fail", "reason": "missing_expected_values", "alignment": None,
    }


def test_expected_empty_result_is_contained_in_any_valid_result():
    for actual in (result([]), result([[999]]), result([[None, "extra"]], ["a", "b"])):
        assert assess_result(actual, result([], ["x", "y"])) == {
            "verdict": "pass", "reason": "expected_values_included",
            "alignment": {"layout": "value_subset", "cells": []},
        }


@pytest.mark.parametrize("replacement", [0, "0.00", "", "NULL", False])
def test_null_is_required_as_null_and_does_not_equal_zero_or_boolean(replacement):
    assert not compare_submission_results(result([[replacement]]), result([[None]]))
    assert compare_submission_results(result([[replacement], [None]]), result([[None]]))


def test_decimal_equivalence_preserves_all_digits():
    exact = "123456789012345678901234567890.12345678901234567890"
    different = "123456789012345678901234567890.12345678901234567891"
    with localcontext() as context:
        context.prec = 3
        assert compare_submission_results(result([[exact]]), result([[Decimal(exact)]]))
        assert not compare_submission_results(result([[different]]), result([[Decimal(exact)]]))
        assert compare_submission_results(result([["1.2300"]]), result([["123e-2"]]))


@pytest.mark.parametrize("expected_column", [
    "stat_month", "month", "month_label", "stat_month_label", "月份", "统计月份", "业务月份",
])
@pytest.mark.parametrize(("actual_value", "expected_value"), [
    ("2026-06", "2026-06-01"), ("2026-06-01", "2026-06"),
])
def test_expected_month_dimension_allows_first_day_equivalence(
        expected_column, actual_value, expected_value):
    assert compare_submission_results(
        result([[actual_value]], ["completely unrelated alias"]),
        result([[expected_value]], [expected_column]))


def test_month_dimensions_do_not_reintroduce_row_bindings():
    expected = result([["2026-06-01", 1], ["2026-07-01", 2]], ["stat_month", "amount"])
    actual = result([[2, "2026-06"], [1, "2026-07"]], ["arbitrary", "arbitrary"])
    assert compare_submission_results(actual, expected)


@pytest.mark.parametrize("actual_value", [
    "2026-06-02", "2026-06-01T00:00:00", "2026-6", "2026-07",
])
def test_different_month_or_more_precise_date_is_not_the_required_month(actual_value):
    assert not compare_submission_results(
        result([[actual_value]]), result([["2026-06-01"]], ["stat_month"]))


@pytest.mark.parametrize("column", ["start_date", "end_date", "event_date", "custom_date"])
def test_non_month_dates_keep_their_full_day(column):
    assert not compare_submission_results(
        result([["2026-06"]]), result([["2026-06-01"]], [column]))


def test_exact_date_and_month_require_separate_cells_without_greedy_loss():
    expected = result([["2026-06-01", "2026-06-01"]], ["stat_month", "start_date"])
    assert compare_submission_results(result([["2026-06-01"], ["2026-06"]]), expected)
    assert not compare_submission_results(result([["2026-06-01"]]), expected)


def test_observed_c4_l1_02_e1_arbitrary_aliases_pass():
    values = [["18179455.58", "-518256.88", "-2.850783"]]
    actual = result(values, ["period_operating_revenue_yuan", "a", "b"])
    expected = result(values, ["period_revenue_yuan", "period_total_profit_yuan",
                               "overall_profit_rate_pct"])
    assert compare_submission_results(actual, expected)


def test_observed_c2_l1_01_e2_month_aliases_pass():
    values = [[3, 2, 1, 0, 1, "-3723254.78", "-8970184.77", "48480590.17", "35787150.62"]]
    actual = result(values, [f"actual_{index}" for index in range(9)])
    expected = result(values, [f"expected_{index}" for index in range(9)])
    assert compare_submission_results(actual, expected)


def test_observed_c4_l3_02_e1_complex_label_and_long_layout_pass():
    actual = result([
        ["2026-06", "26.429327"], ["2026-07", "24.897740"],
        ["2026-06~2026-07平均", "25.663533"],
    ], ["unrecognized label", "unrecognized value"])
    expected = result([["26.429327", "24.897740", "25.663533"]],
                      ["june_profit_rate_pct", "july_profit_rate_pct", "mean_monthly_profit_rate_pct"])
    before = deepcopy((actual, expected))
    assert compare_submission_results(actual, expected)
    assert (actual, expected) == before


@pytest.mark.parametrize("malformed", [
    None, {}, {"columns": [], "rows": []}, {"columns": [None], "rows": [[1]]},
    {"columns": ["a"]}, {"columns": ["a"], "rows": None},
    {"columns": ["a"], "rows": [1]}, {"columns": ["a"], "rows": [[1, 2]]},
])
def test_invalid_table_structure_fails(malformed):
    assert assess_result(malformed, result([[1]]))["reason"] == "invalid_result"
    assert assess_result(result([[1]]), malformed)["reason"] == "invalid_result"


def test_evaluation_only_scores_explicit_submission_without_mutating_it():
    first, last = operation("first", [[1]]), operation("last", [[2]])
    submitted = record(first, last, query_id="first")
    submitted["answer"] = "回复正文不参与SQL结果评分。"
    before = deepcopy(submitted)
    evaluated = evaluate_result(submitted, result([[1]]))
    assert evaluated["verdict"] == "pass"
    assert evaluated["reason"] == "expected_values_included"
    assert evaluated["query_id"] == "first"
    assert evaluated["actual"] == result([[1]], ["arbitrary_name"])
    assert submitted == before
    assert evaluate_result(record(first, last, query_id="last"), result([[1]]))["verdict"] == "fail"


@pytest.mark.parametrize("sql", [
    "SELECT 1", "SELECT value FROM another_table", "WITH x AS (SELECT 1) SELECT * FROM x",
])
def test_sql_text_and_query_source_have_no_scoring_role(sql):
    assert evaluate_result(record(operation(sql=sql)), result([[1]]))["verdict"] == "pass"


@pytest.mark.parametrize("query_id", [
    None, "", " submitted", "submitted ", "sub\nmitted", 1, "missing", "null",
])
def test_missing_or_invalid_submission_identity_fails(query_id):
    evaluated = evaluate_result(record(operation(), query_id=query_id), result([[1]]))
    assert evaluated["verdict"] == "fail"
    assert evaluated["reason"] == "invalid_submission"
    assert evaluated["sql"] is None
    assert evaluated["actual"] is None


@pytest.mark.parametrize("operations", [
    [], [operation(), operation()], [operation(kind="schema")],
    [operation(), operation(kind="schema")], [operation(sql="")],
])
def test_absent_duplicate_or_non_sql_submission_fails(operations):
    assert evaluate_result(record(*operations), result([[1]]))["reason"] == "invalid_submission"


@pytest.mark.parametrize("changes", [
    {"success": False}, {"success": 1}, {"is_complete": False}, {"is_complete": None},
])
def test_failed_or_incomplete_query_does_not_provide_a_completed_result(changes):
    evaluated = evaluate_result(record(operation(**changes)), result([[1]]))
    assert evaluated["verdict"] == "fail"
    assert evaluated["reason"] == "submission_not_completed"


@pytest.mark.parametrize("status", ["failed", "limit_reached", "running", None])
def test_unfinished_task_is_not_scored_as_a_completed_submission(status):
    assert evaluate_result(record(operation(), status=status), result([[1]]))["reason"] == "submission_not_completed"


def test_invalid_record_or_result_fails():
    assert evaluate_result(None, result([[1]]))["reason"] == "invalid_submission"
    assert evaluate_result(record(operation()), None)["reason"] == "invalid_result"
    assert evaluate_result(record(operation(rows=[[1, 2]])), result([[1]]))["reason"] == "invalid_result"


@pytest.mark.parametrize(("question", "count"), [
    ("核查2026年3月的营收。", 1),
    ("核查2025年11月、12月两个月的营收。", 2),
    ("汇总2025年11月、12月和2026年3月的营收。", 3),
    ("汇总2025年3月和2026年3月的营收。", 2),
    ("汇总2026年3月至5月的营收。", 3),
    ("汇总2025年11月至2026年2月的营收。", 4),
    ("汇总2026年三月、四月和五月的营收。", 3),
    ("汇总2026年第二季度的营收。", 3),
    ("汇总2026-03、2026-04和2026-05的营收。", 3),
])
def test_given_calendar_count_is_not_a_required_answer(question, count):
    expected = result([[count, "20460585.78"]], ["business_month_count", "revenue"])
    actual = result([["20460585.78"]], ["任意别名"])
    before = deepcopy(expected)
    assessed = assess_result(actual, expected, question=question)
    assert assessed["verdict"] == "pass"
    assert assessed["alignment"]["omitted_known_inputs"] == [
        {"expected": [0, 0], "value": count, "reason": "month_count_given_in_question"}]
    assert expected == before
    assert not compare_submission_results(result([["20460585.79"]]), expected, question=question)


@pytest.mark.parametrize("question", [
    None, "汇总营收。", "汇总2026年13月的营收。", "汇总2026年十三月的营收。",
    "查询2026年3月15日的收入。", "查询2026-03-15的收入。",
    "2026年3月的营收，统计月份的数量是多少？",
    "2026年3月的营收及月份个数。", "2026年3月有几个月有数据？",
    "汇总2026年3月中有营业收入的月份。", "汇总2026年3月利润为正的月份。",
    "汇总截至2026年3月的营收。",
])
def test_unknown_or_requested_coverage_count_is_still_required(question):
    expected = result([[1, 12]], ["business_month_count", "revenue"])
    assert not compare_submission_results(result([[12]]), expected, question=question)


@pytest.mark.parametrize("column", [
    "reported_month_count", "known_tax_month_count", "observed_month_count",
    "loss_month_count", "person_count", "assignment_count",
])
def test_explicit_calendar_scope_does_not_remove_other_counts(column):
    expected = result([[3, 12]], [column, "amount"])
    assert not compare_submission_results(
        result([[12]]), expected, question="汇总2026年3月、4月、5月的数据。")


def test_fewer_observed_business_months_remain_a_query_result():
    expected = result([[2, 12]], ["business_month_count", "revenue"])
    question = "汇总2026年3月、4月、5月的营收。"
    assert not compare_submission_results(result([[12]]), expected, question=question)
    assert compare_submission_results(result([[2, 12]], ["a", "b"]), expected, question=question)


def test_only_explicit_month_reference_reuses_user_history():
    history = [{"role": "user", "content": "查询2026年3月、4月、5月的营收。"}]
    expected = result([[3, 12]], ["business_month_count", "revenue"])
    submitted = record(operation(rows=[[12]]))
    assert evaluate_result(submitted, expected, question="把这些月份汇总一下。",
                           history=history)["verdict"] == "pass"
    assert evaluate_result(submitted, expected, question="汇总实际经营期间的收入。",
                           history=history)["verdict"] == "fail"
    assert not compare_submission_results(result([[12]]), expected, question="汇总收入。",
                                          history=history)


def test_tax_status_equivalence_does_not_supply_a_missing_status_or_tax_value():
    expected = result([[None, "未知"], ["35082.92", "已知正数"]],
                      ["tax_amount_yuan", "value_status"])
    # Arbitrary aliases/layout are still accepted; both answers need their own cells.
    actual = result([[1, "35082.92"], [None, None]], ["a", "b"])
    assessed = assess_result(actual, expected)
    assert assessed["verdict"] == "pass"
    cells = assessed["alignment"]["cells"]
    assert sum(pair.get("match") == "tax_status_equivalent" for pair in cells) == 2
    assert len({tuple(pair["actual"]) for pair in cells}) == 4
    assert not compare_submission_results(result([[None], ["35082.92"], [1]]), expected)
    assert not compare_submission_results(result([[None], [None], ["35082.92"]]), expected)
    assert not compare_submission_results(result([[None], [None], [1]]), expected)


def test_tax_status_aliases_do_not_change_other_text_or_null_meanings():
    assert compare_submission_results(
        result([[0, 0]], ["a", "b"]),
        result([[0, "确认零"]], ["tax_amount_yuan", "value_status"]))
    assert not compare_submission_results(result([[None]]), result([["未知"]], ["remarks"]))
    assert not compare_submission_results(result([[1]]), result([["已知正数"]], ["value_status"]))
    assert not compare_submission_results(
        result([[None, 0]], ["a", "b"]),
        result([[None, "未知"]], ["tax_amount_yuan", "value_status"]))


def test_ordinary_values_are_matched_before_tax_status_alternatives():
    expected = result([["未知", None, "未知"]], ["value_status", "tax_amount_yuan", "remarks"])
    assert compare_submission_results(result([["未知", None, None]], ["a", "b", "c"]), expected)
    assert not compare_submission_results(result([["未知", None]], ["a", "b"]), expected)
