"""Submission comparison contracts; execution is reserved for the user."""
from copy import deepcopy
from decimal import Decimal, localcontext

import pytest

from src.psu_memory.validation.comparison import (
    compare_results,
    compare_submission_results,
    evaluate_result,
)


SINGLE_ENTERPRISE = {
    "enterprises": [{"enterprise_id": "E1", "enterprise_name": "甲公司"}],
    "amount_unit": "元",
}
TWO_ENTERPRISES = {
    "enterprises": [
        {"enterprise_id": "E1", "enterprise_name": "甲公司"},
        {"enterprise_id": "E2", "enterprise_name": "乙公司"},
    ],
    "amount_unit": "元",
}


def result(columns, rows):
    return {"columns": columns, "rows": rows}


def test_explicit_column_reordering_does_not_change_reference_verification():
    actual = result(["net_profit_yuan", "stat_month"], [[12, "2026-08-01"]])
    expected = result(["stat_month", "net_profit_yuan"], [["2026-08-01", 12]])
    assert compare_submission_results(actual, expected)
    assert not compare_results(actual, expected)


@pytest.mark.parametrize(
    "month_name", ["stat_month", "month", "month_label", "stat_month_label", "月份"])
@pytest.mark.parametrize(
    "revenue_name",
    ["operating_revenue_yuan", "operating_revenue", "monthly_revenue_yuan",
     "revenue", "营业收入", "该月收入"],
)
def test_fixed_aliases_support_explicit_reordering(month_name, revenue_name):
    actual = result([revenue_name, month_name], [["12.00", "2026-08-01"]])
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    assert compare_submission_results(actual, expected)


@pytest.mark.parametrize(
    ("actual_name", "expected_name"),
    [("净利润", "net_profit_yuan"), ("该月净利润", "net_profit_yuan"),
     ("当月净利润", "net_profit_yuan"),
     ("税额", "tax_amount_yuan"), ("tax_amount", "tax_amount_yuan")],
)
def test_fixed_measure_aliases(actual_name, expected_name):
    assert compare_submission_results(result([actual_name], [[12]]),
                                      result([expected_name], [["12.00"]]))


@pytest.mark.parametrize("suffix", ["_元", "(元)", "（元）", "_yuan", " (元)", " （元）"])
@pytest.mark.parametrize(
    ("measure", "canonical"),
    [("营业收入", "operating_revenue_yuan"),
     ("operating_revenue", "operating_revenue_yuan"),
     ("net_profit_yuan", "net_profit_yuan"),
     ("当月净利润", "net_profit_yuan"),
     ("税额", "tax_amount_yuan"),
     ("未偿本金", "outstanding_principal_yuan"),
     ("季度累计净利润", "quarter_net_profit_yuan"),
     ("期间收入合计", "period_revenue_yuan")],
)
def test_explicit_yuan_suffixes_work_with_reordering_and_valid_helpers(
        measure, canonical, suffix):
    actual = result(
        ["企业名称", measure + suffix, "单位", "stat_month_label"],
        [["甲公司", "12.0000", "元", "2026-08"]],
    )
    expected = result(["stat_month", canonical], [["2026-08-01", "12.00"]])
    before = deepcopy((actual, expected))
    assert compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)
    assert (actual, expected) == before


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("营业收入_元", "operating_revenue_yuan"),
     ("营业收入 (元)", "operating_revenue_yuan"),
     ("当月净利润_元", "net_profit_yuan"),
     ("净利润_元", "net_profit_yuan"),
     ("净利润 （元）", "net_profit_yuan")],
)
def test_same_unit_alias_passes_with_or_without_allowed_enterprise_column(alias, canonical):
    expected = result(["stat_month", canonical], [["2026-08-01", "12.00"]])
    without_helper = result(["月份", alias], [["2026-08", "12.00"]])
    with_helper = result(
        ["企业名称", "月份", alias], [["甲公司", "2026-08", "12.00"]])
    assert compare_submission_results(without_helper, expected)
    assert compare_submission_results(with_helper, expected, context=SINGLE_ENTERPRISE)


@pytest.mark.parametrize(
    ("actual_columns", "measure", "month_count"),
    [
        pytest.param(
            ["企业名称", "月份", "当月净利润_元"], "net_profit_yuan", 3,
            id="fifo-C6-L1-S02-T1"),
        pytest.param(
            ["企业名称", "月份", "净利润_元"], "net_profit_yuan", 2,
            id="fifo-C6-L1-S02-T2"),
        pytest.param(
            ["企业名称", "月份", "净利润_元"], "net_profit_yuan", 4,
            id="fifo-C6-L1-S02-T3"),
        pytest.param(
            ["企业名称", "月份", "营业收入_元"], "operating_revenue_yuan", 1,
            id="adaptive-A4-L1-01"),
        pytest.param(
            ["enterprise_name", "月份", "净利润_元"], "net_profit_yuan", 3,
            id="adaptive-C6-L1-S02-T1"),
        pytest.param(
            ["企业名称", "企业编号", "月份", "净利润_元"], "net_profit_yuan", 4,
            id="adaptive-C6-L1-S02-T3"),
    ],
)
def test_six_observed_false_fail_layouts_use_self_contained_fixtures(
        actual_columns, measure, month_count):
    # Preserve the six observed column layouts, using synthetic values and entity
    # context so these regressions do not depend on local experiment artifacts.
    actual_rows = []
    expected_rows = []
    for month, value in [("2026-04", "-12.00"), ("2026-05", "-15.00"),
                         ("2026-06", "-18.00"), ("2026-07", "20.00")][:month_count]:
        fields = {
            "企业名称": "甲公司", "enterprise_name": "甲公司", "企业编号": "E1",
            "月份": month, "当月净利润_元": value, "净利润_元": value,
            "营业收入_元": value,
        }
        actual_rows.append([fields[column] for column in actual_columns])
        expected_rows.append([month + "-01", value])
    actual = result(actual_columns, actual_rows)
    expected = result(["stat_month", measure], expected_rows)
    assert compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


@pytest.mark.parametrize(
    ("actual_name", "expected_name"),
    [("季度累计净利润_元", "net_profit_yuan"),
     ("quarter_net_profit_yuan_元", "net_profit_yuan"),
     ("april_net_profit_yuan_元", "may_net_profit_yuan"),
     ("april_net_profit_yuan_元", "net_profit_yuan"),
     ("期间收入合计_元", "operating_revenue_yuan"),
     ("收入合计_元", "operating_revenue_yuan"),
     ("期间利润总额_元", "total_profit"),
     ("利润总额_元", "net_profit_yuan")],
)
def test_yuan_suffix_normalization_preserves_scope_and_measure(actual_name, expected_name):
    assert not compare_submission_results(
        result([actual_name], [[12]]), result([expected_name], [[12]]))


@pytest.mark.parametrize(
    ("actual_name", "expected_name"),
    [("营业收入_万元", "operating_revenue_yuan"),
     ("营业收入 (万元)", "operating_revenue_yuan"),
     ("营业收入（亿元）", "operating_revenue_yuan"),
     ("净利润(万元)", "net_profit_yuan"),
     ("净利润 （USD）", "net_profit_yuan"),
     ("net_profit_yuan_USD", "net_profit_yuan"),
     ("net_profit_USD", "net_profit_yuan"),
     ("税额_%", "tax_amount_yuan"),
     ("tax_amount_万元", "tax_amount_yuan"),
     ("person_count_元", "person_count"),
     ("自然人数(元)", "person_count"),
     ("overall_profit_rate_pct_元", "overall_profit_rate_pct"),
     ("stat_month_元", "stat_month"),
     ("net_profit_yuan_total", "net_profit_yuan")],
)
def test_incompatible_unit_or_qualifier_cannot_use_same_position_alias_fallback(
        actual_name, expected_name):
    expected = result([expected_name], [[12]])
    assert not compare_submission_results(result([actual_name], [[12]]), expected)
    assert not compare_submission_results(
        result(["enterprise_name", actual_name], [["甲公司", 12]]),
        expected, context=SINGLE_ENTERPRISE)


def test_yuan_suffix_does_not_convert_amounts():
    assert not compare_submission_results(
        result(["营业收入_元", "enterprise_name"], [["12.00", "甲公司"]]),
        result(["operating_revenue_yuan"], [["120000.00"]]),
        context=SINGLE_ENTERPRISE)


@pytest.mark.parametrize(
    "columns",
    [["净利润_元", "net_profit_yuan"],
     ["当月净利润_元", "净利润（元）"],
     ["营业收入_yuan", "revenue"],
     ["stat_month_label", "stat_month"]],
)
def test_suffix_and_alias_normalization_cannot_hide_duplicate_semantic_columns(columns):
    actual = result(columns, [[12, 12]])
    assert not compare_submission_results(actual, actual)


@pytest.mark.parametrize("defect", ["amount", "entity", "duplicate", "missing", "null", "range"])
def test_yuan_alias_fix_does_not_accept_incorrect_business_results(defect):
    actual = result(
        ["企业名称", "净利润_元", "月份"], [["甲公司", "12.1234567", "2026-08"]])
    expected = result(
        ["stat_month", "net_profit_yuan"], [["2026-08-01", "12.1234567"]])
    if defect == "amount":
        actual["rows"][0][1] = "12.1234568"
    elif defect == "entity":
        actual["rows"][0][0] = "乙公司"
    elif defect == "duplicate":
        actual["rows"].append(list(actual["rows"][0]))
    elif defect == "missing":
        actual["rows"] = []
    elif defect == "null":
        actual["rows"][0][1] = None
    elif defect == "range":
        actual["rows"].append(["甲公司", "15.00", "2026-07"])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


def test_arbitrary_identical_names_can_be_reordered_without_value_guessing():
    actual = result(["second_custom_measure", "first_custom_measure"], [[2, 1]])
    expected = result(["first_custom_measure", "second_custom_measure"], [[1, 2]])
    assert compare_submission_results(actual, expected)


def test_unknown_alias_keeps_same_position_legacy_behavior_only():
    expected = result(["stat_month", "net_profit_yuan"], [["2026-08-01", 12]])
    assert compare_submission_results(
        result(["stat_month", "custom_answer"], [["2026-08-01", 12]]), expected)
    assert not compare_submission_results(
        result(["custom_answer", "stat_month"], [[12, "2026-08-01"]]), expected)


def test_unknown_alias_cannot_fill_gap_when_other_columns_have_moved():
    expected = result(
        ["stat_month", "person_count", "net_profit_yuan"],
        [["2026-08-01", 2, 12]],
    )
    actual = result(["net_profit", "custom_count", "stat_month"],
                    [[12, 2, "2026-08-01"]])
    assert not compare_submission_results(actual, expected)


@pytest.mark.parametrize(
    ("actual_name", "expected_name"),
    [("tax_amount", "operating_revenue_yuan"),
     ("net_profit", "tax_amount_yuan"),
     ("person_count", "assignment_count"),
     ("quantity", "person_count"),
     ("valuation", "operating_revenue_yuan"),
     ("quarter_net_profit_yuan", "net_profit_yuan"),
     ("april_net_profit_yuan", "may_net_profit_yuan")],
)
def test_equal_values_do_not_override_known_conflicting_meanings(actual_name, expected_name):
    assert not compare_submission_results(
        result([actual_name], [[0]]), result([expected_name], [[0]]))


@pytest.mark.parametrize(
    "columns",
    [["stat_month", "month"], ["net_profit", "net_profit_yuan"],
     ["enterprise_name", "企业名称"], ["unit", "amount_unit"], ["value", "value"]],
)
def test_duplicate_semantic_columns_are_ambiguous_even_when_values_match(columns):
    actual = result(columns, [[0, 0]])
    assert not compare_submission_results(actual, actual, context=SINGLE_ENTERPRISE)


def test_all_three_auxiliary_fields_can_accompany_explicit_required_columns():
    actual = result(
        ["企业名称", "营业收入", "unit", "企业编号", "month"],
        [["甲公司", "12.00", "元", "E1", "2026-08-01"]],
    )
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    before = deepcopy((actual, expected, SINGLE_ENTERPRISE))
    assert compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)
    assert (actual, expected, SINGLE_ENTERPRISE) == before
    assert not compare_results(actual, expected)


def test_extra_column_does_not_allow_guessing_unknown_required_alias():
    actual = result(["month", "custom_answer", "enterprise_name"],
                    [["2026-08-01", 12, "甲公司"]])
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


@pytest.mark.parametrize(
    ("extra_name", "extra_value"),
    [("main_business_revenue", 12), ("total_profit", 12), ("net_profit", 12),
     ("report_status", "正式"), ("source_system", "企业报送系统"),
     ("value_status", "已知正数"), ("monthly_record_id", "MR-1")],
)
def test_additional_business_and_report_columns_remain_failures(extra_name, extra_value):
    actual = result(["operating_revenue_yuan", extra_name], [[12, extra_value]])
    expected = result(["operating_revenue_yuan"], [[12]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


@pytest.mark.parametrize("unit", ["万元", "人民币", None, "", "元 "])
def test_auxiliary_unit_must_match_yuan_without_conversion(unit):
    actual = result(["operating_revenue_yuan", "unit"], [[12, unit]])
    expected = result(["operating_revenue_yuan"], [[12]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


def test_yuan_label_cannot_compensate_for_incorrect_amount():
    actual = result(["operating_revenue_yuan", "unit"], [[12, "元"]])
    expected = result(["operating_revenue_yuan"], [[120000]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


@pytest.mark.parametrize("measure", ["person_count", "overall_profit_rate_pct"])
def test_money_unit_is_not_auxiliary_to_nonmonetary_output(measure):
    actual = result([measure, "unit"], [[12, "元"]])
    expected = result([measure], [[12]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


@pytest.mark.parametrize("context", [None, {}, {"amount_unit": "万元"}])
def test_auxiliary_unit_requires_private_yuan_contract(context):
    actual = result(["operating_revenue_yuan", "unit"], [[12, "元"]])
    expected = result(["operating_revenue_yuan"], [[12]])
    assert not compare_submission_results(actual, expected, context=context)


@pytest.mark.parametrize(
    ("extra_name", "extra_value"),
    [("enterprise_id", "E2"), ("enterprise_name", "乙公司"),
     ("enterprise_id", None), ("enterprise_name", "甲")],
)
def test_wrong_auxiliary_enterprise_is_not_ignored(extra_name, extra_value):
    actual = result(["operating_revenue_yuan", extra_name], [[12, extra_value]])
    expected = result(["operating_revenue_yuan"], [[12]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


def test_entity_id_and_name_must_describe_the_same_enterprise():
    actual = result(["enterprise_id", "enterprise_name", "operating_revenue_yuan"],
                    [["E1", "乙公司", 12]])
    expected = result(["enterprise_id", "operating_revenue_yuan"], [["E1", 12]])
    assert not compare_submission_results(actual, expected, context=TWO_ENTERPRISES)


def test_multi_enterprise_auxiliary_names_bind_to_required_row_identity():
    expected = result(["enterprise_id", "operating_revenue_yuan"], [["E1", 12], ["E2", 12]])
    actual = result(["operating_revenue_yuan", "enterprise_name", "enterprise_id"],
                    [[12, "乙公司", "E2"], [12, "甲公司", "E1"]])
    assert compare_submission_results(actual, expected, context=TWO_ENTERPRISES)
    actual["rows"] = [[12, "甲公司", "E2"], [12, "乙公司", "E1"]]
    assert not compare_submission_results(actual, expected, context=TWO_ENTERPRISES)


def test_multiple_enterprises_without_required_row_identity_cannot_be_guessed():
    actual = result(["enterprise_name", "operating_revenue_yuan"], [["甲公司", 12]])
    expected = result(["operating_revenue_yuan"], [[12]])
    assert not compare_submission_results(actual, expected, context=TWO_ENTERPRISES)


@pytest.mark.parametrize(
    "context",
    [None, {}, {"enterprises": []}, {"enterprises": "甲公司"},
     {"enterprises": [None]}, {"enterprises": [{}]},
     {"enterprises": [{"enterprise_id": "E1"}]},
     {"enterprises": [{"enterprise_id": "E1", "enterprise_name": "甲公司"},
                      {"enterprise_id": "E1", "enterprise_name": "甲公司"}]},
     {"enterprises": [{"enterprise_id": "E1", "enterprise_name": ""}]}],
)
@pytest.mark.parametrize("empty", [False, True])
def test_entity_context_must_be_valid_even_for_empty_results(context, empty):
    actual = result(["enterprise_name", "operating_revenue_yuan"],
                    [] if empty else [["甲公司", 12]])
    expected = result(["operating_revenue_yuan"], [] if empty else [[12]])
    assert not compare_submission_results(actual, expected, context=context)


@pytest.mark.parametrize("field", ["enterprise_id", "enterprise_name", "amount_unit"])
def test_whitelisted_field_is_not_optional_when_reference_requires_it(field):
    expected = result(["operating_revenue_yuan", field], [[12, "required"]])
    actual = result(["operating_revenue_yuan"], [[12]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


def test_status_cannot_replace_required_month_even_when_width_matches():
    actual = result(["tax_amount", "value_status"], [[0, "确认零"]])
    expected = result(["stat_month", "tax_amount_yuan"], [["2026-02-01", 0]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


def test_month_label_normalization_is_symmetric_and_does_not_relax_strict_reference():
    actual = result(["month"], [["2026-02"]])
    expected = result(["stat_month"], [["2026-02-01"]])
    assert compare_submission_results(actual, expected)
    assert compare_submission_results(expected, actual)
    assert not compare_results(actual, expected)


@pytest.mark.parametrize("value", ["2026-02-02", "2026-02-28", "2026-02-01T00:00:00",
                                  "2026-2", "2026-02-01 00:00:00"])
def test_month_normalization_does_not_truncate_other_dates_or_timestamps(value):
    assert not compare_submission_results(
        result(["stat_month"], [[value]]), result(["stat_month"], [["2026-02"]]))


@pytest.mark.parametrize("value", ["2026-00", "2026-13", "0000-01", "2026-13-01"])
def test_invalid_calendar_months_do_not_pass_even_if_identical(value):
    actual = result(["stat_month"], [[value]])
    assert not compare_submission_results(actual, actual)


@pytest.mark.parametrize("field", ["event_date", "start_date", "balance_as_of_date", "custom_date"])
def test_month_normalization_does_not_apply_to_other_date_fields(field):
    assert not compare_submission_results(
        result([field], [["2026-02"]]), result([field], [["2026-02-01"]]))


def test_projection_preserves_duplicate_row_multiplicity():
    expected = result(["stat_month", "net_profit_yuan"], [["2026-08-01", 12]])
    actual = result(["enterprise_name", "month", "net_profit"],
                    [["甲公司", "2026-08", 12], ["甲公司", "2026-08", 12]])
    assert not compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)
    expected["rows"].append(["2026-08-01", 12])
    assert compare_submission_results(actual, expected, context=SINGLE_ENTERPRISE)


def test_row_order_and_decimal_trailing_zeros_are_irrelevant():
    actual = result(["net_profit", "month"], [["1.2300", "2026-08"], [None, "2026-07"]])
    expected = result(["stat_month", "net_profit_yuan"],
                      [["2026-07-01", None], ["2026-08-01", Decimal("1.23")]])
    assert compare_submission_results(actual, expected)


@pytest.mark.parametrize("replacement", [0, "0.00", "", "NULL", False])
def test_null_remains_distinct_from_zero_text_and_boolean(replacement):
    assert not compare_submission_results(
        result(["tax_amount"], [[replacement]]), result(["tax_amount_yuan"], [[None]]))


def test_decimal_precision_is_not_rounded_or_tolerated():
    value = "123456789012345678901234567890.12345678901234567890"
    different = "123456789012345678901234567890.12345678901234567891"
    expected = result(["net_profit_yuan"], [[Decimal(value)]])
    with localcontext() as context:
        context.prec = 3
        assert compare_submission_results(result(["net_profit"], [[value]]), expected)
        assert not compare_submission_results(result(["net_profit"], [[different]]), expected)
        assert not compare_submission_results(
            result(["net_profit"], [["1.2345671"]]), result(["net_profit_yuan"], [["1.234567"]]))


def test_comparison_never_aggregates_or_selects_a_preferred_month():
    actual = result(["stat_month", "operating_revenue_yuan"],
                    [["2026-07-01", 12], ["2026-08-01", 15]])
    assert not compare_submission_results(
        actual, result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 15]]))
    assert not compare_submission_results(
        result(["net_profit"], [[12], [15]]), result(["net_profit_yuan"], [[27]]))


@pytest.mark.parametrize(
    "malformed",
    [None, {}, {"columns": [], "rows": []},
     result([None], [[1]]), result(["net_profit"], [[1, 2]]),
     result(["net_profit"], [[{"nested": 1}]]),
     result(["net_profit"], [[float("nan")]])],
)
def test_malformed_submission_does_not_match(malformed):
    assert not compare_submission_results(malformed, malformed)
    assert not compare_submission_results(malformed, result(["net_profit_yuan"], [[1]]))


def test_evaluation_forwards_private_context_and_preserves_original_record():
    actual = result(["enterprise_name", "net_profit", "month"],
                    [["甲公司", "12.00", "2026-08"]])
    operation = {
        "kind": "sql", "query_id": "submitted", "sql": "SELECT submitted_result",
        "success": True, "is_complete": True, **actual,
    }
    record = {"status": "completed", "result_query_id": "submitted",
              "operations": [operation], "answer": "正文不参与评分。"}
    expected = result(["stat_month", "net_profit_yuan"], [["2026-08-01", 12]])
    before = deepcopy(record)
    evaluated = evaluate_result(record, expected, context=SINGLE_ENTERPRISE)
    assert evaluated == {
        "verdict": "pass", "query_id": "submitted", "sql": "SELECT submitted_result",
        "actual": actual,
    }
    assert record == before
    assert evaluate_result(record, expected)["verdict"] == "fail"
    record["operations"][0]["is_complete"] = False
    assert evaluate_result(record, expected, context=SINGLE_ENTERPRISE)["verdict"] == "fail"
