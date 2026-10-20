"""Expected-coverage contracts; all execution is reserved for the user."""
from copy import deepcopy

import pytest

from src.psu_memory.validation.comparison import compare_submission_results
from src.psu_memory.validation.coverage_comparison import (
    COMPARISON_VERSION,
    compare_coverage_results,
    evaluate_coverage_result,
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


def evidence_context(expected, extra_rows, *, base=SINGLE_ENTERPRISE):
    """Build synthetic, independently supplied evidence in expected row order."""
    context = deepcopy(base)
    context["coverage_evidence"] = {
        "version": "expected_coverage_v3",
        "rows": [
            {**dict(zip(expected["columns"], row)), **extras}
            for row, extras in zip(expected["rows"], extra_rows)
        ],
    }
    return context


def submission(actual, *, sql="SELECT stat_month, operating_revenue FROM monthly"):
    return {
        "status": "completed",
        "result_query_id": "q_final",
        "operations": [{
            "kind": "sql",
            "query_id": "q_final",
            "sql": sql,
            "success": True,
            "is_complete": True,
            **deepcopy(actual),
        }],
    }


def test_coverage_version_is_separate_from_historical_v2():
    assert COMPARISON_VERSION == "expected_coverage_v3"
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    actual = result(
        ["stat_month", "operating_revenue_yuan", "main_business_revenue"],
        [["2026-08-01", 12, 9]],
    )
    context = evidence_context(expected, [{"main_business_revenue": 9}])
    assert compare_coverage_results(actual, expected, context=context)
    assert not compare_submission_results(actual, expected, context=context)


@pytest.mark.parametrize(
    ("extra_column", "canonical", "value"),
    [
        ("主营收入", "main_business_revenue", "9.00"),
        ("净利润", "net_profit_yuan", "-3.00"),
        ("invoice_sales_amount", "invoice_sales_amount", "15.00"),
        ("销项发票额_元", "invoice_sales_amount", "15.00"),
        ("net_profit_margin_pct", "net_profit_margin_pct", "-25.000000"),
    ],
)
def test_verified_business_extras_are_accepted(extra_column, canonical, value):
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    actual = result(
        ["企业名称", extra_column, "营业收入_元", "月份", "单位"],
        [["甲公司", value, "12.00", "2026-08", "元"]],
    )
    context = evidence_context(expected, [{canonical: value}])
    before = deepcopy((actual, expected, context))
    assert compare_coverage_results(actual, expected, context=context)
    assert (actual, expected, context) == before


@pytest.mark.parametrize("defect", ["wrong", "null", "missing_evidence", "unknown_extra"])
def test_correct_core_cannot_hide_unverified_or_wrong_business_extra(defect):
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    actual = result(
        ["stat_month", "operating_revenue_yuan", "net_profit_yuan"],
        [["2026-08-01", 12, -3]],
    )
    context = evidence_context(expected, [{"net_profit_yuan": -3}])
    if defect == "wrong":
        actual["rows"][0][2] = -2
    elif defect == "null":
        actual["rows"][0][2] = None
    elif defect == "missing_evidence":
        context = deepcopy(SINGLE_ENTERPRISE)
    else:
        actual["columns"][2] = "some_business_measure"
    assert not compare_coverage_results(actual, expected, context=context)


@pytest.mark.parametrize("defect", ["wrong_version", "missing_row", "core_conflict"])
def test_business_evidence_must_match_the_current_expected_result(defect):
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    actual = result(
        ["stat_month", "operating_revenue_yuan", "net_profit_yuan"],
        [["2026-08-01", 12, -3]],
    )
    context = evidence_context(expected, [{"net_profit_yuan": -3}])
    if defect == "wrong_version":
        context["coverage_evidence"]["version"] = "explicit_columns_aux_v2"
    elif defect == "missing_row":
        context["coverage_evidence"]["rows"] = []
    else:
        context["coverage_evidence"]["rows"][0]["operating_revenue_yuan"] = 13
    assert not compare_coverage_results(actual, expected, context=context)


@pytest.mark.parametrize("missing", ["business_month_count", "period_revenue_yuan"])
def test_required_counts_and_totals_are_not_reconstructed_from_detail(missing):
    expected = result(
        ["stat_month", "operating_revenue_yuan", "business_month_count",
         "period_revenue_yuan"],
        [["2026-07-01", 10, 2, 30], ["2026-08-01", 20, 2, 30]],
    )
    kept = [index for index, column in enumerate(expected["columns"]) if column != missing]
    actual = result(
        [expected["columns"][index] for index in kept],
        [[row[index] for index in kept] for row in expected["rows"]],
    )
    context = evidence_context(expected, [{}, {}])
    assert not compare_coverage_results(actual, expected, context=context)


def test_required_judgment_is_not_inferred_from_monthly_values():
    expected = result(
        ["stat_month", "net_profit_yuan", "any_loss"],
        [["2026-07-01", -2, 1], ["2026-08-01", 4, 1]],
    )
    actual = result(["stat_month", "net_profit_yuan"],
                    [["2026-07-01", -2], ["2026-08-01", 4]])
    assert not compare_coverage_results(
        actual, expected, context=evidence_context(expected, [{}, {}]))


def test_core_and_verified_extras_can_be_reordered_with_month_normalization():
    expected = result(
        ["stat_month", "operating_revenue_yuan"],
        [["2026-07-01", 10], ["2026-08-01", 20]],
    )
    actual = result(
        ["net_profit_yuan", "营业收入", "企业名称", "月份"],
        [[-3, 20, "甲公司", "2026-08"], [2, 10, "甲公司", "2026-07"]],
    )
    context = evidence_context(expected, [{"net_profit_yuan": 2}, {"net_profit_yuan": -3}])
    assert compare_coverage_results(actual, expected, context=context)
    # Extras must follow the matched row, not merely match the same multiset.
    actual["rows"][0][0], actual["rows"][1][0] = 2, -3
    assert not compare_coverage_results(actual, expected, context=context)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "wrong_month", "wrong_core"])
def test_coverage_keeps_the_exact_core_row_multiset(defect):
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    actual = result(
        ["stat_month", "operating_revenue_yuan", "net_profit_yuan"],
        [["2026-08-01", 12, -3]],
    )
    context = evidence_context(expected, [{"net_profit_yuan": -3}])
    if defect == "duplicate":
        actual["rows"].append(list(actual["rows"][0]))
    elif defect == "missing":
        actual["rows"] = []
    elif defect == "wrong_month":
        actual["rows"][0][0] = "2026-07-01"
    else:
        actual["rows"][0][1] = 13
    assert not compare_coverage_results(actual, expected, context=context)


def test_null_is_distinct_from_zero_in_both_core_and_extra_values():
    expected = result(["stat_month", "net_profit_yuan"], [["2026-08-01", None]])
    actual = result(
        ["stat_month", "net_profit_yuan", "invoice_sales_amount"],
        [["2026-08-01", None, None]],
    )
    context = evidence_context(expected, [{"invoice_sales_amount": None}])
    assert compare_coverage_results(actual, expected, context=context)
    for index in (1, 2):
        wrong = deepcopy(actual)
        wrong["rows"][0][index] = 0
        assert not compare_coverage_results(wrong, expected, context=context)


def test_multiple_enterprises_bind_business_extras_to_the_correct_entity():
    expected = result(
        ["enterprise_id", "stat_month", "operating_revenue_yuan"],
        [["E1", "2026-08-01", 12], ["E2", "2026-08-01", 12]],
    )
    actual = result(
        ["enterprise_name", "net_profit_yuan", "enterprise_id", "stat_month",
         "operating_revenue_yuan"],
        [["乙公司", -5, "E2", "2026-08", 12], ["甲公司", -3, "E1", "2026-08", 12]],
    )
    context = evidence_context(
        expected, [{"net_profit_yuan": -3}, {"net_profit_yuan": -5}], base=TWO_ENTERPRISES)
    assert compare_coverage_results(actual, expected, context=context)
    actual["rows"][0][1], actual["rows"][1][1] = -3, -5
    assert not compare_coverage_results(actual, expected, context=context)


def test_extra_enterprise_label_must_agree_with_the_required_identifier():
    expected = result(["enterprise_id", "operating_revenue_yuan"], [["E1", 12]])
    actual = result(
        ["enterprise_id", "operating_revenue_yuan", "enterprise_name"],
        [["E1", 12, "乙公司"]],
    )
    context = evidence_context(expected, [{}], base=TWO_ENTERPRISES)
    assert not compare_coverage_results(actual, expected, context=context)
    actual["rows"][0][2] = "甲公司"
    assert compare_coverage_results(actual, expected, context=context)


@pytest.mark.parametrize("suffix", ["_元", "(元)", "（元）", "_yuan"])
def test_explicit_yuan_suffix_on_core_and_verified_extra_is_supported(suffix):
    expected = result(["operating_revenue_yuan"], [[12]])
    actual = result(["营业收入" + suffix, "净利润" + suffix], [[12, -3]])
    context = evidence_context(expected, [{"net_profit_yuan": -3}])
    assert compare_coverage_results(actual, expected, context=context)


@pytest.mark.parametrize("defect", ["core_unit", "extra_unit", "unit_value"])
def test_conflicting_units_fail_even_when_numeric_values_match(defect):
    expected = result(["operating_revenue_yuan"], [[12]])
    actual = result(["营业收入_元", "净利润_元", "amount_unit"], [[12, -3, "元"]])
    context = evidence_context(expected, [{"net_profit_yuan": -3}])
    if defect == "core_unit":
        actual["columns"][0] = "营业收入_万元"
    elif defect == "extra_unit":
        actual["columns"][1] = "净利润(万元)"
    else:
        actual["rows"][0][2] = "万元"
    assert not compare_coverage_results(actual, expected, context=context)


def test_duplicate_semantic_core_columns_remain_ambiguous():
    expected = result(["operating_revenue_yuan"], [[12]])
    actual = result(["operating_revenue_yuan", "营业收入_元"], [[12, 12]])
    assert not compare_coverage_results(
        actual, expected, context=evidence_context(expected, [{}]))


@pytest.mark.parametrize("column", ["custom_answer", "tax_amount_yuan"])
def test_equal_values_do_not_establish_column_correspondence(column):
    expected = result(["operating_revenue_yuan"], [[12]])
    actual = result([column], [[12]])
    assert not compare_coverage_results(
        actual, expected, context=evidence_context(expected, [{}]))


SUM_REVENUE = (
    "SUM(CASE WHEN eom.amount_unit = '万元' THEN eom.operating_revenue * 10000 "
    "ELSE eom.operating_revenue END)"
)
SUM_PROFIT = (
    "SUM(CASE WHEN eom.amount_unit = '万元' THEN eom.total_profit * 10000 "
    "ELSE eom.total_profit END)"
)
SUM_NET_PROFIT = (
    "SUM(CASE WHEN eom.amount_unit = '万元' THEN eom.net_profit * 10000 "
    "ELSE eom.net_profit END)"
)
PERIOD_SQL = f"""
SELECT eb.enterprise_name,
       ROUND({SUM_REVENUE}, 2) AS total_operating_revenue_yuan,
       ROUND({SUM_PROFIT}, 2) AS total_profit_yuan,
       ROUND({SUM_PROFIT} / NULLIF({SUM_REVENUE}, 0) * 100, 6) AS profit_margin_pct,
       ROUND({SUM_NET_PROFIT}, 2) AS total_net_profit_yuan
FROM enterprise_basic eb
JOIN enterprise_operation_monthly eom ON eb.enterprise_id = eom.enterprise_id
WHERE eb.enterprise_id = 'E1'
  AND eom.stat_month IN ('2026-07-01', '2026-08-01')
  AND eom.report_status = '正式'
GROUP BY eb.enterprise_id, eb.enterprise_name
"""


def period_fixture():
    expected = result(
        ["period_revenue_yuan", "period_total_profit_yuan", "overall_profit_rate_pct"],
        [[30, 6, 20]],
    )
    actual = result(
        ["enterprise_name", "total_operating_revenue_yuan", "total_profit_yuan",
         "profit_margin_pct", "total_net_profit_yuan"],
        [["甲公司", 30, 6, 20, 4]],
    )
    context = evidence_context(expected, [{"period_net_profit_yuan": 4}])
    return actual, expected, context


def test_period_aliases_require_aggregate_sql_and_independent_extra_evidence():
    actual, expected, context = period_fixture()
    assert compare_coverage_results(actual, expected, context=context, sql=PERIOD_SQL)
    verdict = evaluate_coverage_result(
        submission(actual, sql=PERIOD_SQL), expected, context=context)
    assert verdict["verdict"] == "pass"
    assert verdict["query_id"] == "q_final"
    assert verdict["sql"] == PERIOD_SQL
    assert verdict["actual"] == actual
    assert "reason" in verdict


def test_period_aliases_are_not_proven_by_values_or_monthly_sql():
    actual, expected, context = period_fixture()
    monthly_sql = """
        SELECT eb.enterprise_name,
               eom.operating_revenue AS total_operating_revenue_yuan,
               eom.total_profit AS total_profit_yuan,
               eom.total_profit / NULLIF(eom.operating_revenue, 0) * 100 AS profit_margin_pct,
               eom.net_profit AS total_net_profit_yuan
        FROM enterprise_basic eb
        JOIN enterprise_operation_monthly eom ON eb.enterprise_id = eom.enterprise_id
        WHERE eb.enterprise_id = 'E1' AND eom.stat_month = '2026-08-01'
    """
    assert not compare_coverage_results(actual, expected, context=context)
    assert not compare_coverage_results(actual, expected, context=context, sql=monthly_sql)


def test_sum_of_a_different_measure_cannot_prove_period_profit_alias():
    actual, expected, context = period_fixture()
    wrong_measure = PERIOD_SQL.replace(SUM_PROFIT, SUM_NET_PROFIT)
    assert not compare_coverage_results(actual, expected, context=context, sql=wrong_measure)


def test_average_monthly_ratio_cannot_prove_overall_profit_rate_alias():
    actual, expected, context = period_fixture()
    wrong_rate = PERIOD_SQL.replace(
        f"ROUND({SUM_PROFIT} / NULLIF({SUM_REVENUE}, 0) * 100, 6)",
        "ROUND(AVG(eom.total_profit / NULLIF(eom.operating_revenue, 0)) * 100, 6)")
    assert not compare_coverage_results(actual, expected, context=context, sql=wrong_rate)


@pytest.mark.parametrize("defect", ["wrong_extra", "missing_extra_evidence"])
def test_period_sql_does_not_itself_prove_extra_values(defect):
    actual, expected, context = period_fixture()
    if defect == "wrong_extra":
        actual["rows"][0][4] = 5
    else:
        del context["coverage_evidence"]["rows"][0]["period_net_profit_yuan"]
    assert not compare_coverage_results(actual, expected, context=context, sql=PERIOD_SQL)


def test_verified_period_net_profit_rate_can_accompany_complete_core():
    actual, expected, context = period_fixture()
    actual["columns"].append("net_profit_margin_pct")
    actual["rows"][0].append("13.333333")
    context["coverage_evidence"]["rows"][0]["overall_net_profit_rate_pct"] = "13.333333"
    sql = PERIOD_SQL.replace(
        f"ROUND({SUM_NET_PROFIT}, 2) AS total_net_profit_yuan",
        f"ROUND({SUM_NET_PROFIT}, 2) AS total_net_profit_yuan, "
        f"ROUND({SUM_NET_PROFIT} / NULLIF({SUM_REVENUE}, 0) * 100, 6) AS net_profit_margin_pct")
    assert compare_coverage_results(actual, expected, context=context, sql=sql)
    actual["rows"][0][-1] = "13.000000"
    assert not compare_coverage_results(actual, expected, context=context, sql=sql)


@pytest.mark.parametrize(
    "defect",
    ["failed_status", "failed_sql", "incomplete_result", "duplicate_submission",
     "missing_query_id"],
)
def test_correct_values_do_not_override_execution_or_submission_failures(defect):
    expected = result(["stat_month", "operating_revenue_yuan"], [["2026-08-01", 12]])
    record = submission(expected)
    if defect == "failed_status":
        record["status"] = "failed"
    elif defect == "failed_sql":
        record["operations"][0]["success"] = False
    elif defect == "incomplete_result":
        record["operations"][0]["is_complete"] = False
    elif defect == "duplicate_submission":
        record["operations"].append(deepcopy(record["operations"][0]))
    else:
        record["result_query_id"] = None
    verdict = evaluate_coverage_result(
        record, expected, context=evidence_context(expected, [{}]))
    assert verdict["verdict"] == "fail"
    assert verdict.get("reason")
