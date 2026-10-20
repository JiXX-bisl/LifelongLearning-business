"""Static evidence preparation fixtures; execution is reserved for the user."""
from copy import deepcopy
import json

import pytest

from src.psu_memory.validation.coverage_evidence import build_coverage_contexts


PERIOD_SQL = """
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY enterprise_id, source_system, source_record_ref
               ORDER BY report_revision DESC, ingested_at DESC, monthly_record_id DESC
           ) AS rn
    FROM enterprise_operation_monthly
    WHERE enterprise_id = 'E1'
      AND stat_month IN ('2026-07-01', '2026-08-01')
      AND ingested_at <= '2026-09-15 23:59:59'
)
SELECT SUM(operating_revenue) AS period_revenue_yuan,
       SUM(total_profit) AS period_total_profit_yuan,
       ROUND(SUM(total_profit) / NULLIF(SUM(operating_revenue), 0) * 100, 6)
           AS overall_profit_rate_pct
FROM ranked
WHERE rn = 1 AND report_status = '正式'
"""


def monthly(month="2026-08-01", **changes):
    row = {
        "monthly_record_id": "record-" + month,
        "enterprise_id": "E1",
        "stat_month": month,
        "source_system": "monthly-report",
        "source_record_ref": "report-" + month,
        "report_revision": 1,
        "report_status": "正式",
        "reported_at": "2026-09-01 09:00:00",
        "ingested_at": "2026-09-01 10:00:00",
        "amount_unit": "元",
        "operating_revenue": "12.00",
        "main_business_revenue": "9.00",
        "total_profit": "3.00",
        "net_profit": "2.00",
        "tax_amount": "1.00",
        "invoice_purchase_amount": "4.00",
        "invoice_sales_amount": "15.00",
        "social_security_amount": "0.50",
        "travel_hospitality_invoice_amount": "0.00",
        "utility_invoice_amount": "0.20",
    }
    row.update(changes)
    return row


def expected_month(value="12.00"):
    return {
        "columns": ["stat_month", "operating_revenue_yuan"],
        "rows": [["2026-08-01", value]],
    }


def expected_period():
    return {
        "columns": [
            "period_revenue_yuan", "period_total_profit_yuan", "overall_profit_rate_pct",
        ],
        "rows": [["30.00", "6.00", "20.000000"]],
    }


def inputs(tmp_path, records, expected, *, reference_sql=PERIOD_SQL):
    tables = {
        "enterprise_basic": [{"enterprise_id": "E1", "enterprise_name": "甲公司"}],
        "enterprise_operation_monthly": records,
        "enterprise_person": [],
        "enterprise_negative_event": [],
        "enterprise_operation_event": [],
    }
    tables_path = tmp_path / "tables.json"
    tables_path.write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps({"cases": [{"id": "case-1", "reference_sql": reference_sql}]},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "cases": [{"id": "case-1"}],
        "standards": {"case-1": deepcopy(expected)},
        "ground_truth_path": ground_truth_path,
        "baseline_dir": tmp_path,
        "contexts": {
            "case-1": {
                "enterprises": [{"enterprise_id": "E1", "enterprise_name": "甲公司"}],
                "amount_unit": "元",
            },
        },
    }


def evidence_rows(contexts):
    evidence = contexts["case-1"]["coverage_evidence"]
    assert evidence["version"] == "expected_coverage_v3"
    return evidence["rows"]


def test_latest_withdrawal_cannot_fall_back_to_an_older_formal_revision(tmp_path):
    older = monthly()
    latest = monthly(
        monthly_record_id="latest-withdrawn",
        report_revision=2,
        report_status="撤回",
        ingested_at="2026-09-02 10:00:00",
    )
    args = inputs(tmp_path, [older, latest], expected_month())
    row = evidence_rows(build_coverage_contexts(**args))[0]
    assert row["operating_revenue_yuan"] == "12.00"
    assert "net_profit_yuan" not in row
    assert "main_business_revenue" not in row

    # Restoring formal status on that same latest revision authorizes its values.
    latest.update(report_status="正式", net_profit="8.00")
    args = inputs(tmp_path, [older, latest], expected_month())
    row = evidence_rows(build_coverage_contexts(**args))[0]
    assert row["net_profit_yuan"] == "8.00"
    assert row["report_revision"] == 2


def test_ten_thousand_yuan_is_converted_before_half_up_rounding(tmp_path):
    source = monthly(
        amount_unit="万元",
        operating_revenue="0.0012345",
        main_business_revenue="0.0011005",
        net_profit="-0.0001005",
    )
    args = inputs(tmp_path, [source], expected_month("12.35"))
    row = evidence_rows(build_coverage_contexts(**args))[0]
    assert row["amount_unit"] == "元"
    assert row["operating_revenue_yuan"] == "12.35"
    assert row["main_business_revenue"] == "11.01"
    assert row["net_profit_yuan"] == "-1.01"


def test_static_supplements_never_rewrite_the_required_answer(tmp_path):
    expected = {
        "columns": [
            "stat_month", "operating_revenue_yuan", "business_month_count",
            "period_revenue_yuan",
        ],
        "rows": [["2026-08-01", "99.00", 2, "120.00"]],
    }
    args = inputs(tmp_path, [monthly()], expected)
    before = deepcopy((args["standards"], args["contexts"]))
    row = evidence_rows(build_coverage_contexts(**args))[0]
    for key, value in zip(expected["columns"], expected["rows"][0]):
        assert row[key] == value
    assert "net_profit_yuan" not in row
    assert "main_business_revenue" not in row
    assert (args["standards"], args["contexts"]) == before
    # Required counts/totals remain requirements; static data cannot remove them
    # from the original expected result or substitute a different expected answer.
    assert args["standards"]["case-1"] == expected


def test_conflicting_sources_for_the_same_month_do_not_get_arbitrary_priority(tmp_path):
    first = monthly()
    second = monthly(
        monthly_record_id="other-source-row",
        source_system="other-report-system",
        source_record_ref="other-source-report",
        net_profit="7.00",
    )
    args = inputs(tmp_path, [first, second], expected_month())
    row = evidence_rows(build_coverage_contexts(**args))[0]
    assert row["operating_revenue_yuan"] == "12.00"
    assert "net_profit_yuan" not in row
    assert "main_business_revenue" not in row


@pytest.mark.parametrize("conflict_index", [None, 0, 1, 2])
def test_period_supplement_uses_only_explicit_scope_and_consistent_core(
        tmp_path, conflict_index):
    records = [
        monthly("2026-07-01", operating_revenue="10", total_profit="2", net_profit="1"),
        monthly("2026-08-01", operating_revenue="20", total_profit="4", net_profit="3"),
        # An existing month outside the approved reference scope must not enter it.
        monthly("2026-06-01", operating_revenue="900", total_profit="90", net_profit="90"),
    ]
    expected = expected_period()
    if conflict_index is not None:
        expected["rows"][0][conflict_index] = "999"
    args = inputs(tmp_path, records, expected)
    row = evidence_rows(build_coverage_contexts(**args))[0]
    for key, value in zip(expected["columns"], expected["rows"][0]):
        assert row[key] == value
    if conflict_index is None:
        assert row["period_net_profit_yuan"] == "4.00"
        assert row["business_month_count"] == 2
    else:
        assert "period_net_profit_yuan" not in row
        assert "business_month_count" not in row


def test_period_totals_and_ratio_do_not_sum_already_rounded_monthly_amounts(tmp_path):
    records = [
        monthly("2026-07-01", operating_revenue="0.015", total_profit="0.004",
                net_profit="0.004"),
        monthly("2026-08-01", operating_revenue="0.015", total_profit="0.004",
                net_profit="0.004"),
    ]
    expected = expected_period()
    expected["rows"] = [["0.03", "0.01", "26.666667"]]
    args = inputs(tmp_path, records, expected)
    row = evidence_rows(build_coverage_contexts(**args))[0]
    assert row["period_net_profit_yuan"] == "0.01"
    assert row["overall_net_profit_rate_pct"] == "26.666667"


@pytest.mark.parametrize("defect", ["unknown_amount", "missing_month", "implicit_scope"])
def test_unknown_or_unbounded_period_cannot_be_presented_as_a_known_total(tmp_path, defect):
    records = [
        monthly("2026-07-01", operating_revenue="10", total_profit="2", net_profit="1"),
        monthly("2026-08-01", operating_revenue="20", total_profit="4", net_profit="3"),
    ]
    reference_sql = PERIOD_SQL
    if defect == "unknown_amount":
        records[1]["net_profit"] = None
    elif defect == "missing_month":
        records.pop()
    else:
        reference_sql = PERIOD_SQL.replace(
            "stat_month IN ('2026-07-01', '2026-08-01')", "YEAR(stat_month) = 2026")
    args = inputs(tmp_path, records, expected_period(), reference_sql=reference_sql)
    row = evidence_rows(build_coverage_contexts(**args))[0]
    assert "period_net_profit_yuan" not in row
    assert "overall_net_profit_rate_pct" not in row
    assert row["period_revenue_yuan"] == "30.00"


def test_returned_evidence_is_frozen_against_later_source_and_input_changes(tmp_path):
    args = inputs(tmp_path, [monthly()], expected_month())
    frozen = build_coverage_contexts(**args)
    snapshot = deepcopy(frozen)
    assert evidence_rows(frozen)[0]["net_profit_yuan"] == "2.00"

    tables_path = tmp_path / "tables.json"
    changed_tables = json.loads(tables_path.read_text(encoding="utf-8"))
    changed_tables["enterprise_operation_monthly"][0]["net_profit"] = "999.00"
    tables_path.write_text(json.dumps(changed_tables, ensure_ascii=False), encoding="utf-8")
    args["ground_truth_path"].write_text('{"cases": []}', encoding="utf-8")
    args["standards"]["case-1"]["rows"][0][1] = "999.00"
    args["contexts"]["case-1"]["enterprises"][0]["enterprise_name"] = "后改名称"
    assert frozen == snapshot


def test_duplicate_sources_cannot_silently_reduce_supplementary_period_sum(tmp_path):
    # The reference SUM sees three source rows, even though two are identical.
    # Zero core amounts cannot expose an erroneous collapse by themselves.
    first = monthly("2026-07-01", operating_revenue="0", total_profit="0", net_profit="1")
    duplicate = {**first, "monthly_record_id": "duplicate", "source_record_ref": "duplicate-source"}
    second = monthly("2026-08-01", operating_revenue="0", total_profit="0", net_profit="1")
    expected = expected_period()
    expected["rows"] = [["0.00", "0.00", None]]
    args = inputs(tmp_path, [first, duplicate, second], expected)
    row = evidence_rows(build_coverage_contexts(**args))[0]
    assert "period_net_profit_yuan" not in row


def test_downstream_month_filter_cannot_be_ignored_by_supplementary_evidence(tmp_path):
    records = [
        monthly("2026-07-01", operating_revenue="10", total_profit="2", net_profit="1"),
        monthly("2026-08-01", operating_revenue="20", total_profit="4", net_profit="3"),
    ]
    sql = PERIOD_SQL.replace(
        "WHERE rn = 1 AND report_status = '正式'",
        "WHERE rn = 1 AND report_status = '正式' AND stat_month = '2026-08-01'")
    expected = expected_period()
    expected["rows"] = [["20.00", "4.00", "20.000000"]]
    row = evidence_rows(build_coverage_contexts(
        **inputs(tmp_path, records, expected, reference_sql=sql)))[0]
    assert "period_net_profit_yuan" not in row


def test_report_count_does_not_authorize_latest_monthly_business_values(tmp_path):
    expected = {"columns": ["stat_month", "submitted_record_count"],
                "rows": [["2026-08-01", 1]]}
    row = evidence_rows(build_coverage_contexts(
        **inputs(tmp_path, [monthly()], expected)))[0]
    assert "operating_revenue_yuan" not in row
    assert "net_profit_yuan" not in row


@pytest.mark.parametrize("wrong_count", [False, True])
def test_single_month_count_and_sum_can_bind_supplementary_month_fields(tmp_path, wrong_count):
    expected = {"columns": ["business_month_count", "operating_revenue_yuan"],
                "rows": [[2 if wrong_count else 1, "12.00"]]}
    sql = PERIOD_SQL.replace(
        "stat_month IN ('2026-07-01', '2026-08-01')", "stat_month = '2026-08-01'")
    args = inputs(tmp_path, [monthly()], expected, reference_sql=sql)
    row = evidence_rows(build_coverage_contexts(**args))[0]
    if wrong_count:
        assert "stat_month" not in row
        assert "main_business_revenue" not in row
    else:
        assert row["stat_month"] == "2026-08-01"
        assert row["main_business_revenue"] == "9.00"
