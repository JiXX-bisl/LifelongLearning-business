#!/usr/bin/env python3
"""Run read-only SQL acceptance checks for the 40-company V2.1 baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = (
    ROOT / 'data/sql/synthetic_enterprise_v21_40_v1.0.0'
)

EXPECTED_COUNTS = {
    "enterprise_basic": 40,
    "enterprise_person": 234,
    "enterprise_operation_profile": 40,
    "enterprise_operation_monthly": 480,
    "enterprise_operation_event": 72,
    "enterprise_negative_event": 34,
}

EXPECTED_ENTERPRISE_ID_HASH = ""  # Initialized deterministically in main.


def json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


def comparable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(comparable(item) for item in value)
    if isinstance(value, tuple):
        return tuple(comparable(item) for item in value)
    return value


def fetch_one(cursor: Any, sql: str) -> Any:
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None:
        return None
    return row[0] if len(row) == 1 else tuple(row)


def check(
    cursor: Any,
    category: str,
    name: str,
    sql: str,
    expected: Any,
) -> dict[str, Any]:
    actual = fetch_one(cursor, sql)
    passed = comparable(actual) == comparable(expected)
    return {
        "category": category,
        "name": name,
        "actual": json_value(actual),
        "expected": json_value(expected),
        "passed": passed,
    }


def build_checks(cursor: Any, database: str, enterprise_id_hash: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    checks.append(check(
        cursor, "environment", "mysql_major_minor", "SELECT SUBSTRING_INDEX(VERSION(), '.', 2)", "8.0"
    ))
    checks.append(check(
        cursor,
        "environment",
        "database_charset_and_collation",
        "SELECT @@character_set_database, @@collation_database",
        ("utf8mb4", "utf8mb4_0900_ai_ci"),
    ))
    checks.append(check(
        cursor,
        "schema",
        "table_count",
        f"SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA = '{database}' AND TABLE_TYPE = 'BASE TABLE'",
        6,
    ))
    for constraint_type, expected in (
        ("PRIMARY KEY", 6),
        ("FOREIGN KEY", 6),
        ("UNIQUE", 3),
        ("CHECK", 47),
    ):
        checks.append(check(
            cursor,
            "schema",
            f"constraint_{constraint_type.lower().replace(' ', '_')}",
            (
                "SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS "
                f"WHERE CONSTRAINT_SCHEMA = '{database}' AND CONSTRAINT_TYPE = '{constraint_type}'"
            ),
            expected,
        ))

    for table, expected in EXPECTED_COUNTS.items():
        checks.append(check(cursor, "record_count", table, f"SELECT COUNT(*) FROM `{table}`", expected))

    checks.extend([
        check(
            cursor,
            "identity",
            "enterprise_id_range",
            "SELECT COUNT(*), COUNT(DISTINCT enterprise_id), MIN(enterprise_id), MAX(enterprise_id) FROM enterprise_basic",
            (40, 40, "SYN-E000001", "SYN-E000040"),
        ),
        check(
            cursor,
            "identity",
            "enterprise_id_sequence_hash",
            "SELECT SHA2(GROUP_CONCAT(enterprise_id ORDER BY enterprise_id SEPARATOR ','), 256) FROM enterprise_basic",
            enterprise_id_hash,
        ),
        check(
            cursor,
            "identity",
            "duplicate_enterprise_name",
            "SELECT COUNT(*) FROM (SELECT enterprise_name FROM enterprise_basic GROUP BY enterprise_name HAVING COUNT(*) > 1) q",
            0,
        ),
        check(
            cursor,
            "identity",
            "duplicate_credit_code",
            "SELECT COUNT(*) FROM (SELECT synthetic_credit_code FROM enterprise_basic GROUP BY synthetic_credit_code HAVING COUNT(*) > 1) q",
            0,
        ),
    ])

    orphan_queries = {
        "orphan_person": (
            "SELECT COUNT(*) FROM enterprise_person c LEFT JOIN enterprise_basic p "
            "ON p.enterprise_id=c.enterprise_id WHERE p.enterprise_id IS NULL"
        ),
        "orphan_profile": (
            "SELECT COUNT(*) FROM enterprise_operation_profile c LEFT JOIN enterprise_basic p "
            "ON p.enterprise_id=c.enterprise_id WHERE p.enterprise_id IS NULL"
        ),
        "orphan_monthly": (
            "SELECT COUNT(*) FROM enterprise_operation_monthly c LEFT JOIN enterprise_basic p "
            "ON p.enterprise_id=c.enterprise_id WHERE p.enterprise_id IS NULL"
        ),
        "orphan_operation_event": (
            "SELECT COUNT(*) FROM enterprise_operation_event c LEFT JOIN enterprise_basic p "
            "ON p.enterprise_id=c.enterprise_id WHERE p.enterprise_id IS NULL"
        ),
        "orphan_negative_event": (
            "SELECT COUNT(*) FROM enterprise_negative_event c LEFT JOIN enterprise_basic p "
            "ON p.enterprise_id=c.enterprise_id WHERE p.enterprise_id IS NULL"
        ),
        "orphan_related_person": (
            "SELECT COUNT(*) FROM enterprise_negative_event n LEFT JOIN enterprise_person p "
            "ON p.person_id=n.related_person_id AND p.enterprise_id=n.enterprise_id "
            "WHERE n.related_person_id IS NOT NULL AND p.person_id IS NULL"
        ),
    }
    for name, sql in orphan_queries.items():
        checks.append(check(cursor, "foreign_key", name, sql, 0))

    checks.extend([
        check(
            cursor,
            "person",
            "employment_before_establishment",
            "SELECT COUNT(*) FROM enterprise_person p JOIN enterprise_basic b USING(enterprise_id) WHERE p.start_date < b.established_date",
            0,
        ),
        check(
            cursor,
            "person",
            "invalid_employment_status",
            "SELECT COUNT(*) FROM enterprise_person WHERE (is_current=1 AND end_date IS NOT NULL) OR (is_current=0 AND end_date IS NULL)",
            0,
        ),
        check(
            cursor,
            "person",
            "missing_current_legal_representative",
            "SELECT COUNT(*) FROM enterprise_basic b WHERE b.registration_status='存续' AND NOT EXISTS "
            "(SELECT 1 FROM enterprise_person p WHERE p.enterprise_id=b.enterprise_id AND p.role='法定代表人' AND p.is_current=1)",
            0,
        ),
        check(
            cursor,
            "person",
            "missing_current_finance_lead",
            "SELECT COUNT(*) FROM enterprise_basic b WHERE b.registration_status='存续' AND NOT EXISTS "
            "(SELECT 1 FROM enterprise_person p WHERE p.enterprise_id=b.enterprise_id AND p.role='财务负责人' AND p.is_current=1)",
            0,
        ),
        check(
            cursor,
            "monthly",
            "companies_without_exact_12_months",
            "SELECT COUNT(*) FROM (SELECT enterprise_id FROM enterprise_operation_monthly "
            "GROUP BY enterprise_id HAVING COUNT(*)<>12 OR COUNT(DISTINCT stat_month)<>12) q",
            0,
        ),
        check(
            cursor,
            "monthly",
            "month_window_and_day_errors",
            "SELECT COUNT(*) FROM enterprise_operation_monthly WHERE stat_month<'2025-09-01' "
            "OR stat_month>'2026-08-01' OR DAY(stat_month)<>1",
            0,
        ),
        check(
            cursor,
            "monthly",
            "main_revenue_above_operating_revenue",
            "SELECT COUNT(*) FROM enterprise_operation_monthly WHERE main_business_revenue>operating_revenue",
            0,
        ),
        check(
            cursor,
            "monthly",
            "invoice_subitems_above_purchase_invoice",
            "SELECT COUNT(*) FROM enterprise_operation_monthly "
            "WHERE utility_invoice_amount+travel_hospitality_invoice_amount>invoice_purchase_amount",
            0,
        ),
        check(
            cursor,
            "monthly",
            "strict_monotonic_employee_curves",
            "WITH diffs AS (SELECT enterprise_id, CAST(employee_count AS SIGNED)-"
            "CAST(LAG(employee_count) OVER "
            "(PARTITION BY enterprise_id ORDER BY stat_month) AS SIGNED) AS d FROM enterprise_operation_monthly), "
            "flags AS (SELECT enterprise_id, SUM(d>0) ups, SUM(d<0) downs FROM diffs WHERE d IS NOT NULL GROUP BY enterprise_id) "
            "SELECT COUNT(*) FROM flags WHERE (ups>0 AND downs=0) OR (downs>0 AND ups=0)",
            0,
        ),
        check(
            cursor,
            "monthly",
            "february_revenue_minimum_companies",
            "SELECT COUNT(*) FROM enterprise_operation_monthly feb WHERE stat_month='2026-02-01' "
            "AND operating_revenue=(SELECT MIN(m.operating_revenue) FROM enterprise_operation_monthly m "
            "WHERE m.enterprise_id=feb.enterprise_id)",
            16,
        ),
        check(cursor, "monthly", "loss_month_count", "SELECT COUNT(*) FROM enterprise_operation_monthly WHERE net_profit<0", 34),
        check(cursor, "profile", "wechat_account_count", "SELECT SUM(has_wechat_account) FROM enterprise_operation_profile", Decimal("26")),
        check(
            cursor,
            "profile",
            "wechat_field_mismatch",
            "SELECT COUNT(*) FROM enterprise_operation_profile WHERE "
            "(has_wechat_account=0 AND wechat_content_summary IS NOT NULL) OR "
            "(has_wechat_account=1 AND wechat_content_summary IS NULL)",
            0,
        ),
        check(
            cursor,
            "basic",
            "missing_clean_contacts",
            "SELECT SUM(registered_email IS NULL), SUM(registered_phone IS NULL) FROM enterprise_basic",
            (Decimal("0"), Decimal("0")),
        ),
        check(
            cursor,
            "event",
            "completed_bond_count",
            "SELECT COUNT(*) FROM enterprise_operation_event WHERE event_type='债券发行' AND event_status='已完成'",
            4,
        ),
        check(
            cursor,
            "event",
            "operation_event_date_errors",
            "SELECT COUNT(*) FROM enterprise_operation_event WHERE event_date<'2025-09-01' OR event_date>'2026-08-31'",
            0,
        ),
        check(
            cursor,
            "event",
            "negative_event_date_errors",
            "SELECT COUNT(*) FROM enterprise_negative_event WHERE event_date<'2016-01-01' OR event_date>'2026-08-31'",
            0,
        ),
        check(
            cursor,
            "event",
            "executive_case_without_person",
            "SELECT COUNT(*) FROM enterprise_negative_event WHERE event_type='高管涉案' AND related_person_id IS NULL",
            0,
        ),
    ])

    checks.extend([
        check(
            cursor,
            "amount_total",
            "registered_capital",
            "SELECT SUM(registered_capital) FROM enterprise_basic",
            Decimal("6448100000.00"),
        ),
        check(
            cursor,
            "amount_total",
            "assets_and_liabilities",
            "SELECT SUM(total_assets), SUM(total_liabilities) FROM enterprise_operation_profile",
            (Decimal("70392213416.84"), Decimal("33511997831.28")),
        ),
        check(
            cursor,
            "amount_total",
            "monthly_amounts",
            "SELECT SUM(social_security_amount), SUM(tax_amount), SUM(invoice_purchase_amount), "
            "SUM(invoice_sales_amount), SUM(utility_invoice_amount), SUM(travel_hospitality_invoice_amount), "
            "SUM(operating_revenue), SUM(main_business_revenue), SUM(total_profit), SUM(net_profit) "
            "FROM enterprise_operation_monthly",
            (
                Decimal("863986040.00"), Decimal("1972271800.00"),
                Decimal("29197542200.00"), Decimal("58907427000.00"),
                Decimal("1264338800.00"), Decimal("1214841160.00"),
                Decimal("63939908700.00"), Decimal("61573158500.00"),
                Decimal("4887492500.00"), Decimal("3689203765.00"),
            ),
        ),
        check(
            cursor,
            "amount_total",
            "operation_event_amounts",
            "SELECT SUM(COALESCE(amount,0)), SUM(COALESCE(valuation,0)) FROM enterprise_operation_event",
            (Decimal("16466800000.00"), Decimal("21654000000.00")),
        ),
    ])
    return checks


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"Baseline: {report['baseline_id']}",
        f"Database: {report['database']}",
        f"MySQL: {report['mysql_version']}",
        f"Checked at: {report['checked_at']}",
        f"Result: {report['result']}",
        f"Hard error count: {report['hard_error_count']}",
        "",
        "STATUS\tCATEGORY\tCHECK\tACTUAL\tEXPECTED",
    ]
    for item in report["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        actual = json.dumps(item["actual"], ensure_ascii=False)
        expected = json.dumps(item["expected"], ensure_ascii=False)
        lines.append(f"{status}\t{item['category']}\t{item['name']}\t{actual}\t{expected}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行虚拟企业V2.1基线SQL验收")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="synthetic")
    parser.add_argument("--password-env", default="ENTERPRISE_DB_PASSWORD")
    parser.add_argument("--database", default="synthetic_enterprise_v21")
    parser.add_argument("--baseline-id", default="synthetic_enterprise_v21_40_v1.0.0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"未设置环境变量{args.password_env}")
    try:
        import mysql.connector
    except ModuleNotFoundError as error:
        raise RuntimeError("请先安装mysql-connector-python") from error

    import hashlib

    expected_ids = ",".join(f"SYN-E{index:06d}" for index in range(1, 41))
    enterprise_id_hash = hashlib.sha256(expected_ids.encode("utf-8")).hexdigest()
    connection = mysql.connector.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=password,
        database=args.database,
        charset="utf8mb4",
        use_unicode=True,
        autocommit=False,
    )
    try:
        connection.start_transaction(consistent_snapshot=True, readonly=True)
        cursor = connection.cursor()
        try:
            mysql_version = str(fetch_one(cursor, "SELECT VERSION()"))
            checks = build_checks(cursor, args.database, enterprise_id_hash)
        finally:
            cursor.close()
        connection.rollback()
    finally:
        connection.close()

    hard_error_count = sum(not item["passed"] for item in checks)
    report = {
        "baseline_id": args.baseline_id,
        "database": args.database,
        "mysql_version": mysql_version,
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "result": "PASS" if hard_error_count == 0 else "FAIL",
        "hard_error_count": hard_error_count,
        "check_count": len(checks),
        "checks": checks,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "acceptance_report.json"
    text_path = output_dir / "acceptance_report.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text_path.write_text(render_text(report), encoding="utf-8")
    print(render_text(report), end="")
    print(f"JSON report: {json_path}")
    print(f"Text report: {text_path}")
    return 0 if hard_error_count == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"验收失败：{error}", file=sys.stderr)
        raise SystemExit(2)
