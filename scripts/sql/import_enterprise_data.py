#!/usr/bin/env python3
"""Import baseline or append-only enterprise JSON data into MySQL 8.0 safely.

JSON decimal literals are parsed directly as Decimal. Monetary values never
pass through Python binary float before being sent to MySQL DECIMAL(18,2).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = (
    ROOT / 'data/sql/synthetic_enterprise_v21_40_v1.0.0/六表数据_01-40_修正.json'
)

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "enterprise_basic": (
        "enterprise_id",
        "enterprise_name",
        "synthetic_credit_code",
        "enterprise_type",
        "registration_status",
        "industry",
        "registered_capital",
        "established_date",
        "registered_address",
        "registered_email",
        "registered_phone",
        "registered_business_scope",
        "main_business_description",
    ),
    "enterprise_person": (
        "person_id",
        "enterprise_id",
        "person_name",
        "age",
        "household_region",
        "role",
        "start_date",
        "end_date",
        "is_current",
    ),
    "enterprise_operation_profile": (
        "enterprise_id",
        "as_of_date",
        "has_wechat_account",
        "wechat_content_summary",
        "total_assets",
        "total_liabilities",
    ),
    "enterprise_operation_monthly": (
        "enterprise_id",
        "stat_month",
        "employee_count",
        "social_security_count",
        "social_security_amount",
        "tax_amount",
        "invoice_purchase_amount",
        "invoice_sales_amount",
        "utility_invoice_amount",
        "travel_hospitality_invoice_amount",
        "operating_revenue",
        "main_business_revenue",
        "total_profit",
        "net_profit",
    ),
    "enterprise_operation_event": (
        "operation_event_id",
        "enterprise_id",
        "event_date",
        "event_type",
        "amount",
        "valuation",
        "financing_stage",
        "term_months",
        "quantity",
        "event_status",
        "description",
    ),
    "enterprise_negative_event": (
        "negative_event_id",
        "enterprise_id",
        "related_person_id",
        "event_date",
        "event_type",
        "case_category",
        "description",
        "handling_result",
    ),
}

INSERT_ORDER = (
    "enterprise_basic",
    "enterprise_person",
    "enterprise_operation_profile",
    "enterprise_operation_monthly",
    "enterprise_operation_event",
    "enterprise_negative_event",
)

DELETE_ORDER = tuple(reversed(INSERT_ORDER))

BASELINE_EXPECTED_COUNTS = {
    "enterprise_basic": 40,
    "enterprise_person": 234,
    "enterprise_operation_profile": 40,
    "enterprise_operation_monthly": 480,
    "enterprise_operation_event": 72,
    "enterprise_negative_event": 34,
}

EXPECTED_MONTHS = {
    f"{year}-{month:02d}-01"
    for year, month in (
        (2025, 9), (2025, 10), (2025, 11), (2025, 12),
        (2026, 1), (2026, 2), (2026, 3), (2026, 4),
        (2026, 5), (2026, 6), (2026, 7), (2026, 8),
    )
}

MONEY_FIELDS = {
    "enterprise_basic": {"registered_capital"},
    "enterprise_operation_profile": {"total_assets", "total_liabilities"},
    "enterprise_operation_monthly": {
        "social_security_amount",
        "tax_amount",
        "invoice_purchase_amount",
        "invoice_sales_amount",
        "utility_invoice_amount",
        "travel_hospitality_invoice_amount",
        "operating_revenue",
        "main_business_revenue",
        "total_profit",
        "net_profit",
    },
    "enterprise_operation_event": {"amount", "valuation"},
}

PRIMARY_KEY_FIELDS = {
    "enterprise_basic": ("enterprise_id",),
    "enterprise_person": ("person_id",),
    "enterprise_operation_profile": ("enterprise_id",),
    "enterprise_operation_monthly": ("enterprise_id", "stat_month"),
    "enterprise_operation_event": ("operation_event_id",),
    "enterprise_negative_event": ("negative_event_id",),
}


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON存在重复键：{key}")
        result[key] = value
    return result


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(
            file,
            parse_float=Decimal,
            parse_int=int,
            object_pairs_hook=reject_duplicate_keys,
        )
    if not isinstance(data, list):
        raise ValueError("JSON顶层必须是企业数组")
    return data


def validate_company_blocks(companies: list[dict[str, Any]], mode: str) -> None:
    if not companies:
        raise ValueError("JSON至少必须包含1家企业")

    expected_tables = set(INSERT_ORDER)
    enterprise_ids: list[str] = []
    enterprise_names: list[str] = []
    credit_codes: list[str] = []
    for company_index, company in enumerate(companies):
        if set(company) != expected_tables:
            missing = sorted(expected_tables - set(company))
            extra = sorted(set(company) - expected_tables)
            raise ValueError(
                f"第{company_index + 1}家企业六表键不完整："
                f"missing={missing}, extra={extra}"
            )
        for table in INSERT_ORDER:
            if not isinstance(company[table], list):
                raise ValueError(f"第{company_index + 1}家企业/{table}必须是数组")
        if len(company["enterprise_basic"]) != 1:
            raise ValueError(f"第{company_index + 1}家企业必须有且1条工商记录")
        if len(company["enterprise_operation_profile"]) != 1:
            raise ValueError(f"第{company_index + 1}家企业必须有且1条运营概况记录")

        basic = company["enterprise_basic"][0]
        enterprise_id = basic.get("enterprise_id")
        if not isinstance(enterprise_id, str) or not re.fullmatch(r"SYN-E\d{6}", enterprise_id):
            raise ValueError(f"第{company_index + 1}家企业ID格式错误：{enterprise_id}")
        enterprise_number = int(enterprise_id.removeprefix("SYN-E"))
        enterprise_ids.append(enterprise_id)
        enterprise_names.append(basic.get("enterprise_name"))
        credit_codes.append(basic.get("synthetic_credit_code"))

        people = company["enterprise_person"]
        if not 3 <= len(people) <= 8:
            raise ValueError(f"{enterprise_id}核心人员必须为3至8条")
        local_person_ids = {person.get("person_id") for person in people}
        if not any(person.get("role") == "法定代表人" and person.get("is_current") == 1 for person in people):
            raise ValueError(f"{enterprise_id}缺少在任法定代表人")
        if not any(person.get("role") == "财务负责人" and person.get("is_current") == 1 for person in people):
            raise ValueError(f"{enterprise_id}缺少在任财务负责人")

        if len(company["enterprise_operation_monthly"]) != 12:
            raise ValueError(f"{enterprise_id}必须包含12条月度记录")
        months = {row.get("stat_month") for row in company["enterprise_operation_monthly"]}
        if months != EXPECTED_MONTHS:
            raise ValueError(f"{enterprise_id}月份集合不是2025-09至2026-08的12个月")

        for table in INSERT_ORDER[1:]:
            for row_index, row in enumerate(company[table]):
                if row.get("enterprise_id") != enterprise_id:
                    raise ValueError(
                        f"{enterprise_id}/{table}[{row_index}]企业外键不一致"
                    )
        for event in company["enterprise_negative_event"]:
            related_person_id = event.get("related_person_id")
            if related_person_id is not None and related_person_id not in local_person_ids:
                raise ValueError(
                    f"{event.get('negative_event_id')}引用了非本企业人员：{related_person_id}"
                )

        if mode == "append":
            person_prefix = f"SYN-P{enterprise_number:04d}"
            operation_prefix = f"SYN-OE{enterprise_number:06d}"
            negative_prefix = f"SYN-NE{enterprise_number:06d}"
            if any(not re.fullmatch(re.escape(person_prefix) + r"\d{2}", str(person_id)) for person_id in local_person_ids):
                raise ValueError(f"{enterprise_id}人员ID应使用前缀{person_prefix}")
            if any(
                not re.fullmatch(re.escape(operation_prefix) + r"\d{2}", str(event.get("operation_event_id")))
                for event in company["enterprise_operation_event"]
            ):
                raise ValueError(f"{enterprise_id}运营事件ID应使用前缀{operation_prefix}")
            if any(
                not re.fullmatch(re.escape(negative_prefix) + r"\d{2}", str(event.get("negative_event_id")))
                for event in company["enterprise_negative_event"]
            ):
                raise ValueError(f"{enterprise_id}负面事件ID应使用前缀{negative_prefix}")

    for label, values in (
        ("企业ID", enterprise_ids),
        ("企业名称", enterprise_names),
        ("虚拟信用代码", credit_codes),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"新增JSON内部存在重复{label}")

    if mode == "append":
        numbers = [int(enterprise_id.removeprefix("SYN-E")) for enterprise_id in enterprise_ids]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError("append文件中的企业ID必须升序且连续")


def flatten_dataset(companies: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {table: [] for table in INSERT_ORDER}
    expected_tables = set(INSERT_ORDER)
    for company_index, company in enumerate(companies):
        if set(company) != expected_tables:
            missing = sorted(expected_tables - set(company))
            extra = sorted(set(company) - expected_tables)
            raise ValueError(
                f"第{company_index + 1}家企业六表键不完整："
                f"missing={missing}, extra={extra}"
            )
        for table in INSERT_ORDER:
            if not isinstance(company[table], list):
                raise ValueError(f"{table}必须是数组")
            rows[table].extend(company[table])
    return rows


def as_decimal_18_2(value: Any, label: str) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, float):
        raise TypeError(f"{label}意外变成了float，拒绝导入")
    decimal_value = value if isinstance(value, Decimal) else Decimal(value)
    if decimal_value.as_tuple().exponent < -2:
        raise ValueError(f"{label}超过DECIMAL(18,2)的两位小数精度：{decimal_value}")
    integer_digits = max(
        len(decimal_value.as_tuple().digits) + decimal_value.as_tuple().exponent,
        0,
    )
    if integer_digits > 16:
        raise ValueError(f"{label}超过DECIMAL(18,2)的有效数字范围：{decimal_value}")
    return decimal_value.quantize(Decimal("0.01"))


def validate_and_convert_rows(
    rows: dict[str, list[dict[str, Any]]],
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    actual_counts = {table: len(table_rows) for table, table_rows in rows.items()}
    if mode != "append" and actual_counts != BASELINE_EXPECTED_COUNTS:
        raise ValueError(
            "记录数与40家修正版不一致："
            + json.dumps(actual_counts, ensure_ascii=False)
        )

    if mode == "append":
        enterprise_count = actual_counts["enterprise_basic"]
        expected_relational_counts = {
            "enterprise_operation_profile": enterprise_count,
            "enterprise_operation_monthly": enterprise_count * 12,
        }
        for table, expected_count in expected_relational_counts.items():
            if actual_counts[table] != expected_count:
                raise ValueError(
                    f"append文件的{table}记录数错误："
                    f"expected={expected_count}, actual={actual_counts[table]}"
                )

    enterprise_ids = [row["enterprise_id"] for row in rows["enterprise_basic"]]
    expected_ids = [f"SYN-E{index:06d}" for index in range(1, 41)]
    if mode != "append" and enterprise_ids != expected_ids:
        raise ValueError("企业ID必须为SYN-E000001至SYN-E000040且按顺序排列")

    for table, table_rows in rows.items():
        expected_fields = set(TABLE_COLUMNS[table])
        seen_keys: set[tuple[Any, ...]] = set()
        for row_index, row in enumerate(table_rows):
            if set(row) != expected_fields:
                missing = sorted(expected_fields - set(row))
                extra = sorted(set(row) - expected_fields)
                raise ValueError(
                    f"{table}[{row_index}]字段不完整：missing={missing}, extra={extra}"
                )
            key = tuple(row[field] for field in PRIMARY_KEY_FIELDS[table])
            if key in seen_keys:
                raise ValueError(f"{table}存在重复主键：{key}")
            seen_keys.add(key)
            for field in MONEY_FIELDS.get(table, set()):
                row[field] = as_decimal_18_2(row[field], f"{table}[{row_index}].{field}")
    return rows


def insert_sql(table: str) -> str:
    columns = TABLE_COLUMNS[table]
    column_sql = ", ".join(f"`{column}`" for column in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    return f"INSERT INTO `{table}` ({column_sql}) VALUES ({placeholders})"


def row_values(table: str, rows: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
    columns = TABLE_COLUMNS[table]
    return [tuple(row[column] for column in columns) for row in rows]


def database_counts(connection: Any) -> dict[str, int]:
    cursor = connection.cursor()
    try:
        result: dict[str, int] = {}
        for table in INSERT_ORDER:
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            result[table] = int(cursor.fetchone()[0])
        return result
    finally:
        cursor.close()


def existing_scalar_values(
    connection: Any,
    table: str,
    column: str,
    values: list[Any],
) -> set[Any]:
    if not values:
        return set()
    placeholders = ", ".join(["%s"] * len(values))
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"SELECT `{column}` FROM `{table}` WHERE `{column}` IN ({placeholders})",
            tuple(values),
        )
        return {row[0] for row in cursor.fetchall()}
    finally:
        cursor.close()


def validate_append_against_database(
    connection: Any,
    rows: dict[str, list[dict[str, Any]]],
) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT MAX(enterprise_id) FROM enterprise_basic")
        current_max_id = cursor.fetchone()[0]
    finally:
        cursor.close()

    first_number = 1 if current_max_id is None else int(current_max_id.removeprefix("SYN-E")) + 1
    expected_enterprise_ids = [
        f"SYN-E{number:06d}"
        for number in range(first_number, first_number + len(rows["enterprise_basic"]))
    ]
    actual_enterprise_ids = [row["enterprise_id"] for row in rows["enterprise_basic"]]
    if actual_enterprise_ids != expected_enterprise_ids:
        raise RuntimeError(
            "append企业ID未紧接数据库当前最大编号："
            f"expected={expected_enterprise_ids}, actual={actual_enterprise_ids}"
        )

    collision_checks = (
        ("enterprise_basic", "enterprise_id", [row["enterprise_id"] for row in rows["enterprise_basic"]]),
        ("enterprise_basic", "enterprise_name", [row["enterprise_name"] for row in rows["enterprise_basic"]]),
        ("enterprise_basic", "synthetic_credit_code", [row["synthetic_credit_code"] for row in rows["enterprise_basic"]]),
        ("enterprise_person", "person_id", [row["person_id"] for row in rows["enterprise_person"]]),
        (
            "enterprise_operation_event",
            "operation_event_id",
            [row["operation_event_id"] for row in rows["enterprise_operation_event"]],
        ),
        (
            "enterprise_negative_event",
            "negative_event_id",
            [row["negative_event_id"] for row in rows["enterprise_negative_event"]],
        ),
    )
    collisions: dict[str, list[Any]] = {}
    for table, column, values in collision_checks:
        existing = existing_scalar_values(connection, table, column, values)
        if existing:
            collisions[f"{table}.{column}"] = sorted(existing)
    if collisions:
        raise RuntimeError(
            "append数据与数据库已有唯一键冲突："
            + json.dumps(collisions, ensure_ascii=False)
        )


def clear_existing_rows(connection: Any) -> None:
    cursor = connection.cursor()
    try:
        for table in DELETE_ORDER:
            cursor.execute(f"DELETE FROM `{table}`")
    finally:
        cursor.close()


def insert_rows(
    connection: Any,
    rows: dict[str, list[dict[str, Any]]],
) -> None:
    cursor = connection.cursor()
    try:
        for table in INSERT_ORDER:
            if not rows[table]:
                continue
            cursor.executemany(insert_sql(table), row_values(table, rows[table]))
            if cursor.rowcount != len(rows[table]):
                raise RuntimeError(
                    f"{table}写入数与预期不一致："
                    f"expected={len(rows[table])}, actual={cursor.rowcount}"
                )
    finally:
        cursor.close()


def expected_decimal_values(
    rows: dict[str, list[dict[str, Any]]],
    table: str,
) -> dict[tuple[Any, ...], tuple[Decimal | None, ...]]:
    key_fields = PRIMARY_KEY_FIELDS[table]
    money_fields = tuple(field for field in TABLE_COLUMNS[table] if field in MONEY_FIELDS.get(table, set()))
    return {
        tuple(row[field] for field in key_fields): tuple(row[field] for field in money_fields)
        for row in rows[table]
    }


def database_decimal_values(
    connection: Any,
    table: str,
) -> dict[tuple[Any, ...], tuple[Decimal | None, ...]]:
    key_fields = PRIMARY_KEY_FIELDS[table]
    money_fields = tuple(field for field in TABLE_COLUMNS[table] if field in MONEY_FIELDS.get(table, set()))
    selected_fields = key_fields + money_fields
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT "
            + ", ".join(f"`{field}`" for field in selected_fields)
            + f" FROM `{table}`"
        )
        result: dict[tuple[Any, ...], tuple[Decimal | None, ...]] = {}
        for database_row in cursor.fetchall():
            key_parts = database_row[: len(key_fields)]
            normalized_key = tuple(
                value.isoformat() if hasattr(value, "isoformat") else value
                for value in key_parts
            )
            result[normalized_key] = tuple(database_row[len(key_fields) :])
        return result
    finally:
        cursor.close()


def verify_import(
    connection: Any,
    rows: dict[str, list[dict[str, Any]]],
    expected_database_counts: dict[str, int],
) -> dict[str, Any]:
    counts = database_counts(connection)
    if counts != expected_database_counts:
        raise RuntimeError(
            "导入后记录数验证失败："
            + json.dumps(counts, ensure_ascii=False)
        )

    verified_money_rows: dict[str, int] = {}
    for table in MONEY_FIELDS:
        expected = expected_decimal_values(rows, table)
        all_database_values = database_decimal_values(connection, table)
        actual = {
            key: all_database_values.get(key)
            for key in expected
            if key in all_database_values
        }
        if actual != expected:
            differing_keys = sorted(set(actual) | set(expected))
            differing_keys = [key for key in differing_keys if actual.get(key) != expected.get(key)]
            raise RuntimeError(
                f"{table}金额精度验证失败，差异主键示例：{differing_keys[:5]}"
            )
        verified_money_rows[table] = len(expected)

    return {
        "record_counts": counts,
        "decimal_exact_match": True,
        "verified_money_rows": verified_money_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将虚拟企业基准或增量JSON导入MySQL V2.1六表"
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="待导入JSON路径")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="synthetic")
    parser.add_argument("--password-env", default="ENTERPRISE_DB_PASSWORD")
    parser.add_argument("--database", default="synthetic_enterprise_v21")
    parser.add_argument(
        "--mode",
        choices=("empty-only", "replace", "append"),
        default="empty-only",
        help=(
            "empty-only仅允许空表导入；"
            "replace先删除六表旧数据；"
            "append仅追加紧接当前最大编号的新企业"
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只校验JSON结构和DECIMAL精度，不连接数据库",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path = args.json.expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到JSON文件：{json_path}")

    companies = load_json(json_path)
    validate_company_blocks(companies, args.mode)
    rows = validate_and_convert_rows(flatten_dataset(companies), args.mode)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validation_passed",
                    "json": str(json_path),
                    "record_counts": {table: len(value) for table, value in rows.items()},
                    "decimal_transport": "JSON Decimal -> Python Decimal -> MySQL DECIMAL(18,2)",
                    "database_collision_check": "pending" if args.mode == "append" else "not_applicable",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(
            f"未设置环境变量{args.password_env}，拒绝在命令行中传递明文密码"
        )

    try:
        import mysql.connector
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "未安装mysql-connector-python，请先执行："
            "python -m pip install mysql-connector-python"
        ) from error

    connection: Any | None = None
    try:
        connection = mysql.connector.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=password,
            database=args.database,
            autocommit=False,
            charset="utf8mb4",
            use_unicode=True,
        )
        before_counts = database_counts(connection)
        if args.mode == "empty-only" and any(before_counts.values()):
            raise RuntimeError(
                "数据表非空，empty-only模式拒绝导入："
                + json.dumps(before_counts, ensure_ascii=False)
            )
        if args.mode == "replace":
            clear_existing_rows(connection)
            expected_database_counts = {
                table: len(table_rows) for table, table_rows in rows.items()
            }
        elif args.mode == "append":
            validate_append_against_database(connection, rows)
            expected_database_counts = {
                table: before_counts[table] + len(rows[table])
                for table in INSERT_ORDER
            }
        else:
            expected_database_counts = {
                table: len(table_rows) for table, table_rows in rows.items()
            }

        insert_rows(connection, rows)
        verification = verify_import(connection, rows, expected_database_counts)
        connection.commit()
        print(
            json.dumps(
                {
                    "status": "import_committed",
                    "json": str(json_path),
                    "database": args.database,
                    "mode": args.mode,
                    "before_counts": before_counts,
                    "added_counts": {table: len(value) for table, value in rows.items()},
                    **verification,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception:
        if connection is not None and connection.is_connected():
            connection.rollback()
        raise
    finally:
        if connection is not None and connection.is_connected():
            connection.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"导入失败：{error}", file=sys.stderr)
        raise SystemExit(1)
