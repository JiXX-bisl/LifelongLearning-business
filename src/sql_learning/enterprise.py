"""Independent SQLite smoke bundle from the frozen enterprise table export."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from .comparison import COMPARISON_VERSION

from src.enterprise_dataset_v30.schema import (
    COLUMN_TYPES, FIELD_MEANINGS, PRIMARY_KEYS, PUBLIC_RULES, TABLE_DESCRIPTIONS,
)


# Hand-authored interface smoke cases, separate from the original scoring set.
_SMOKE_CASES = (
    ("enterprise-smoke-detail", "查询锐序精工制造有限公司的企业名称、行业和成立日期。",
     "SELECT enterprise_name, industry, established_date FROM enterprise_basic "
     "WHERE enterprise_name = '锐序精工制造有限公司'", "detail"),
    ("enterprise-smoke-aggregate", "按行业统计企业数量，返回行业和数量，按行业升序排列。",
     "SELECT industry, COUNT(*) AS enterprise_count FROM enterprise_basic "
     "GROUP BY industry ORDER BY industry", "aggregate"),
    ("enterprise-smoke-compare", "比较启衡机电制造有限公司与锐序精工制造有限公司的注册资本，返回两家公司注册资本及前者减后者的差额，金额单位为元。",
     "SELECT CAST(a.registered_capital AS NUMERIC) AS first_capital, "
     "CAST(b.registered_capital AS NUMERIC) AS second_capital, "
     "CAST(a.registered_capital AS NUMERIC) - CAST(b.registered_capital AS NUMERIC) AS capital_difference "
     "FROM enterprise_basic a CROSS JOIN enterprise_basic b "
     "WHERE a.enterprise_name = '启衡机电制造有限公司' "
     "AND b.enterprise_name = '锐序精工制造有限公司'", "compare"),
    ("enterprise-smoke-rank", "查询注册资本最高的五家企业，返回企业名称和注册资本（元）；按注册资本降序，同额按企业名称升序。",
     "SELECT enterprise_name, CAST(registered_capital AS NUMERIC) AS registered_capital "
     "FROM enterprise_basic ORDER BY CAST(registered_capital AS NUMERIC) DESC, enterprise_name ASC LIMIT 5", "rank"),
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identifier(name):
    return '"' + name.replace('"', '""') + '"'


def prepare_enterprise(tables_path, output_dir, *, namespace="enterprise_v2_smoke"):
    """Preserve source values and keep all generated files outside the old baseline.

    This SQLite mirror is for interface verification, not an assertion of MySQL
    dialect or numeric equivalence for the original benchmark.
    """
    source = Path(tables_path).resolve()
    tables = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(tables, dict) or not tables:
        raise ValueError("tables export must be a nonempty object")
    if set(tables) != set(COLUMN_TYPES):
        raise ValueError("source is not the frozen enterprise v3 table export")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    database = output / "database.sqlite3"
    schema = {}
    counts = {}
    with closing(sqlite3.connect(database)) as connection, connection:
        for table, expected_types in COLUMN_TYPES.items():
            rows = tables[table]
            columns = list(expected_types)
            if (not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != set(columns)
                                                  for row in rows)):
                raise ValueError("source table fields differ: " + table)
            # DECIMAL/date/text remain strings, retaining exact frozen input values.
            types = {name: "INTEGER" if "INT" in raw_type.upper() else "TEXT"
                     for name, raw_type in expected_types.items()}
            definition = ", ".join(f"{_identifier(name)} {types[name]}" for name in columns)
            primary = PRIMARY_KEYS[table]
            ddl = f"CREATE TABLE {_identifier(table)} ({definition}, PRIMARY KEY (" + ", ".join(
                _identifier(name) for name in primary) + "))"
            connection.execute(ddl)
            connection.executemany(f"INSERT INTO {_identifier(table)} VALUES (" + ",".join(
                "?" for _ in columns) + ")", [[row[name] for name in columns] for row in rows])
            restored = connection.execute(f"SELECT * FROM {_identifier(table)} ORDER BY rowid").fetchall()
            if restored != [tuple(row[name] for name in columns) for row in rows]:
                raise ValueError("SQLite import changed frozen values: " + table)
            counts[table] = len(rows)
            schema[table] = (f"{table} — {TABLE_DESCRIPTIONS[table]}\n{ddl};\n" + "\n".join(
                f"{name}: {FIELD_MEANINGS.get(name, name)}" for name in columns))
        private = {}
        private_sql = {}
        groups = {}
        cases = []
        for case_id, question, sql, group in _SMOKE_CASES:
            cursor = connection.execute(sql)
            private[case_id] = {"columns": [column[0] for column in cursor.description],
                                "rows": [list(row) for row in cursor.fetchall()]}
            private_sql[case_id] = sql
            groups[case_id] = group
            cases.append({"id": case_id, "question": question})
    public = {
        "namespace": namespace,
        "data_version": source.parent.name + "-sqlite-smoke-v2",
        "schema": schema,
        "default_rules": "SQLite 数据库。企业外键 enterprise_id 关联 enterprise_basic.enterprise_id。\n"
                         "为精确保留源数据，金额以十进制字符串 TEXT 存储；金额计算及数值排序需要转换为数值。\n"
                         "这是独立的接口冒烟测试数据镜像。\n" + "\n".join(PUBLIC_RULES),
        "cases": cases,
    }
    _write_json(output / "public.json", public)
    _write_json(output / "private.json", private)
    _write_json(output / "private_sql.json", private_sql)
    _write_json(output / "comparison_policy.json", {"ordered": True})
    _write_json(output / "provenance.json", {
        "kind": "enterprise_interface_smoke", "full_benchmark": False,
        "source_tables": str(source), "source_sha256": _sha(source),
        "data_version": public["data_version"], "row_counts": counts,
        "sqlite_value_roundtrip_verified": True,
        "score_protocol": COMPARISON_VERSION + "; ordered=True; required-column projection; duplicate-sensitive",
        "group_labels": groups,
        "artifacts_sha256": {name: _sha(output / name) for name in ("database.sqlite3", "public.json", "private.json", "private_sql.json", "comparison_policy.json")},
    })
    return output
