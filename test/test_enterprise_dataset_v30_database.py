"""Offline regressions for MySQL CHECK metadata and semantic validation."""

import json
import re

import pytest

from src.enterprise_dataset_v30 import database


PROFILE = "enterprise_operation_profile"
MONTHLY = "enterprise_operation_monthly"
UNIT = "chk_v30_operation_profile_amount_unit"
STATUS = "chk_v30_operation_profile_report_status"
COUNTS = "chk_v30_monthly_counts"
DATES = "chk_v30_monthly_dates"

EXPECTED_EXPRESSIONS = {
    (PROFILE, UNIT): "`amount_unit` IN ('元', '万元')",
    (PROFILE, STATUS): "`report_status` IN ('正式', '撤回')",
    (MONTHLY, COUNTS): (
        "(`employee_count` IS NULL OR `employee_count` >= 0) AND "
        "(`social_security_count` IS NULL OR `social_security_count` >= 0)"
    ),
    (MONTHLY, DATES): "DAY(`stat_month`) = 1",
}

SHOW_EXPRESSIONS = {
    (PROFILE, UNIT): "(`amount_unit` in (_utf8mb4'元',_utf8mb4'万元'))",
    (PROFILE, STATUS): "(`report_status` in (_utf8mb4'正式',_utf8mb4'撤回'))",
    (MONTHLY, COUNTS): (
        "(((`employee_count` is null) or (`employee_count` >= 0)) and "
        "((`social_security_count` is null) or (`social_security_count` >= 0)))"
    ),
    (MONTHLY, DATES): "((dayofmonth(`stat_month`) = 1))",
}

# Observed in SHOW CREATE TABLE on native Windows MySQL 8.0.16.
MYSQL_8016_COUNTS_EXPRESSION = (
    "((isnull(`employee_count`) or (`employee_count` >= 0)) and "
    "(isnull(`social_security_count`) or (`social_security_count` >= 0)))"
)


def make_statement(table, expressions):
    constraints = [
        f"  CONSTRAINT `{name}` CHECK ({expression})"
        for (owner, name), expression in expressions.items()
        if owner == table
    ]
    return (
        f"CREATE TABLE `{table}` (\n"
        "  `fixture_column` int DEFAULT NULL,\n"
        + ",\n".join(constraints)
        + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    )


def expected_checks():
    return {
        identity: database._normalize_check(expression)
        for identity, expression in EXPECTED_EXPRESSIONS.items()
    }


def mock_check_sources(monkeypatch, *, metadata=None, statements=None):
    connection = object()
    calls = []
    if metadata is None:
        metadata = [(table, name, "YES") for table, name in EXPECTED_EXPRESSIONS]
    if statements is None:
        statements = {
            table: make_statement(table, SHOW_EXPRESSIONS)
            for table in (PROFILE, MONTHLY)
        }

    def fetch_all(actual_connection, sql, parameters=()):
        assert actual_connection is connection
        calls.append((sql, parameters))
        # MySQL 8.0.46 can expose escaped/mojibake CHECK_CLAUSE text while
        # SHOW CREATE TABLE contains the correct expression. Reject use of
        # that source even if a future decoder happens to accept this fixture.
        assert "CHECK_CLAUSE" not in sql.upper()
        assert "INFORMATION_SCHEMA.CHECK_CONSTRAINTS" not in sql.upper()
        if "INFORMATION_SCHEMA.TABLE_CONSTRAINTS" in sql.upper():
            assert parameters == (database.TARGET_DATABASE,)
            return list(metadata)
        match = re.fullmatch(r"SHOW CREATE TABLE `([^`]+)`;?", sql.strip(), re.I)
        assert match is not None, f"Unexpected validation query: {sql}"
        table = match.group(1)
        return [(table, statements[table])]

    monkeypatch.setattr(database, "_fetch_all", fetch_all)
    return connection, calls


@pytest.mark.parametrize(
    ("expression", "canonical"),
    [
        (
            "(`amount_unit` in (_utf8mb4'元',_utf8mb4'万元'))",
            ["in", ["identifier", "amount_unit"], [["literal", "元"], ["literal", "万元"]]],
        ),
        (
            "((dayofmonth(`stat_month`) = 1))",
            ["compare", "=", ["function", "day", ["identifier", "stat_month"]], ["number", "1"]],
        ),
        (
            "((`is_current` is null) or (`is_current` in (0,1)))",
            ["or", [["is_null", ["identifier", "is_current"]], ["in", ["identifier", "is_current"], [["number", "0"], ["number", "1"]]]]],
        ),
        (
            "ISNULL((`employee_count`))",
            ["is_null", ["identifier", "employee_count"]],
        ),
    ],
)
def test_check_normalization_preserves_expected_semantics(expression, canonical):
    assert json.loads(database._normalize_check(expression)) == canonical


@pytest.mark.parametrize(
    "expression",
    ["isnull()", "isnull(`employee_count`, 0)", "isnull(`employee_count`"],
)
def test_check_normalization_rejects_malformed_isnull(expression):
    with pytest.raises(ValueError):
        database._normalize_check(expression)


def test_check_definitions_extract_all_nested_and_chinese_checks():
    for table in (PROFILE, MONTHLY):
        expected = {name: expression for (owner, name), expression in expected_checks().items() if owner == table}
        assert database._check_definitions(make_statement(table, SHOW_EXPRESSIONS)) == expected


def test_check_contract_uses_show_create_instead_of_corrupt_metadata(monkeypatch):
    connection, calls = mock_check_sources(monkeypatch)
    database._check_contract(connection, {PROFILE, MONTHLY}, expected_checks())
    show_tables = {
        re.fullmatch(r"SHOW CREATE TABLE `([^`]+)`;?", sql.strip(), re.I).group(1)
        for sql, _ in calls
        if sql.upper().startswith("SHOW CREATE TABLE")
    }
    assert show_tables == {PROFILE, MONTHLY}


def test_check_contract_accepts_mysql_8016_isnull_rendering(monkeypatch):
    expressions = {**SHOW_EXPRESSIONS, (MONTHLY, COUNTS): MYSQL_8016_COUNTS_EXPRESSION}
    statements = {table: make_statement(table, expressions) for table in (PROFILE, MONTHLY)}
    connection, _ = mock_check_sources(monkeypatch, statements=statements)
    database._check_contract(connection, {PROFILE, MONTHLY}, expected_checks())


@pytest.mark.parametrize(
    ("identity", "changed_expression"),
    [
        ((PROFILE, UNIT), "`amount_unit` IN (_utf8mb4'元', _utf8mb4'千元')"),
        ((PROFILE, STATUS), "`report_status` IN (_utf8mb4'正式')"),
        ((MONTHLY, DATES), "dayofmonth(`stat_month`) > 1"),
        (
            (MONTHLY, COUNTS),
            "(`employee_count` IS NULL) OR ((`employee_count` >= 0) AND "
            "(`social_security_count` IS NULL OR `social_security_count` >= 0))",
        ),
        (
            (MONTHLY, COUNTS),
            MYSQL_8016_COUNTS_EXPRESSION.replace(
                "isnull(`employee_count`)", "isnull(`social_security_count`)"
            ),
        ),
        (
            (MONTHLY, COUNTS),
            MYSQL_8016_COUNTS_EXPRESSION.replace(" and ", " or "),
        ),
        (
            (MONTHLY, COUNTS),
            MYSQL_8016_COUNTS_EXPRESSION.replace(
                "isnull(`employee_count`)", "isnull(isnull(`employee_count`))"
            ),
        ),
    ],
    ids=[
        "changed-literal", "removed-literal", "changed-operator", "changed-grouping",
        "isnull-wrong-column", "isnull-changed-grouping", "isnull-nested",
    ],
)
def test_check_contract_rejects_semantic_changes(monkeypatch, identity, changed_expression):
    expressions = {**SHOW_EXPRESSIONS, identity: changed_expression}
    statements = {table: make_statement(table, expressions) for table in (PROFILE, MONTHLY)}
    connection, _ = mock_check_sources(monkeypatch, statements=statements)
    with pytest.raises(ValueError):
        database._check_contract(connection, {PROFILE, MONTHLY}, expected_checks())


@pytest.mark.parametrize("case", ["not-enforced", "missing", "extra", "wrong-owner"])
def test_check_contract_rejects_invalid_metadata_identity_or_enforcement(monkeypatch, case):
    metadata = [(table, name, "YES") for table, name in EXPECTED_EXPRESSIONS]
    if case == "not-enforced":
        table, name, _ = metadata[0]
        metadata[0] = (table, name, "NO")
    elif case == "missing":
        metadata.pop()
    elif case == "extra":
        metadata.append((PROFILE, "chk_unexpected", "YES"))
    else:
        _, name, enforced = metadata[0]
        metadata[0] = (MONTHLY, name, enforced)
    connection, _ = mock_check_sources(monkeypatch, metadata=metadata)
    with pytest.raises(ValueError):
        database._check_contract(connection, {PROFILE, MONTHLY}, expected_checks())


@pytest.mark.parametrize("case", ["missing", "extra", "malformed", "unknown-function", "unsupported-symbol"])
def test_check_contract_rejects_missing_or_unparseable_show_create_checks(monkeypatch, case):
    expressions = dict(SHOW_EXPRESSIONS)
    if case == "missing":
        expressions.pop((PROFILE, UNIT))
    elif case == "extra":
        expressions[(PROFILE, "chk_unexpected")] = "`fixture_column` >= 0"
    elif case == "unknown-function":
        expressions[(MONTHLY, DATES)] = "ABS(`stat_month`) = 1"
    elif case == "unsupported-symbol":
        expressions[(MONTHLY, DATES)] = "`fixture_column` + 1 > 0"
    statements = {table: make_statement(table, expressions) for table in (PROFILE, MONTHLY)}
    if case == "malformed":
        statements[PROFILE] = statements[PROFILE].replace(
            f"CONSTRAINT `{UNIT}` CHECK ((`amount_unit` in (_utf8mb4'元',_utf8mb4'万元')))",
            f"CONSTRAINT `{UNIT}` CHECK ((`amount_unit` in (_utf8mb4'元',_utf8mb4'万元')",
        )
    connection, _ = mock_check_sources(monkeypatch, statements=statements)
    with pytest.raises(ValueError):
        database._check_contract(connection, {PROFILE, MONTHLY}, expected_checks())
