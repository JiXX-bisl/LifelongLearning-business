"""Explicit user-invoked import and read-only acceptance for V3.0 artifacts.

The database driver is imported only inside a database operation. Merely
importing this module, generating artifacts, or offline validation does not
connect to MySQL.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .schema import COLUMN_TYPES, INSERT_ORDER, MONEY_FIELDS, NULLABLE_COLUMNS, PRIMARY_KEYS, TABLE_COLUMNS, TABLES, build_schema_sql
from .validation import _identity_and_relations, _maps


TARGET_DATABASE = "synthetic_enterprise_v30_120"
PASSWORD_ENVIRONMENT = "ENTERPRISE_MYSQL_PASSWORD"
BATCH_SIZE = 500


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, parse_float=Decimal, object_pairs_hook=_object_pairs)


def _artifact_file(root: Path, relative: Any) -> Path:
    _require(isinstance(relative, str) and bool(relative), "artifact filename is missing")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root) and path.is_file(), "artifact file is missing or outside the artifact directory")
    return path


def _load_artifact(artifact_dir: str | Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], str]:
    root = Path(artifact_dir).expanduser().resolve()
    manifest = _read_json(_artifact_file(root, "manifest.json"))
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == "3.0", "expected a version 3.0 manifest")
    _require(isinstance(manifest.get("baseline_id"), str) and re.fullmatch(r"synthetic_enterprise_v30_120_v[1-9]\d*\.\d+\.\d+", manifest["baseline_id"]) is not None, "expected an independent V3.0 120-enterprise baseline")
    _require(manifest.get("status") == "GENERATED", "expected an unmodified GENERATED dataset artifact")
    _require(manifest.get("mysql", {}).get("database") == TARGET_DATABASE, "manifest targets another database")
    files = manifest.get("files", {})
    names = {"tables": files.get("tables", "tables.json"), "schema": files.get("schema", "database_schema.sql")}
    artifacts = manifest.get("artifacts", {})
    _require(isinstance(artifacts, dict) and all(name in artifacts for name in names.values()), "manifest has no checksums for the data and schema")
    for relative, metadata in artifacts.items():
        path = _artifact_file(root, relative)
        _require(isinstance(metadata, dict) and metadata.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest(), f"artifact checksum differs: {relative}")
        _require(metadata.get("bytes") == path.stat().st_size, f"artifact byte count differs: {relative}")
    schema_sql = _artifact_file(root, names["schema"]).read_text(encoding="utf-8")
    _require(schema_sql == build_schema_sql(), "artifact schema does not match this implementation's V3.0 contract")
    tables = _read_json(_artifact_file(root, names["tables"]))
    _require(isinstance(tables, dict), "expected table arrays")
    _maps(tables, "import_artifact")
    _identity_and_relations(tables, "import_artifact")
    actual_counts = {table: len(tables[table]) for table in TABLES}
    _require(manifest.get("record_counts") == actual_counts, "artifact record counts differ from the manifest")
    _require(len(tables["enterprise_basic"]) == 120, "expected 120 enterprises")
    return manifest, tables, schema_sql


def _connection(connection_options: Mapping[str, Any], *, select_database: bool) -> Any:
    _require(isinstance(connection_options, Mapping), "connection options must be an object")
    database = connection_options.get("database", TARGET_DATABASE)
    _require(database == TARGET_DATABASE, "only the new synthetic_enterprise_v30_120 database is allowed; the frozen V2.1 database must not be modified")
    _require("password" not in connection_options, "read MySQL credentials from ENTERPRISE_MYSQL_PASSWORD, not a plaintext argument")
    password_environment = connection_options.get("password_env", PASSWORD_ENVIRONMENT)
    _require(password_environment == PASSWORD_ENVIRONMENT, "expected credential environment ENTERPRISE_MYSQL_PASSWORD")
    password = os.environ.get(PASSWORD_ENVIRONMENT)
    _require(bool(password), f"environment {PASSWORD_ENVIRONMENT} is not set")
    try:
        import mysql.connector
    except ModuleNotFoundError as error:
        raise RuntimeError("mysql-connector-python is required only for the explicit import/accept commands") from error
    options = {"host": connection_options.get("host", "127.0.0.1"), "port": int(connection_options.get("port", 3306)), "user": connection_options.get("user", "synthetic"), "password": password, "autocommit": False, "charset": "utf8mb4", "use_unicode": True}
    _require(1 <= options["port"] <= 65535, "invalid MySQL port")
    if select_database:
        options["database"] = TARGET_DATABASE
    return mysql.connector.connect(**options)


def _fetch_all(connection: Any, sql: str, parameters: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, parameters)
        return list(cursor.fetchall())
    finally:
        cursor.close()


def _execute(connection: Any, sql: str, parameters: tuple[Any, ...] = ()) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, parameters)
    finally:
        cursor.close()


def _schema_statements(schema_sql: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for statement in schema_sql.split(";"):
        if not statement.strip():
            continue
        match = re.match(r"\s*CREATE TABLE `([^`]+)`", statement)
        _require(match is not None and match.group(1) in TABLES, "schema contains an unexpected statement")
        result[match.group(1)] = statement.strip()
    _require(set(result) == set(TABLES), "schema statement set is incomplete")
    return result


def _normalize_type(value: str) -> str:
    normalized = re.sub(r"\b(tinyint|smallint|mediumint|int|bigint)\(\d+\)", r"\1", value.lower())
    return re.sub(r"\s+", " ", normalized.strip())


def _normalize_check(value: str) -> str:
    """Parse the limited CHECK grammar while retaining AND/OR grouping.

    MySQL adds redundant parentheses and literal charset introducers when it
    prints CHECK_CLAUSE. Text stripping loses significant operator grouping;
    comparing syntax trees removes only those harmless presentation changes.
    Unsupported expressions fail closed rather than being approximated.
    """
    expression = re.sub(r"_[a-zA-Z0-9]+(?=')", "", value)
    lexer = re.compile(r"\s*(<=|>=|<>|!=|=|<|>|[(),]|`[^`]+`|'(?:''|[^'])*'|\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)")
    tokens = []
    position = 0
    while position < len(expression):
        if not expression[position:].strip():
            break
        match = lexer.match(expression, position)
        _require(match is not None, "unsupported token in schema CHECK expression")
        token = match.group(1)
        tokens.append(token if token.startswith("'") else token.strip("`").lower())
        position = match.end()
    cursor = 0

    def take(expected: str | None = None) -> str:
        nonlocal cursor
        _require(cursor < len(tokens), "incomplete schema CHECK expression")
        token = tokens[cursor]
        _require(expected is None or token == expected, "unexpected schema CHECK token")
        cursor += 1
        return token

    def primary() -> Any:
        token = take()
        if token == "(":
            node = logical_or()
            take(")")
            return node
        if token.startswith("'"):
            return ("literal", token[1:-1].replace("''", "'"))
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return ("number", str(Decimal(token).normalize()))
        _require(re.fullmatch(r"[a-z_][a-z0-9_]*", token) is not None, "unsupported schema CHECK operand")
        if cursor < len(tokens) and tokens[cursor] == "(":
            _require(token in {"day", "dayofmonth", "year", "month", "isnull"}, "unsupported schema CHECK function")
            take("(")
            argument = logical_or()
            take(")")
            # MySQL 8.0.16 renders IS NULL as ISNULL() in SHOW CREATE TABLE.
            if token == "isnull":
                return ("is_null", argument)
            return ("function", "day" if token == "dayofmonth" else token, argument)
        return ("identifier", token)

    def comparison() -> Any:
        node = primary()
        if cursor >= len(tokens):
            return node
        operation = tokens[cursor]
        if operation in {"=", "<", ">", "<=", ">=", "<>", "!="}:
            take()
            return ("compare", "<>" if operation == "!=" else operation, node, primary())
        if operation == "is":
            take("is")
            take("null")
            return ("is_null", node)
        if operation == "in":
            take("in")
            take("(")
            values = [primary()]
            while cursor < len(tokens) and tokens[cursor] == ",":
                take(",")
                values.append(primary())
            take(")")
            return ("in", node, tuple(values))
        return node

    def logical_and() -> Any:
        node = comparison()
        while cursor < len(tokens) and tokens[cursor] == "and":
            take("and")
            node = ("and", node, comparison())
        return node

    def logical_or() -> Any:
        node = logical_and()
        while cursor < len(tokens) and tokens[cursor] == "or":
            take("or")
            node = ("or", node, logical_and())
        return node

    tree = logical_or()
    _require(cursor == len(tokens), "unconsumed schema CHECK expression")
    def canonical(node: Any) -> Any:
        if not isinstance(node, tuple):
            return node
        if node and node[0] in {"and", "or"}:
            children = []
            for child in (canonical(node[1]), canonical(node[2])):
                if isinstance(child, tuple) and child[0] == node[0]:
                    children.extend(child[1])
                else:
                    children.append(child)
            return (node[0], tuple(children))
        return tuple(canonical(child) for child in node)
    return json.dumps(canonical(tree), ensure_ascii=False, separators=(",", ":"))


def _check_definitions(statement: str) -> dict[str, str]:
    """Read named CHECK definitions from the V3.0 DDL or SHOW CREATE TABLE."""
    checks: dict[str, str] = {}
    for line in statement.splitlines():
        match = re.search(r"CONSTRAINT `([^`]+)` CHECK \((.*)\),?\s*$", line)
        if match:
            name, expression = match.groups()
            _require(name not in checks, f"duplicate CHECK definition: {name}")
            checks[name] = _normalize_check(expression)
    return checks


def _check_contract(connection: Any, existing: set[str], expected_checks: dict[tuple[str, str], str]) -> None:
    """Compare actual CHECK semantics without relying on encoded CHECK_CLAUSE text.

    Some MySQL metadata responses escape literal delimiters and misencode
    non-ASCII CHECK_CLAUSE values. SHOW CREATE TABLE preserves the executable
    expressions; TABLE_CONSTRAINTS independently supplies identity/enforcement.
    """
    _require(existing <= set(TABLES), "unexpected table in CHECK contract")
    check_rows = _fetch_all(connection, "SELECT TABLE_NAME, CONSTRAINT_NAME, ENFORCED FROM information_schema.TABLE_CONSTRAINTS WHERE CONSTRAINT_SCHEMA=%s AND CONSTRAINT_TYPE='CHECK'", (TARGET_DATABASE,))
    metadata_checks: set[tuple[str, str]] = set()
    for table, name, enforced in check_rows:
        identity = (table, name)
        _require(identity not in metadata_checks, f"duplicate CHECK metadata: {name}")
        _require(enforced == "YES", f"CHECK is not enforced: {name}")
        metadata_checks.add(identity)
    _require(metadata_checks == set(expected_checks), "CHECK identities differ from V3.0")

    actual_checks: dict[tuple[str, str], str] = {}
    for table in sorted(existing):
        rows = _fetch_all(connection, f"SHOW CREATE TABLE `{table}`")
        _require(len(rows) == 1 and len(rows[0]) == 2 and rows[0][0] == table
                 and isinstance(rows[0][1], str), f"unexpected SHOW CREATE TABLE response: {table}")
        for name, expression in _check_definitions(rows[0][1]).items():
            actual_checks[(table, name)] = expression
    _require(set(actual_checks) == metadata_checks, "SHOW CREATE TABLE CHECK identities differ from metadata")
    _require(actual_checks == expected_checks, "CHECK contract differs from V3.0")


def _schema_contract(connection: Any, schema_sql: str, *, allow_partial: bool = False) -> set[str]:
    database_properties = _fetch_all(connection, "SELECT DEFAULT_CHARACTER_SET_NAME, DEFAULT_COLLATION_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME=%s", (TARGET_DATABASE,))
    _require(database_properties == [("utf8mb4", "utf8mb4_0900_ai_ci")], "the target database charset/collation differs from V3.0")
    table_rows = _fetch_all(connection, "SELECT TABLE_NAME, TABLE_TYPE, ENGINE, TABLE_COLLATION FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s", (TARGET_DATABASE,))
    existing = {row[0] for row in table_rows}
    _require(existing <= set(TABLES), "the target database contains unexpected tables or views")
    if not allow_partial:
        _require(existing == set(TABLES), "the target schema is incomplete")
    for table, kind, engine, collation in table_rows:
        _require(kind == "BASE TABLE" and str(engine).lower() == "innodb" and collation == "utf8mb4_0900_ai_ci", f"unexpected engine/collation for {table}")
    column_rows = _fetch_all(connection, "SELECT TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, COLUMN_TYPE, IS_NULLABLE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME, ORDINAL_POSITION", (TARGET_DATABASE,))
    by_table: dict[str, list[tuple[Any, ...]]] = {table: [] for table in existing}
    for row in column_rows:
        _require(row[0] in existing, "unexpected column owner")
        by_table[row[0]].append(row)
    for table, rows in by_table.items():
        _require(tuple(row[1] for row in rows) == TABLE_COLUMNS[table], f"column names/order differ for {table}")
        for _, column, _, column_type, nullable in rows:
            _require(_normalize_type(column_type) == _normalize_type(COLUMN_TYPES[table][column]), f"column type differs for {table}.{column}")
            _require((nullable == "YES") == (column in NULLABLE_COLUMNS[table]), f"column nullability differs for {table}.{column}")
    key_rows = _fetch_all(connection, "SELECT TABLE_NAME, CONSTRAINT_NAME, COLUMN_NAME, ORDINAL_POSITION, REFERENCED_TABLE_SCHEMA, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME, CONSTRAINT_NAME, ORDINAL_POSITION", (TARGET_DATABASE,))
    primary: dict[str, list[str]] = {table: [] for table in existing}
    foreign: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for table, constraint, column, _, referenced_schema, referenced_table, referenced_column in key_rows:
        if constraint == "PRIMARY":
            primary[table].append(column)
        elif referenced_table is not None:
            _require(referenced_schema == TARGET_DATABASE, "foreign key points outside the new dataset database")
            foreign.setdefault((table, constraint), []).append((column, referenced_table, referenced_column))
    for table in existing:
        _require(tuple(primary[table]) == PRIMARY_KEYS[table], f"primary key differs for {table}")
    statements = _schema_statements(schema_sql)
    expected_foreign: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    expected_checks: dict[tuple[str, str], str] = {}
    expected_indexes: dict[tuple[str, str], tuple[int, tuple[str, ...]]] = {}
    for table in existing:
        expected_indexes[(table, "PRIMARY")] = (0, PRIMARY_KEYS[table])
        for name, expression in _check_definitions(statements[table]).items():
            expected_checks[(table, name)] = expression
        for line in statements[table].splitlines():
            foreign_match = re.search(r"CONSTRAINT `([^`]+)` FOREIGN KEY \(([^)]+)\) REFERENCES `([^`]+)` \(([^)]+)\)", line)
            if foreign_match:
                name, source, target, destination = foreign_match.groups()
                source_columns = re.findall(r"`([^`]+)`", source)
                target_columns = re.findall(r"`([^`]+)`", destination)
                expected_foreign[(table, name)] = list(zip(source_columns, [target] * len(source_columns), target_columns))
            index_match = re.search(r"(UNIQUE )?KEY `([^`]+)` \(([^)]+)\)", line)
            if index_match:
                unique, name, columns = index_match.groups()
                expected_indexes[(table, name)] = (0 if unique else 1, tuple(re.findall(r"`([^`]+)`", columns)))
    _require(foreign == expected_foreign, "foreign-key contract differs from V3.0")
    _check_contract(connection, existing, expected_checks)
    index_rows = _fetch_all(connection, "SELECT TABLE_NAME, INDEX_NAME, NON_UNIQUE, SEQ_IN_INDEX, COLUMN_NAME FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=%s ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX", (TARGET_DATABASE,))
    actual_indexes: dict[tuple[str, str], tuple[int, tuple[str, ...]]] = {}
    grouped_indexes: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for table, name, non_unique, _, column in index_rows:
        grouped_indexes.setdefault((table, name), []).append((non_unique, column))
    for identity, values in grouped_indexes.items():
        actual_indexes[identity] = (values[0][0], tuple(value[1] for value in values))
    _require(actual_indexes == expected_indexes, "index contract differs from V3.0")
    return existing


def _counts(connection: Any, tables: set[str]) -> dict[str, int]:
    return {table: int(_fetch_all(connection, f"SELECT COUNT(*) FROM `{table}`")[0][0]) for table in TABLES if table in tables}


def _transport_value(table: str, column: str, value: Any) -> Any:
    if value is None:
        return None
    column_type = COLUMN_TYPES[table][column].upper()
    if column in MONEY_FIELDS.get(table, ()):
        _require(not isinstance(value, (float, bool)), f"money transport must use Decimal: {table}.{column}")
        return value if isinstance(value, Decimal) else Decimal(value)
    if column_type == "DATE":
        if isinstance(value, datetime):
            return value.date()
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    if column_type == "DATETIME":
        return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return value


def _insert(connection: Any, tables: dict[str, list[dict[str, Any]]]) -> None:
    cursor = connection.cursor()
    try:
        for table in INSERT_ORDER:
            columns = TABLE_COLUMNS[table]
            sql = f"INSERT INTO `{table}` (" + ", ".join(f"`{column}`" for column in columns) + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
            rows = tables[table]
            for offset in range(0, len(rows), BATCH_SIZE):
                batch = [tuple(_transport_value(table, column, row[column]) for column in columns) for row in rows[offset:offset + BATCH_SIZE]]
                cursor.executemany(sql, batch)
                _require(cursor.rowcount == len(batch), f"insert count differs for {table}; the transaction will be rolled back")
    finally:
        cursor.close()


def _compare(connection: Any, tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    checked_rows = {}
    cursor = connection.cursor()
    try:
        for table in TABLES:
            columns = TABLE_COLUMNS[table]
            key_indices = tuple(columns.index(field) for field in PRIMARY_KEYS[table])
            expected = {}
            for row in tables[table]:
                values = tuple(_transport_value(table, column, row[column]) for column in columns)
                key = tuple(values[index] for index in key_indices)
                expected[key] = values
            seen = set()
            cursor.execute("SELECT " + ", ".join(f"`{column}`" for column in columns) + f" FROM `{table}`")
            while True:
                batch = cursor.fetchmany(BATCH_SIZE)
                if not batch:
                    break
                for actual in batch:
                    values = tuple(actual)
                    for column, value in zip(columns, values):
                        if column in MONEY_FIELDS.get(table, ()) and value is not None:
                            _require(isinstance(value, Decimal) and value.is_finite(), f"database driver did not return an exact Decimal for {table}.{column}")
                    key = tuple(values[index] for index in key_indices)
                    _require(key not in seen, f"database repeats a technical key in {table}")
                    _require(key in expected and values == expected[key], f"database differs from the artifact in {table} at key {key}; no overwrite or partial retry is allowed")
                    seen.add(key)
            _require(seen == set(expected), f"database has missing rows in {table}; no partial retry is allowed")
            checked_rows[table] = len(seen)
    finally:
        cursor.close()
    return {"record_counts": checked_rows, "all_fields_exact_match": True, "decimal_exact_match": True, "read_batch_size": BATCH_SIZE}


def import_dataset(artifact_dir: str | Path, connection_options: Mapping[str, Any]) -> dict[str, Any]:
    """Import into a separately deployed empty V3.0 schema, or confirm an exact retry.

    This entry point executes no CREATE/ALTER/DROP statements. The user deploys
    create_database.sql and database_schema.sql before invoking import.
    """
    manifest, tables, schema_sql = _load_artifact(artifact_dir)
    connection = _connection(connection_options, select_database=True)
    lock_acquired = False
    try:
        lock_acquired = _fetch_all(connection, "SELECT GET_LOCK(%s, 0)", ("enterprise_v30_120_import",))[0][0] == 1
        _require(lock_acquired, "another importer is working on the new dataset database")
        existing = _schema_contract(connection, schema_sql)
        before_counts = _counts(connection, existing)
        connection.rollback()
        if any(before_counts.values()):
            _require(existing == set(TABLES), "a partially populated/incomplete database cannot be retried automatically")
            connection.start_transaction(consistent_snapshot=True, readonly=True)
            result = _compare(connection, tables)
            connection.rollback()
            return {"status": "already_imported", "database": TARGET_DATABASE, "baseline_id": manifest.get("baseline_id"), "schema_version": "3.0", "replayed": True, **result}
        connection.start_transaction(isolation_level="SERIALIZABLE")
        # Lock the complete empty primary-key ranges before writing. This also
        # prevents a second writer that does not cooperate with GET_LOCK from
        # inserting extra rows during this transaction.
        for table in INSERT_ORDER:
            columns = ", ".join(f"`{column}`" for column in PRIMARY_KEYS[table])
            _require(not _fetch_all(connection, f"SELECT {columns} FROM `{table}` FOR UPDATE"), "target tables changed before import; refusing a partial/conflicting write")
        _insert(connection, tables)
        result = _compare(connection, tables)
        connection.commit()
        return {"status": "import_committed", "database": TARGET_DATABASE, "baseline_id": manifest.get("baseline_id"), "schema_version": "3.0", "replayed": False, **result}
    except BaseException:
        connection.rollback()
        raise
    finally:
        if lock_acquired:
            try:
                _fetch_all(connection, "SELECT RELEASE_LOCK(%s)", ("enterprise_v30_120_import",))
            finally:
                connection.close()
        else:
            connection.close()


def accept_dataset(artifact_dir: str | Path, connection_options: Mapping[str, Any], *, private_dir: str | Path | None = None) -> dict[str, Any]:
    """User-operated full-field database readback and offline scenario acceptance."""
    manifest, tables, schema_sql = _load_artifact(artifact_dir)
    _require(private_dir is not None, "complete acceptance requires --private-dir for the separate offline fact/scenario/pollution evidence")
    # Private evidence stays in the local evaluator. It is never imported or
    # exposed as a SQL table; exact readback connects the observed scenarios to
    # the database without changing any public records or candidate manifests.
    from .build import validate_artifacts
    offline = validate_artifacts(artifact_dir, private_dir)
    connection = _connection(connection_options, select_database=True)
    try:
        connection.start_transaction(consistent_snapshot=True, readonly=True)
        _schema_contract(connection, schema_sql)
        result = _compare(connection, tables)
        connection.rollback()
        return {"status": "PASS", "database": TARGET_DATABASE, "baseline_id": manifest.get("baseline_id"), "schema_version": "3.0", "readonly": True, "schema_contract_exact_match": True, "offline_structure_business_scenario_pollution": offline, "candidate_manifest_unchanged": True, **result}
    finally:
        connection.rollback()
        connection.close()
