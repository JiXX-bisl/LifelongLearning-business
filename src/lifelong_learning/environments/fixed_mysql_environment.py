from __future__ import annotations

import os
import re
from typing import Any

import mysql.connector


FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|REPLACE|CREATE|ALTER|DROP|TRUNCATE|"
    r"RENAME|GRANT|REVOKE|LOCK|UNLOCK|CALL|LOAD|HANDLER|DO|SET|USE)\b|"
    r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b|\bFOR\s+UPDATE\b|"
    r"\b(?:SLEEP|BENCHMARK|LOAD_FILE|GET_LOCK|RELEASE_LOCK)\s*\(",
    re.IGNORECASE,
)


class ReadOnlySQLViolation(ValueError):
    """The submitted SQL is outside the benchmark read-only boundary."""


def validate_read_only_sql(sql: str) -> str:
    """Validate and normalize one Agent-generated SELECT statement."""
    normalized = sql.strip()

    if not normalized:
        raise ReadOnlySQLViolation("SQL must not be empty.")

    # One optional trailing semicolon is accepted; internal semicolons are not.
    statement = normalized[:-1].strip() if normalized.endswith(";") else normalized

    if ";" in statement:
        raise ReadOnlySQLViolation("Only one SQL statement is allowed per round.")

    if not re.match(r"^(?:SELECT|WITH)\b", statement, re.IGNORECASE):
        raise ReadOnlySQLViolation("Only SELECT or WITH queries are allowed.")

    if FORBIDDEN_SQL_PATTERN.search(statement):
        raise ReadOnlySQLViolation("SQL contains a forbidden operation or function.")

    return statement


class FixedMySQLReadOnlyEnvironment:
    """Connect to the frozen enterprise database without managing its container."""

    def __init__(
        self,
        *,
        database: str,
        host: str = "127.0.0.1",
        port: int = 3306,
        user: str = "synthetic",
        password: str | None = None,
        password_env: str = "ENTERPRISE_MYSQL_PASSWORD",
        connection_timeout_seconds: int = 10,
        statement_timeout_ms: int = 10_000,
        max_result_rows: int = 50,
    ) -> None:
        resolved_password = password or os.environ.get(password_env)

        if not resolved_password:
            raise ValueError(
                f"MySQL password is missing. Set {password_env} or pass password explicitly."
            )

        if max_result_rows <= 0:
            raise ValueError("max_result_rows must be positive.")

        self.database = database
        self.max_result_rows = max_result_rows
        self.connection = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=resolved_password,
            database=database,
            connection_timeout=connection_timeout_seconds,
            autocommit=True,
            pool_reset_session=True,
            charset="utf8mb4",
            collation="utf8mb4_0900_ai_ci",
        )

        cursor = self.connection.cursor()
        try:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
            cursor.execute(
                "SET SESSION MAX_EXECUTION_TIME = %s",
                (statement_timeout_ms,),
            )
        finally:
            cursor.close()

    def _ensure_connection(self) -> None:
        if not self.connection.is_connected():
            self.connection.reconnect(attempts=1, delay=0)

    def verify_record_counts(
        self,
        expected_record_counts: dict[str, int],
    ) -> None:
        """Fail before an experiment if the live database is not the expected baseline."""
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            for table_name, expected_count in expected_record_counts.items():
                if not re.fullmatch(r"[a-z][a-z0-9_]*", table_name):
                    raise ValueError(f"Unsafe table name in manifest: {table_name!r}")

                cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
                row = cursor.fetchone()
                actual_count = None if row is None else int(row[0])

                if actual_count != expected_count:
                    raise RuntimeError(
                        "Live database record count mismatch: "
                        f"table={table_name}, actual={actual_count}, "
                        f"expected={expected_count}."
                    )
        finally:
            cursor.close()

    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        """Execute one validated read-only query and return at most 50 rows."""
        statement = validate_read_only_sql(sql)
        self._ensure_connection()
        cursor = self.connection.cursor()

        try:
            cursor.execute(statement)

            if not cursor.with_rows:
                raise ReadOnlySQLViolation("The statement did not return a result set.")

            rows = cursor.fetchmany(self.max_result_rows + 1)

            if len(rows) > self.max_result_rows:
                # Drain an unbuffered connector result before reusing the
                # connection for a later Agent attempt.
                cursor.fetchall()
                raise RuntimeError(
                    "Query returned more than the configured maximum of "
                    f"{self.max_result_rows} rows."
                )

            return rows
        finally:
            cursor.close()

    def close(self) -> None:
        """
        Best-effort and idempotent connection cleanup.

        runner和入口脚本在异常路径中都可能调用close，
        因此重复关闭不应掩盖原始异常。
        """
        connection = getattr(
            self,
            "connection",
            None,
        )

        if connection is None:
            return

        try:
            connection.close()
        except (
            AttributeError,
            mysql.connector.Error,
        ):
            # 已关闭的连接池代理可能不再持有底层连接。
            pass
