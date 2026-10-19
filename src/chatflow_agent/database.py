from __future__ import annotations

from typing import Any

import mysql.connector

from src.lifelong_learning.environments.fixed_mysql_environment import (
    FixedMySQLReadOnlyEnvironment, ReadOnlySQLViolation, validate_read_only_sql
)
from src.lifelong_learning.tasks.enterprise_query_bench.evaluator import normalize_mysql_rows


class BusinessMySQLReadOnlyEnvironment(FixedMySQLReadOnlyEnvironment):
    def __init__(
        self,
        *,
        fetch_batch_size: int = 50,
        max_total_rows: int = 500,
        **connection_options: Any,
    ) -> None:
        if fetch_batch_size <= 0 or max_total_rows <= 0:
            raise ValueError("Batch和Rows上限必须为正整数。")
        self.fetch_batch_size = fetch_batch_size
        self.max_total_rows = max_total_rows
        self._connection_options = dict(connection_options)
        self._closed = False
        self.connection = None

        self._open_connection()

    def _open_connection(self) -> None:
        try:
            super().__init__(**self._connection_options)
        except Exception:
            super().close()
            self.connection = None
            raise

    def _discard_connection(self) -> None:
        super().close()
        self.connection = None

    def _ensure_connection(self):
        if self._closed:
            raise RuntimeError("业务数据库连接已关闭。")
        
        if self.connection is None:
            self._open_connection()
        elif not self.connection.is_connected():
            raise mysql.connector.InterfaceError("数据库连接已断开。")
    
    def execute_query(self, sql: str) -> dict[str, Any]:
        statement = validate_read_only_sql(sql)
        self._ensure_connection()

        cursor = None
        discard_connection = True

        try:
            cursor = self.connection.cursor()
            cursor.execute(statement)

            if not cursor.with_rows:
                raise ReadOnlySQLViolation("该语句没有返回结果。")
            columns = [column[0] for column in (cursor.description or [])]
            rows = []
            read_limit = self.max_total_rows + 1
            while len(rows) < read_limit:
                batch_size = min(
                    self.fetch_batch_size, 
                    read_limit - len(rows)
                )
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                rows.extend(batch)
            
            is_complete = len(rows) <= self.max_total_rows
            returned_rows = rows[:self.max_total_rows]
            result = {
                "columns": columns,
                "rows": normalize_mysql_rows(returned_rows),
                "row_count": len(returned_rows),
                "is_complete": is_complete,
                "incomplete_reason": (
                    None if is_complete else "max_total_rows"
                ),
            }
            discard_connection = not is_complete
            return result
        finally:
            if discard_connection:
                self._discard_connection()
            if cursor is not None:
                try:
                    cursor.close()
                except (mysql.connector.Error, ReferenceError):
                    self._discard_connection()

    def close(self) -> None:
        self._closed = True
        self._discard_connection()
        