from unittest.mock import Mock, patch

import pytest

from src.chatflow_agent.database import (
    BusinessMySQLReadOnlyEnvironment,
)


def make_connection(total_rows):
    rows = [(index,) for index in range(total_rows)]
    position = 0

    def fetchmany(size):
        nonlocal position
        batch = rows[position:position + size]
        position += len(batch)
        return batch

    setup_cursor = Mock()

    query_cursor = Mock()
    query_cursor.with_rows = True
    query_cursor.description = [("序号",)]
    query_cursor.fetchmany.side_effect = fetchmany

    connection = Mock()
    connection.is_connected.return_value = True
    connection.cursor.side_effect = [
        setup_cursor,
        query_cursor,
    ]

    return connection, setup_cursor, query_cursor


@pytest.mark.parametrize("total_rows", [0, 50, 51, 500, 501])
def test_bounded_fetch_and_connection_cleanup(total_rows):
    connection, setup_cursor, query_cursor = make_connection(
        total_rows
    )
    next_connection, next_setup, next_cursor = make_connection(0)

    with patch(
        "src.lifelong_learning.environments."
        "fixed_mysql_environment.mysql.connector.connect",
        side_effect=[connection, next_connection],
    ) as connect:
        environment = BusinessMySQLReadOnlyEnvironment(
            database="test_database",
            password="test-only-password",
            fetch_batch_size=50,
            max_total_rows=500,
        )

        try:
            result = environment.execute_query(
                "SELECT value AS 序号 FROM test_table;"
            )

            expected_count = min(total_rows, 500)
            expected_complete = total_rows <= 500

            assert result["columns"] == ["序号"]
            assert result["row_count"] == expected_count
            assert result["rows"] == [
                [index] for index in range(expected_count)
            ]
            assert result["is_complete"] is expected_complete
            assert result["incomplete_reason"] == (
                None if expected_complete else "max_total_rows"
            )

            query_cursor.execute.assert_called_once_with(
                "SELECT value AS 序号 FROM test_table"
            )
            query_cursor.fetchall.assert_not_called()
            query_cursor.close.assert_called_once()
            setup_cursor.close.assert_called_once()

            assert all(
                1 <= call.args[0] <= 50
                for call in query_cursor.fetchmany.call_args_list
            )

            if expected_complete:
                connection.close.assert_not_called()
                assert connect.call_count == 1
            else:
                connection.close.assert_called_once()

                # 截断之后，下一条SQL应使用新连接。
                next_result = environment.execute_query(
                    "SELECT value AS 序号 FROM test_table;"
                )
                assert next_result["is_complete"] is True
                assert next_result["rows"] == []
                assert connect.call_count == 2

                # 新连接同样应用只读和超时设置。
                assert (
                    next_setup.execute.call_args_list
                    == setup_cursor.execute.call_args_list
                )
                next_cursor.fetchall.assert_not_called()
        finally:
            environment.close()

        connection.close.assert_called_once()
        if total_rows > 500:
            next_connection.close.assert_called_once()