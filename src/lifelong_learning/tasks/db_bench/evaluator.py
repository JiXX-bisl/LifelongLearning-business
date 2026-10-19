from __future__ import annotations

import ast
import math
import re
from typing import Any

from src.lifelong_learning.environments.mysql_environment import MySQLEnvironment

from src.lifelong_learning.tasks.db_bench.dataset import DBBenchEntry, DBValue


def _normalize_decimal_expression(answer: str) -> str:
    return re.sub(
        r"Decimal\((['\"])(.*?)\1\)", 
        r"\2",
        answer
    )

def _is_value_equal(agent_value: Any, expected_value: DBValue) -> bool:
    """Compare individual query result values."""
    if isinstance(expected_value, (int, float)):
        try:
            processed_agent_value = float(agent_value)
        except (TypeError, ValueError):
                return False
        return math.isclose(
             processed_agent_value, 
             float(expected_value), 
             rel_tol=1e-6,
             abs_tol=1e-6
        )
    return agent_value == expected_value

def validate_direct_answer(agent_answer: str, expected_answer: list[list[DBValue]]) -> bool:
    """Verify the direct query results of the SELECT class tasks"""
    normalized_answer = _normalize_decimal_expression(agent_answer)
    try:
        parsed_answer = ast.literal_eval(normalized_answer)
    except (SyntaxError, ValueError):
         return False
    
    if not isinstance(parsed_answer, list):
         return False
    if len(parsed_answer) != len(expected_answer):
         return False
    
    for agent_row, expected_row in zip(parsed_answer, expected_answer):
         if not isinstance(agent_row, (list, tuple)):
              return False
         if len(agent_row) != len(expected_row):
              return False
         for agent_value , expected_value in zip(agent_row, expected_row):
              if not _is_value_equal(agent_value, expected_value):
                   return False
              
    return True

def calculate_table_state_md5(environment: MySQLEnvironment, entry: DBBenchEntry) -> str:
     """Calculate the MD5 of the entire table state after the execution of the modification task. """
     column_name_sql = ", ".join(
          f"`{column.name}`"
          for column in entry.table_info.column_info_list
     )

     md5_query = (
        "SELECT MD5("
        "GROUP_CONCAT(rowhash ORDER BY rowhash)"
        ") AS hash "
        "FROM ("
        "SELECT SUBSTRING("
        f"MD5(CONCAT_WS(',', {column_name_sql})), "
        "1, 5"
        ") AS rowhash "
        f"FROM `{entry.table_info.name}`"
        ") AS table_state"
    )
     
     result = environment.execute(md5_query, database=entry.database_name)

     table_md5 = result[0][0]
     return str(table_md5)

def validate_table_state(environment: MySQLEnvironment, entry: DBBenchEntry) -> bool:
     """Verify the state of table after modification task"""
     expected_md5 = entry.answer_info.md5
     if expected_md5 is None: return False
     actual_md5 = calculate_table_state_md5(environment, entry)
     return actual_md5 == expected_md5
