from __future__ import annotations

import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lifelong_learning.core.models import (
    ChatMessage,
    EvaluationOutcome,
    Role,
    SampleStatus,
)
from src.lifelong_learning.environments.fixed_mysql_environment import (
    ReadOnlySQLViolation,
    validate_read_only_sql,
)
from src.lifelong_learning.tasks.enterprise_query_bench.dataset import (
    RowOrder,
)
from src.lifelong_learning.tasks.enterprise_query_bench.evaluator import (
    compare_query_results,
    normalize_mysql_rows,
)
from src.lifelong_learning.tasks.enterprise_query_bench.task import (
    EnterpriseQueryTask,
)


DATA_FILE = (
    PROJECT_ROOT / 'data/archive/questions/enterprise_query_bench/enterprise_query_bench_dev_v0.1.0/entry_dict_colloquial_instructions.json'
)
BASELINE_DIR = (
    PROJECT_ROOT / 'data/sql/synthetic_enterprise_v21_40_v1.0.0'
)


class FakeReadOnlyEnvironment:
    database = "synthetic_enterprise_v21"

    def __init__(self) -> None:
        self.closed = False
        self.result_by_sql: dict[str, list[tuple[Any, ...]] | Exception] = {}

    def verify_record_counts(self, expected: dict[str, int]) -> None:
        if len(expected) != 6:
            raise AssertionError("Expected six record-count checks.")

    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        result = self.result_by_sql[sql]
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        self.closed = True


def write_prompt(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "value": {
                    "0": {"role": "user", "content": "规则"},
                    "1": {"role": "agent", "content": "好的。"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class EnterpriseQueryBenchTests(unittest.TestCase):
    def test_read_only_sql_validation(self) -> None:
        self.assertEqual(validate_read_only_sql("SELECT 1;"), "SELECT 1")
        self.assertTrue(validate_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x"))

        for sql in (
            "DELETE FROM enterprise_basic;",
            "SELECT 1; DROP TABLE enterprise_basic;",
            "SELECT SLEEP(1);",
            "SELECT * FROM enterprise_basic FOR UPDATE;",
        ):
            with self.subTest(sql=sql):
                with self.assertRaises(ReadOnlySQLViolation):
                    validate_read_only_sql(sql)

    def test_result_normalization_and_comparison(self) -> None:
        actual = normalize_mysql_rows(
            [(Decimal("1.00"), "A"), (Decimal("2.0000"), "B")]
        )
        expected = [[2, "B"], ["1.000", "A"]]

        self.assertTrue(
            compare_query_results(actual, expected, RowOrder.UNORDERED)
        )
        self.assertFalse(
            compare_query_results(actual, expected, RowOrder.ORDERED)
        )

    def test_b1_uses_last_successful_query_result(self) -> None:
        fake_environment = FakeReadOnlyEnvironment()

        with tempfile.TemporaryDirectory() as temporary_directory:
            prompt_path = Path(temporary_directory) / "prompt.json"
            write_prompt(prompt_path)

            task = EnterpriseQueryTask(
                data_file_path=DATA_FILE,
                baseline_dir=BASELINE_DIR,
                prompt_file_path=prompt_path,
                environment=fake_environment,  # type: ignore[arg-type]
                max_round=4,
            )
            session = task.start_session("0")

            bad_sql = "SELECT missing_column FROM enterprise_basic"
            good_sql = (
                "SELECT enterprise_id, synthetic_credit_code, "
                "registration_status FROM enterprise_basic "
                "WHERE enterprise_name = '锐序精工制造有限公司'"
            )
            fake_environment.result_by_sql[bad_sql] = RuntimeError("unknown column")
            fake_environment.result_by_sql[good_sql] = [
                ("SYN-E000001", "SYN-USCC-000001", "存续")
            ]

            task.interact(
                session,
                ChatMessage(
                    role=Role.AGENT,
                    content=(
                        "Action: Operation\n```sql\n"
                        f"{bad_sql}\n```"
                    ),
                ),
            )
            self.assertEqual(session.sample_status, SampleStatus.RUNNING)

            task.interact(
                session,
                ChatMessage(
                    role=Role.AGENT,
                    content=(
                        "Action: Operation\n```sql\n"
                        f"{good_sql}\n```"
                    ),
                ),
            )

            # A later failed query must not replace the last successful result.
            later_bad_sql = "SELECT another_missing_column FROM enterprise_basic"
            fake_environment.result_by_sql[later_bad_sql] = RuntimeError(
                "unknown column"
            )
            task.interact(
                session,
                ChatMessage(
                    role=Role.AGENT,
                    content=(
                        "Action: Operation\n```sql\n"
                        f"{later_bad_sql}\n```"
                    ),
                ),
            )
            task.interact(
                session,
                ChatMessage(
                    role=Role.AGENT,
                    content=(
                        "Action: Answer\n"
                        "Final Answer: 这段文本不会参与评分"
                    ),
                ),
            )

            task.complete(session)

            self.assertEqual(
                session.evaluation_record.outcome,
                EvaluationOutcome.CORRECT,
            )
            self.assertEqual(len(session.task_output["sql_attempts"]), 3)
            self.assertFalse(session.task_output["sql_attempts"][0]["success"])
            self.assertTrue(session.task_output["sql_attempts"][1]["success"])
            self.assertFalse(session.task_output["sql_attempts"][2]["success"])


if __name__ == "__main__":
    unittest.main()
