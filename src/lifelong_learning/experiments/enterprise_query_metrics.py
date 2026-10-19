from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Sequence

from src.lifelong_learning.core.models import (
    EvaluationOutcome,
    Session,
)
from src.lifelong_learning.experiments.runner import (
    BenchmarkTask,
    calculate_dbbench_metric,
)


DimensionGetter = Callable[[Any], str]


def _new_metric_bucket() -> dict[str, int | float]:
    return {
        "sample_count": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "correct_rate": 0.0,
    }


def _calculate_dimension_metric(
    task: BenchmarkTask,
    session_list: Sequence[Session],
    getter: DimensionGetter,
) -> dict[str, dict[str, int | float]]:
    """统计不同难度问题的正确率"""
    result: dict[
        str,
        dict[str, int | float],
    ] = {}

    for session in session_list:
        entry = task.dataset[
            session.sample_index
        ]
        dimension_value = str(
            getter(entry)
        )

        if dimension_value not in result:
            result[dimension_value] = (
                _new_metric_bucket()
            )

        bucket = result[dimension_value]
        bucket["sample_count"] = (
            int(bucket["sample_count"]) + 1
        )

        if (
            session.evaluation_record.outcome
            == EvaluationOutcome.CORRECT
        ):
            bucket["correct_count"] = (
                int(bucket["correct_count"]) + 1
            )
        else:
            bucket["incorrect_count"] = (
                int(bucket["incorrect_count"]) + 1
            )

    for bucket in result.values():
        sample_count = int(
            bucket["sample_count"]
        )
        correct_count = int(
            bucket["correct_count"]
        )

        bucket["correct_rate"] = (
            correct_count / sample_count
            if sample_count > 0
            else 0.0
        )

    return dict(sorted(result.items()))


def _classify_sql_error(
    error_text: str,
) -> str:
    normalized = error_text.lower()

    if any(
        marker in normalized
        for marker in (
            "only select",
            "only one sql statement",
            "forbidden operation",
            "forbidden keyword",
            "read-only",
            "read only",
            "did not return a result set",
        )
    ):
        return "read_only_boundary"

    if any(
        marker in normalized
        for marker in (
            "1064",
            "syntax",
            "unknown column",
            "1054",
            "doesn't exist",
            "does not exist",
            "1146",
            "unknown table",
            "ambiguous",
            "1052",
        )
    ):
        return "sql_syntax_or_schema"

    if any(
        marker in normalized
        for marker in (
            "timeout",
            "timed out",
            "max_execution_time",
            "maximum statement execution time",
            "3024",
        )
    ):
        return "query_timeout"

    if any(
        marker in normalized
        for marker in (
            "more than the configured maximum",
            "maximum of 50 rows",
        )
    ):
        return "result_row_limit"

    if any(
        marker in normalized
        for marker in (
            "connection",
            "server has gone away",
            "lost connection",
            "2003",
            "2006",
            "2013",
        )
    ):
        return "database_connection"

    return "other"


def _calculate_sql_diagnostics(
    session_list: Sequence[Session],
) -> dict[str, Any]:
    """分析LLM在输出结果的过程表现"""
    total_attempt_count = 0
    successful_attempt_count = 0
    failed_attempt_count = 0

    sample_with_attempt_count = 0  # 执行过SQL的样本数
    sample_without_attempt_count = 0  # 没有执行SQL的样本数
    sample_without_success_count = 0  # 没有成功执行任何一次SQL的样本数
    sample_with_retry_count = 0
    sample_with_failed_attempt_count = 0
    correct_after_retry_count = 0  # 重试之后成功的次数

    error_type_counter: Counter[str] = (
        Counter()
    )
    attempt_count_distribution: Counter[str] = (
        Counter()
    )

    for session in session_list:
        task_output = session.task_output or {}
        raw_attempts = task_output.get(
            "sql_attempts",
            [],
        )

        attempts = (
            raw_attempts
            if isinstance(raw_attempts, list)
            else []
        )

        attempt_count = len(attempts)
        total_attempt_count += attempt_count
        attempt_count_distribution[
            str(attempt_count)
        ] += 1

        if attempt_count == 0:
            sample_without_attempt_count += 1
        else:
            sample_with_attempt_count += 1

        if attempt_count > 1:
            sample_with_retry_count += 1

        successful_for_sample = 0
        failed_for_sample = 0

        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue

            if attempt.get("success") is True:
                successful_attempt_count += 1
                successful_for_sample += 1
            else:
                failed_attempt_count += 1
                failed_for_sample += 1

                error_text = str(
                    attempt.get("error") or ""
                )
                error_type_counter[
                    _classify_sql_error(error_text)
                ] += 1

        if successful_for_sample == 0:
            sample_without_success_count += 1

        if failed_for_sample > 0:
            sample_with_failed_attempt_count += 1

        if (
            attempt_count > 1
            and session.evaluation_record.outcome
            == EvaluationOutcome.CORRECT
        ):
            correct_after_retry_count += 1

    session_count = len(session_list)

    return {
        "total_attempt_count": (
            total_attempt_count
        ),
        "successful_attempt_count": (
            successful_attempt_count
        ),
        "failed_attempt_count": (
            failed_attempt_count
        ),
        "attempt_success_rate": (
            successful_attempt_count
            / total_attempt_count
            if total_attempt_count > 0
            else 0.0
        ),
        "average_attempt_count_per_sample": (
            total_attempt_count / session_count
            if session_count > 0
            else 0.0
        ),
        "sample_with_attempt_count": (
            sample_with_attempt_count
        ),
        "sample_without_attempt_count": (
            sample_without_attempt_count
        ),
        "sample_without_success_count": (
            sample_without_success_count
        ),
        "sample_with_retry_count": (
            sample_with_retry_count
        ),
        "sample_with_failed_attempt_count": (
            sample_with_failed_attempt_count
        ),
        "correct_after_retry_count": (
            correct_after_retry_count
        ),
        "attempt_count_distribution": dict(
            sorted(
                attempt_count_distribution.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "error_type_count": dict(
            sorted(error_type_counter.items())
        ),
    }

def _calculate_schema_diagnostics(
    session_list: Sequence[Session],
) -> dict[str, Any]:
    attempts = [
        attempt
        for session in session_list
        for attempt in (
            session.task_output or {}
        ).get("schema_attempts", [])
    ]

    successful_count = sum(
        attempt.get("success") is True
        for attempt in attempts
    )

    samples_with_lookup = sum(
        bool(
            (session.task_output or {}).get(
                "schema_attempts"
            )
        )
        for session in session_list
    )

    return {
        "request_count": len(attempts),
        "successful_request_count": successful_count,
        "failed_or_rejected_request_count": (
            len(attempts) - successful_count
        ),
        "sample_with_schema_lookup_count": (
            samples_with_lookup
        ),
        "average_requests_per_sample": (
            len(attempts) / len(session_list)
            if session_list
            else 0.0
        ),
    }


def calculate_enterprise_query_metric(
    task: BenchmarkTask,
    session_list: Sequence[Session],
) -> dict[str, Any]:
    """
    企业查询任务专用评价指标。

    总体和Skill指标沿用DBBench定义，在此基础上增加
    企业查询业务维度与B1 SQL执行诊断。
    """
    base_metric = calculate_dbbench_metric(
        task=task,
        session_list=session_list,
    )

    difficulty_metric = (
        _calculate_dimension_metric(
            task,
            session_list,
            lambda entry: (
                entry.metadata.difficulty.value
            ),
        )
    )

    business_type_metric = (
        _calculate_dimension_metric(
            task,
            session_list,
            lambda entry: (
                entry.metadata.business_query_type
            ),
        )
    )

    task_family_metric = (
        _calculate_dimension_metric(
            task,
            session_list,
            lambda entry: (
                entry.metadata.task_family_id
            ),
        )
    )

    result_type_metric = (
        _calculate_dimension_metric(
            task,
            session_list,
            lambda entry: (
                entry.metadata.result_type.value
            ),
        )
    )

    row_order_metric = (
        _calculate_dimension_metric(
            task,
            session_list,
            lambda entry: (
                entry.answer_info
                .comparison_policy
                .row_order
                .value
            ),
        )
    )

    return {
        "overall": base_metric["overall"],
        "difficulty": difficulty_metric,
        "business_query_type": (
            business_type_metric
        ),
        "task_family": task_family_metric,
        "result_type": result_type_metric,
        "row_order": row_order_metric,
        "skill": base_metric["skill"],
        "sql_diagnostics": (
            _calculate_sql_diagnostics(
                session_list
            )
        ),
        "schema_diagnostics": (
            _calculate_schema_diagnostics(session_list)
        ),
    }