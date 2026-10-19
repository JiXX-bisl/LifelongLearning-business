from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from ..models import CompletedTask, UsageMetrics


@dataclass
class HistoricalRecord:
    """标准任务及其历史来源信息。"""

    task: CompletedTask
    source_sha256: str
    original_source_ref: str | None

    session_id: str
    turn_index: int
    mode: str | None
    experiment_id: str | None
    created_at: str | None
    reference_date: str | None

    # 保留完整成本字段，包括 total_tokens、usage_complete 等。
    original_metrics: dict[str, Any]

    # 原始版本与导入目标分开保存；旧记录没有此字段仍然可读。
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_for_review(self) -> bool:
        """满足候选审核条件，不代表已进入可检索经验池。"""
        return (
            self.task.status == "completed"
            and self.task.evaluation_verdict == "pass"
        )


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缺少有效字段：{key}")
    return value


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _seconds(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def history_from_query_result(
    case: dict,
    record: dict,
    record_path: str | Path,
    query_id: str,
    *,
    namespace: str,
    data_version: str,
) -> HistoricalRecord:
    """从已评分查询的内存结果提取经验；不读取结果文件中的私有标准。"""
    if not namespace.strip() or not data_version.strip():
        raise ValueError("namespace 和 data_version 不能为空")
    question = _text(record, "question")
    if _text(case, "question").strip() != question:
        raise ValueError("经验问题与当前题目不同")
    if record.get("baseline_id") != data_version:
        raise ValueError("查询结果的数据版本不一致")
    session_id = _text(record, "session_id")
    turn_index = record.get("turn_index")
    if type(turn_index) is not int or turn_index < 1:
        raise ValueError("turn_index 必须是正整数")
    status = record.get("status")
    if status not in ("completed", "failed", "limit_reached"):
        raise ValueError("无法识别的运行状态")
    answer = record.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError("answer 必须是字符串")
    history = case.get("history", [])
    if (not isinstance(history, list) or any(
        not isinstance(message, dict) or message.get("role") != "user"
        or not isinstance(message.get("content"), str) or not message["content"].strip()
        for message in history
    )):
        raise ValueError("经验上下文只接受题目的公开用户历史")
    context = [{"role": "user", "content": message["content"]} for message in history]
    operations = record.get("operations")
    if not isinstance(query_id, str) or not query_id.strip() or not isinstance(operations, list):
        raise ValueError("必须指定本题实际 SQL 的 query_id")
    selected = [operation for operation in operations if isinstance(operation, dict)
                and operation.get("kind") == "sql" and operation.get("query_id") == query_id]
    if len(selected) != 1:
        raise ValueError("通过的 query_id 必须唯一绑定实际 SQL")
    actual = selected[0]
    columns, rows = actual.get("columns"), actual.get("rows")
    if (actual.get("success") is not True or actual.get("is_complete") is not True
        or not isinstance(columns, list) or not columns
        or any(not isinstance(column, str) for column in columns)
        or not isinstance(rows, list)
        or any(not isinstance(row, (list, tuple)) or len(row) != len(columns) for row in rows)
        or actual.get("row_count", len(rows)) != len(rows)):
        raise ValueError("经验必须来自成功且完整的实际 SQL 结果")
    operation = {"kind": "sql", "query_id": query_id, "sql": _text(actual, "sql"),
                 "success": True, "is_complete": True,
                 "columns": deepcopy(columns), "rows": deepcopy(rows), "row_count": len(rows)}
    scope = actual.get("scope")
    if isinstance(scope, dict):
        operation["scope"] = {key: scope[key] for key in ("enterprise", "time_range", "source")
                              if isinstance(scope.get(key), str)}
    source = Path(record_path).resolve()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    metrics = record.get("metrics") or {}
    usage = metrics.get("usage") or {}
    normalized = record.get("completed_task_usage") or {
        "elapsed_seconds": metrics.get("task_elapsed_seconds"),
        "model_call_count": metrics.get("model_call_count"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
    task_usage = UsageMetrics(
        elapsed_seconds=_seconds(normalized.get("elapsed_seconds")),
        model_call_count=_count(normalized.get("model_call_count")),
        prompt_tokens=_count(normalized.get("prompt_tokens")),
        completion_tokens=_count(normalized.get("completion_tokens")),
        embedding_input_tokens=_count(normalized.get("embedding_input_tokens")),
    )
    identity = json.dumps([namespace, data_version, session_id, turn_index],
                          ensure_ascii=False, separators=(",", ":"))
    task = CompletedTask(
        task_id="business:" + hashlib.sha256(identity.encode()).hexdigest(),
        question=question, answer=answer, namespace=namespace, data_version=data_version,
        status=cast(Literal["completed", "failed", "limit_reached"], status),
        context_text=json.dumps(context, ensure_ascii=False) if context else "",
        evidence_text=json.dumps([operation], ensure_ascii=False, allow_nan=False),
        source_ref=str(source), usage=task_usage,
    )
    return HistoricalRecord(
        task=task, source_sha256=source_sha256, original_source_ref=str(source),
        session_id=session_id, turn_index=turn_index,
        mode=record.get("mode"), experiment_id=record.get("experiment_id"),
        created_at=record.get("created_at"), reference_date=record.get("reference_date"),
        original_metrics={},
        provenance={"original_data_version": data_version, "target_data_version": data_version,
                    "source_version_known": True},
    )


def read_history_file(
    path: str | Path,
    *,
    namespace: str,
    data_version: str,
) -> HistoricalRecord:
    """转换一份业务日志，不写文件、不评价、不调用模型。"""

    if not namespace.strip() or not data_version.strip():
        raise ValueError("namespace 和 data_version 不能为空")

    source = Path(path).resolve()
    content = source.read_bytes()
    raw = json.loads(content)

    if not isinstance(raw, dict):
        raise ValueError("运行日志必须是 JSON 对象")

    original_version = raw.get("baseline_id")
    if original_version is not None and not isinstance(original_version, str):
        raise ValueError("baseline_id 必须是字符串或 null")
    original_version = (original_version or "").strip() or None
    if original_version is not None and original_version != data_version:
        raise ValueError(f"数据版本不一致：{original_version}")
    baseline_id = original_version or data_version

    session_id = _text(raw, "session_id")
    turn_index = raw.get("turn_index")
    if type(turn_index) is not int or turn_index < 1:
        raise ValueError("turn_index 必须是正整数")

    question = _text(raw, "question")
    answer = raw.get("answer")
    if not isinstance(answer, str):
        raise ValueError("answer 必须是字符串")

    status = raw.get("status")
    if status not in ("completed", "failed", "limit_reached"):
        raise ValueError(f"无法识别的运行状态：{status}")

    evaluation = raw.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation 格式不正确")

    verdict = evaluation.get("verdict", "unknown")
    if verdict not in ("pass", "revise", "unknown"):
        raise ValueError(f"无法识别的评价结果：{verdict}")

    messages = raw.get("messages")
    if not isinstance(messages, list) or any(
        not isinstance(message, dict) for message in messages
    ):
        raise ValueError("messages 必须是消息对象列表")

    # 新记录保存任务开始时的边界，允许同一会话重复完全相同的问题。
    # 旧记录仍按问题文字定位，无法唯一定位时继续要求人工核查。
    if "context_message_count" in raw:
        boundary = raw["context_message_count"]
        if (type(boundary) is not int or not 0 <= boundary < len(messages)
                or messages[boundary].get("role") != "user"
                or messages[boundary].get("content") != question):
            raise ValueError("context_message_count 与当前问题位置不一致")
    else:
        positions = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user"
            and message.get("content") == question
        ]
        if len(positions) != 1:
            raise ValueError("无法唯一确定当前问题的位置，需要单独核查上下文")
        boundary = positions[0]

    context = messages[:boundary]

    operations = raw.get("operations")
    if not isinstance(operations, list):
        raise ValueError("operations 必须是列表")

    metrics = raw.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise ValueError("metrics 格式不正确")

    usage = metrics.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("metrics.usage 格式不正确")
    # 新适配日志提供包含PSU的任务成本；旧日志仍按原metrics读取。
    completed_usage = raw.get("completed_task_usage")
    if completed_usage is not None and not isinstance(completed_usage, dict):
        raise ValueError("completed_task_usage 必须是对象")
    normalized_usage = completed_usage if completed_usage is not None else {
        "elapsed_seconds": metrics.get("task_elapsed_seconds"),
        "model_call_count": metrics.get("model_call_count"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "embedding_input_tokens": None,
    }

    # 身份不依赖文件位置；再次迁移目录不会改变执行 ID。
    identity = json.dumps(
        [namespace, baseline_id, session_id, turn_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    execution_id = (
        "business:" + hashlib.sha256(identity.encode()).hexdigest()
    )

    task = CompletedTask(
        task_id=execution_id,
        question=question,
        answer=answer,
        namespace=namespace,
        data_version=baseline_id,
        status=cast(
            Literal["completed", "failed", "limit_reached"],
            status,
        ),
        context_text=(
            json.dumps(context, ensure_ascii=False)
            if context else ""
        ),
        evidence_text=json.dumps(operations, ensure_ascii=False),
        evaluation_verdict=cast(
            Literal["pass", "revise", "unknown"],
            verdict,
        ),
        quality_scores={},
        source_ref=str(source),
        usage=UsageMetrics(
            elapsed_seconds=_seconds(
                normalized_usage.get("elapsed_seconds")
            ),
            model_call_count=_count(
                normalized_usage.get("model_call_count")
            ),
            prompt_tokens=_count(normalized_usage.get("prompt_tokens")),
            completion_tokens=_count(normalized_usage.get("completion_tokens")),
            embedding_input_tokens=_count(normalized_usage.get("embedding_input_tokens")),
        ),
    )

    return HistoricalRecord(
        task=task,
        source_sha256=hashlib.sha256(content).hexdigest(),
        original_source_ref=raw.get("record_path"),
        session_id=session_id,
        turn_index=turn_index,
        mode=raw.get("mode"),
        experiment_id=raw.get("experiment_id"),
        created_at=raw.get("created_at"),
        reference_date=raw.get("reference_date"),
        original_metrics=metrics,
        provenance={
            "original_data_version": original_version,
            "target_data_version": data_version,
            "source_version_known": original_version is not None,
        },
    )
