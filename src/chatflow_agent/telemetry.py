from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any


TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens"
)


class TaskTelemetry:
    """一次query任务的调用统计"""
    def __init__(self) -> None:
        self.phase = "decision"
        self._started_at = perf_counter()
        self._model_calls: list[dict[str, Any]] = []

    def record_call(
        self, 
        *, 
        success: bool, 
        request_count: int, 
        retry_count: int, 
        elapsed_seconds: float,
        usage: dict[str, Any] | None,
        error_type: str | None = None
    ) -> None:
        raw_usage = usage or {}
        normalized_usage: dict[str, int | None] = {}
        for name in TOKEN_FIELDS:
            value = raw_usage.get(name)
            normalized_usage[name] = (
                value
                if type(value) is int and value >= 0 else None
            )
        self._model_calls.append({
            "call_index": len(self._model_calls) + 1,
            "phase": self.phase,
            "success": success,
            "request_count": request_count,
            "retry_count": retry_count,
            "elapsed_seconds": elapsed_seconds,
            "usage": normalized_usage,
            "error_type": error_type
        })

    def snapshot(self) -> dict[str, Any]:
        calls = deepcopy(self._model_calls)
        usage_totals: dict[str, int | None] = {}
        for name in TOKEN_FIELDS:
            values = [
                call["usage"][name]
                for call in calls
            ]
            usage_totals[name] = (
                sum(values)
                if all(value is not None for value in values) else None
            )
            
        metrics = {
            "model_call_count": len(calls),
            "failed_model_call_count": sum(not call["success"] for call in calls),
            "request_count": sum(call["request_count"] for call in calls),
            "retry_count": sum(call["retry_count"] for call in calls),
            "model_elapsed_seconds": round(sum(call["elapsed_seconds"] for call in calls), 6),
            "task_elapsed_seconds": round(perf_counter() - self._started_at, 6),
            "usage": usage_totals,
            "usage_complete": all(value is not None for value in usage_totals.values())
        }  

        return {
            "model_calls": calls,
            "metrics": metrics
        }
