"""接入与验证共用工具；导入时不连接数据库或初始化模型。"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.project_paths import resolve_project_path
from ..evaluation.quality_models import digest
from ..model_client import read_model_config
from ..models import UsageMetrics


COST_FIELDS = ("model_call_count", "http_request_count", "retry_count",
               "prompt_tokens", "completion_tokens", "embedding_input_tokens")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    """沿用原子报告写入；不用于重写原始业务记录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def zero_cost(elapsed_seconds=0.0):
    return {**asdict(UsageMetrics(elapsed_seconds, 0, 0, 0, 0)),
            "http_request_count": 0, "retry_count": 0,
            "token_usage_complete": True}


def add_costs(parts):
    """沿用已有汇总规则；未知值保留未知，已报告部分另列。"""
    parts = list(parts)
    result = {}
    for field in (*COST_FIELDS, "elapsed_seconds"):
        values = [part.get(field) for part in parts]
        result[field] = None if any(v is None for v in values) else sum(values)
        result["reported_" + field] = sum(
            part.get("reported_" + field, part.get(field) or 0) for part in parts)
    result["token_usage_complete"] = all(
        part.get("token_usage_complete", False) for part in parts
    ) and result["prompt_tokens"] is not None and result["completion_tokens"] is not None
    result["total_llm_tokens"] = (
        result["prompt_tokens"] + result["completion_tokens"]
        if result["token_usage_complete"] else None
    )
    return result


def business_cost(metrics):
    usage = metrics.get("usage", {})
    return {
        "elapsed_seconds": metrics.get("task_elapsed_seconds"),
        "model_call_count": metrics.get("model_call_count"),
        "http_request_count": metrics.get("request_count"),
        "retry_count": metrics.get("retry_count"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "embedding_input_tokens": 0,
        "token_usage_complete": metrics.get("usage_complete", False),
    }


def copy_database(source, target):
    """复用已有 SQLite backup 操作；仅在调用时复制，拒绝覆盖目标。"""
    target = Path(target)
    if target.exists():
        raise ValueError("拒绝覆盖已有经验库")
    with closing(sqlite3.connect(Path(source).as_uri() + "?mode=ro", uri=True)) as origin:
        with closing(sqlite3.connect(target)) as destination:
            origin.backup(destination)


def model_options(language_model, config_path):
    """复用本地验证适配的模型参数应用与脱敏签名。"""
    options = read_model_config(resolve_project_path(config_path))
    for name in ("temperature", "max_tokens", "timeout", "max_retries",
                 "request_interval_seconds", "extra_body", "reject_truncated"):
        if name in options:
            setattr(language_model, name, deepcopy(options[name]))
    effective = {
        name: deepcopy(getattr(language_model, name))
        for name in ("model_name", "temperature", "max_tokens", "timeout",
                     "max_retries", "request_interval_seconds", "extra_body")
    }
    effective["endpoint_sha256"] = digest(language_model.base_url or "SDK_DEFAULT")
    return effective


def query_metadata(strategy, result):
    """沿用现有任务成本与数据库计时字段，不累计历史检索成本两次。"""
    metadata = getattr(strategy, "metadata", None)
    psu = metadata() if callable(metadata) else {
        "usage": zero_cost(), "used_experience_ids": [], "prompt_events": [],
        "base_prompt": strategy.agent.system_prompt,
        "final_prompt": strategy.agent.system_prompt,
    }
    completed = add_costs([business_cost(result["metrics"]), psu["usage"]])
    # 外层任务时间已包含增强，不能再相加检索时间。
    completed["elapsed_seconds"] = result["metrics"]["task_elapsed_seconds"]
    return {
        "used_experience_ids": psu["used_experience_ids"],
        "completed_task_usage": {key: completed[key] for key in
                                 ("elapsed_seconds", "model_call_count", "prompt_tokens",
                                  "completion_tokens", "embedding_input_tokens")},
        "psu": psu,
        "database_timing": {
            "call_count": sum(o.get("kind") == "sql" for o in result["operations"]),
            "elapsed_seconds": sum(o.get("elapsed_seconds", 0.0)
                                   for o in result["operations"] if o.get("kind") == "sql"),
        },
    }
