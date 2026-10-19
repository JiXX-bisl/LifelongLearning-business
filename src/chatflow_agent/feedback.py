from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.chatflow_agent.runtime import BusinessRuntimeError
from src.project_paths import resolve_project_path


def save_feedback(
    result: dict[str, Any],
    *,
    rating: str,
    comment: str = "",
) -> Path:
    """保存或更新本次任务的人工反馈，不修改任务与经验。"""
    if rating not in {"helpful", "problem"}:
        raise BusinessRuntimeError("请选择“有帮助”或“有问题”。")

    record_path = resolve_project_path(result["record_path"])

    # 使用业务层产生的会话路径，不接受页面输入的保存路径。
    # .../<模式>/sessions/<会话ID>/<提问序号>.json
    if not record_path.is_file():
        raise BusinessRuntimeError("原会话文件不存在，无法关联反馈。")

    feedback_path = (
        record_path.parents[2]
        / "feedback"
        / record_path.parent.name
        / record_path.name
    )

    payload = {
        "session_id": result["session_id"],
        "turn_index": result["turn_index"],
        "baseline_id": result["baseline_id"],
        "experiment_id": result["experiment_id"],
        "mode": result["mode"],
        "source_record_path": result["record_path"],
        "source": "user_feedback",
        "rating": rating,
        "comment": comment.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        raise BusinessRuntimeError(
            "反馈保存失败，请检查目录权限和磁盘空间。"
        ) from None

    return feedback_path
