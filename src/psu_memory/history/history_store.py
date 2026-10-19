from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .history_adapter import HistoricalRecord


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS history_records (
        execution_id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        data_version TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        record_json TEXT NOT NULL,
        import_state TEXT NOT NULL CHECK (
            import_state IN ('pending_review', 'held', 'conflict')
        ),
        imported_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS source_locations (
        execution_id TEXT NOT NULL REFERENCES history_records(execution_id),
        source_ref TEXT NOT NULL,
        source_sha256 TEXT NOT NULL,
        PRIMARY KEY (execution_id, source_ref, source_sha256)
    )""",
    """CREATE TABLE IF NOT EXISTS import_runs (
        run_id TEXT PRIMARY KEY,
        report_json TEXT NOT NULL
    )""",
)


class SQLiteHistoryStore:
    """保存导入档案；不实现在线检索或经验准入。"""

    def __init__(self, path: str | Path, *, record_runs: bool = True) -> None:
        self.path = Path(path).resolve()
        self.record_runs = record_runs

    def import_batch(
        self,
        records: list[HistoricalRecord],
        *,
        namespace: str,
        data_version: str,
        source_root: str,
        scanned_files: int,
        errors: list[dict[str, str]],
        started_at: float,
    ) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        counts = Counter(inserted=0, skipped=0, conflicts=0)
        conflicts = []
        run_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()

        with closing(
            sqlite3.connect(self.path, timeout=30)
        ) as connection:
            connection.execute("PRAGMA foreign_keys = ON")

            with connection:
                # 记录、来源关系、报告在同一个事务中提交。
                connection.execute("BEGIN IMMEDIATE")

                version = connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
                if version not in (0, 1):
                    raise ValueError(f"不支持的存储版本：{version}")

                for statement in SCHEMA:
                    connection.execute(statement)
                connection.execute("PRAGMA user_version = 1")

                for record in records:
                    task = record.task

                    if (
                        task.namespace,
                        task.data_version,
                    ) != (namespace, data_version):
                        raise ValueError(
                            "本批记录的命名空间或数据版本不一致"
                        )

                    if not task.source_ref:
                        raise ValueError("导入记录缺少来源路径")

                    existing = connection.execute(
                        "SELECT source_sha256 FROM history_records "
                        "WHERE execution_id = ?",
                        (task.task_id,),
                    ).fetchone()

                    if existing is None:
                        state = (
                            "pending_review"
                            if record.candidate_for_review
                            else "held"
                        )

                        connection.execute(
                            "INSERT INTO history_records "
                            "VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                task.task_id,
                                namespace,
                                data_version,
                                record.source_sha256,
                                json.dumps(
                                    asdict(record),
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ),
                                state,
                                now,
                            ),
                        )
                        counts["inserted"] += 1

                    elif existing[0] == record.source_sha256:
                        counts["skipped"] += 1

                    else:
                        # 保留原内容；后续重复导入也不会解除冲突。
                        connection.execute(
                            "UPDATE history_records "
                            "SET import_state = 'conflict' "
                            "WHERE execution_id = ?",
                            (task.task_id,),
                        )

                        conflicts.append({
                            "execution_id": task.task_id,
                            "source_ref": task.source_ref,
                            "stored_sha256": existing[0],
                            "incoming_sha256": record.source_sha256,
                        })
                        counts["conflicts"] += 1

                    connection.execute(
                        "INSERT OR IGNORE INTO source_locations "
                        "VALUES (?, ?, ?)",
                        (
                            task.task_id,
                            task.source_ref,
                            record.source_sha256,
                        ),
                    )

                states = dict(connection.execute(
                    "SELECT import_state, COUNT(*) "
                    "FROM history_records "
                    "WHERE namespace = ? AND data_version = ? "
                    "GROUP BY import_state",
                    (namespace, data_version),
                ).fetchall())

                report = {
                    "run_id": run_id,
                    "created_at": now,
                    "status": (
                        "completed_with_issues"
                        if errors or conflicts
                        else "completed"
                    ),
                    "store_path": str(self.path),
                    "source_root": source_root,
                    "namespace": namespace,
                    "data_version": data_version,
                    "scanned_files": scanned_files,
                    "converted_records": len(records),
                    **dict(counts),
                    "errors": errors,
                    "conflict_details": conflicts,
                    "stored_records_in_scope": sum(states.values()),
                    "stored_states": states,
                    "admitted_by_this_import": 0,
                    "usage": {
                        # 从开始读取到报告生成；
                        # 不含最终提交和终端输出。
                        "elapsed_seconds": round(
                            perf_counter() - started_at, 6
                        ),
                        "model_call_count": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "embedding_input_tokens": 0,
                    },
                }

                if self.record_runs:
                    connection.execute(
                        "INSERT INTO import_runs VALUES (?, ?)",
                        (run_id, json.dumps(report, ensure_ascii=False, allow_nan=False)),
                    )

        return report
