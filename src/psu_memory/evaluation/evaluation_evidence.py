from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from dataclasses import asdict

from ..history.history_adapter import HistoricalRecord, read_history_file
from ..history.source_checks import inspect_source, source_aliases


SIGNALS = (
    "reference_result_match",
    "independent_review",
    "user_feedback",
)
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS evaluation_versions (
        execution_id TEXT NOT NULL REFERENCES history_records(execution_id),
        profile_sha256 TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (execution_id, profile_sha256)
    )""",
    """CREATE TABLE IF NOT EXISTS evaluation_current (
        execution_id TEXT PRIMARY KEY,
        profile_sha256 TEXT NOT NULL,
        FOREIGN KEY (execution_id, profile_sha256)
          REFERENCES evaluation_versions(execution_id, profile_sha256)
    )""",
    """CREATE TABLE IF NOT EXISTS evaluation_runs (
        run_id TEXT PRIMARY KEY,
        report_json TEXT NOT NULL
    )""",
)


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是对象")
    return value


def _read_signals(path, root):
    """返回一个评价文件的全部条目；文件有误时整份不采用。"""
    content = path.read_bytes()
    document = _object(json.loads(content), "评价文件")
    acceptance = path.name == "acceptance.json"
    items = document.get("results") if acceptance else [document]
    if not isinstance(items, list):
        raise ValueError("results 必须是列表")
    evidence = []

    for index, item in enumerate(items):
        item = _object(item, "评价条目")
        if acceptance:
            settings = _object(document.get("settings", {}), "settings")
            data = _object(settings.get("data", {}), "settings.data")
            learning = _object(settings.get("learning", {}), "settings.learning")
            expected = {
                "baseline_id": data.get("baseline_id"),
                "mode": learning.get("mode"),
                "experiment_id": learning.get("experiment_id"),
            }
            b1 = _object(item.get("b1", {}), "b1")
            matched = b1.get("matched")
            if matched is not None and type(matched) is not bool:
                raise ValueError("b1.matched 必须是布尔值或 null")
            if b1.get("reason") == "no_successful_sql":
                if matched is True:
                    raise ValueError("没有成功 SQL 却标记为一致")
                matched = None
            review = _object(item.get("independent_review", {}), "independent_review")
            verdict = review.get("verdict")
            if verdict is not None and not isinstance(verdict, str):
                raise ValueError("独立审核 verdict 必须是字符串或 null")
            signals = {
                "reference_result_match": matched,
                "independent_review": (
                    None if verdict in (None, "", "pending", "unknown") else verdict
                ),
            }
            record_ref = item.get("record_path")
            details = {
                "legacy_metric": "B1",
                "legacy_b1": b1,
                "independent_review": review,
                "dataset_id": document.get("dataset_id"),
                "benchmark_task_id": item.get("task_id"),
            }
        else:
            rating = item.get("rating")
            if rating not in ("helpful", "problem"):
                raise ValueError("反馈 rating 必须是 helpful 或 problem")
            expected = {key: item.get(key) for key in (
                "baseline_id", "session_id", "turn_index", "mode", "experiment_id"
            )}
            signals = {"user_feedback": rating}
            record_ref = item.get("source_record_path")
            details = {
                "comment": item.get("comment"),
                "updated_at": item.get("updated_at"),
            }
        if not isinstance(record_ref, str) or not record_ref.strip():
            raise ValueError("缺少被评价日志的来源路径")
        evidence.append({
            "source_kind": "acceptance" if acceptance else "feedback",
            "source_ref": path.relative_to(root).as_posix(),
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "source_pointer": f"/results/{index}" if acceptance else "",
            "original_record_ref": record_ref,
            "expected_identity": expected,
            "signals": signals,
            "details": details,
        })
    return evidence


def _profile(execution_id, record, evidence, root):
    summary = {}
    conflicts = {}
    for name in SIGNALS:
        values = {
            json.dumps(e["signals"][name], ensure_ascii=False)
            for e in evidence if e["signals"].get(name) is not None
        }
        summary[name] = json.loads(next(iter(values))) if len(values) == 1 else None
        if len(values) > 1:
            conflicts[name] = [json.loads(v) for v in sorted(values)]
    internal = record["task"]["evaluation_verdict"]
    hints = []
    if internal == "pass" and summary["reference_result_match"] is False:
        hints.append("internal_pass_with_reference_mismatch")
    if internal == "pass" and summary["user_feedback"] == "problem":
        hints.append("internal_pass_with_negative_feedback")
    return {
        "schema_version": 1,
        "execution_id": execution_id,
        "source_root": str(root),
        "internal_evaluation_verdict": internal,
        **summary,
        "signal_conflicts": conflicts,
        "review_hints": hints,
        "evidence": evidence,
    }


def attach_sql_result_pass(store_path, historical: HistoricalRecord, query_id: str):
    """保存调用方已直接比较通过的 SQL 证据，不导入标准值或生成审核文件。"""
    task = historical.task
    operations = json.loads(task.evidence_text)
    matches = [operation for operation in operations
               if operation.get("kind") == "sql" and operation.get("query_id") == query_id]
    if (len(matches) != 1 or matches[0].get("success") is not True
            or matches[0].get("is_complete") is not True):
        raise ValueError("通过证据必须绑定唯一、成功且完整的实际 SQL")
    with closing(sqlite3.connect(Path(store_path).resolve().as_uri() + "?mode=rw",
                                uri=True, timeout=30)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json, source_sha256, import_state FROM history_records "
                "WHERE execution_id=? AND namespace=? AND data_version=?",
                (task.task_id, task.namespace, task.data_version),
            ).fetchone()
            if (row is None or row[1] != historical.source_sha256 or row[2] == "conflict"
                    or json.loads(row[0])["task"] != asdict(task)):
                raise ValueError("通过证据与本次已保存经验不一致")
            record = json.loads(row[0])
            source = inspect_source(record, row[1], row[2], lambda value: Path(value))
            if source["status"] != "verified":
                raise ValueError("单题结果文件缺失或已变化")
            evidence = [{
                "source_kind": "direct_sql_result",
                "source_ref": task.source_ref,
                "source_sha256": historical.source_sha256,
                "signals": {"reference_result_match": True, "independent_review": "pass"},
                "details": {
                    "legacy_b1": {"query_id": query_id},
                    "independent_review": {
                        "scope": "sql", "query_id": query_id, "verdict": "pass",
                        "admission_policy": "sql_result_direct_v1", "actor": "direct-sql-result-v1",
                    },
                },
            }]
            profile = _profile(task.task_id, record, evidence, Path(task.source_ref).parent)
            payload = json.dumps(profile, ensure_ascii=False, sort_keys=True, allow_nan=False)
            fingerprint = hashlib.sha256(payload.encode()).hexdigest()
            for statement in SCHEMA:
                connection.execute(statement)
            previous = connection.execute(
                "SELECT profile_sha256 FROM evaluation_current WHERE execution_id=?", (task.task_id,),
            ).fetchone()
            if previous is not None and previous[0] != fingerprint:
                raise ValueError("本题已有不同的外部评价，不能覆盖")
            connection.execute("INSERT OR IGNORE INTO evaluation_versions VALUES (?, ?, ?, ?)",
                               (task.task_id, fingerprint, payload, datetime.now(timezone.utc).isoformat()))
            connection.execute("INSERT OR IGNORE INTO evaluation_current VALUES (?, ?)",
                               (task.task_id, fingerprint))


def attach_evaluations(
    store_path, source_root, *, namespace, data_version, resolve_path,
    execution_ids=None, record_runs=True,
):
    """全量刷新当前范围的评价档案；不修改历史记录或准入状态。"""
    started = perf_counter()
    store_path = Path(store_path).resolve()
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise ValueError(f"日志根目录不存在：{root}")
    paths = sorted(set(root.glob("**/acceptance.json")) | set(
        root.glob("**/feedback/*/*.json")
    ))
    if not paths:
        raise ValueError("没有找到验收或反馈文件，请检查完整日志根目录")
    errors, unmatched, warnings = [], [], []
    linked = inserted = refreshed = 0
    now = datetime.now(timezone.utc).isoformat()

    # mode=rw 要求候选库已存在，防止拼错路径后创建一个空库。
    with closing(sqlite3.connect(
        store_path.as_uri() + "?mode=rw", uri=True, timeout=30
    )) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("请先建立兼容的历史候选库")
            rows = connection.execute(
                "SELECT execution_id, source_sha256, record_json, import_state "
                "FROM history_records WHERE namespace = ? AND data_version = ?",
                (namespace, data_version),
            ).fetchall()
            # 局部接入时只刷新指定执行，避免新实验覆盖已有历史评价档案。
            if execution_ids is not None:
                selected = set(execution_ids)
                rows = [row for row in rows if row[0] in selected]
                if {row[0] for row in rows} != selected:
                    raise ValueError("指定执行不属于当前候选范围")
            if not rows:
                raise ValueError("当前范围没有已导入记录")
            records = {row[0]: json.loads(row[2]) for row in rows}
            hashes = {row[0]: row[1] for row in rows}
            grouped = {key: [] for key in records}
            aliases, source_states = {}, {}
            for row in rows:
                key = row[0]
                locations = [r[0] for r in connection.execute(
                    "SELECT source_ref FROM source_locations WHERE execution_id=? AND source_sha256=?",
                    (key, hashes[key]),
                )]
                source_states[key] = inspect_source(records[key], hashes[key], row[3], resolve_path, locations)
                if source_states[key]["status"] == "conflict":
                    errors.append({"execution_id": key, "reason": "候选记录存在内容或版本冲突"})
                for warning in source_states[key]["warnings"]:
                    warnings.append({"execution_id": key, "reason": warning})
                for alias in source_aliases(records[key], locations, resolve_path):
                    aliases.setdefault(alias, set()).add(key)

            for path in paths:
                try:
                    items = _read_signals(path, root)
                except (OSError, ValueError) as error:
                    errors.append({"source_ref": str(path), "reason": str(error)})
                    continue
                for item in items:
                    location = {"source_ref": str(path), "pointer": item["source_pointer"]}
                    try:
                        target = Path(resolve_path(item["original_record_ref"])).resolve()
                        candidates = aliases.get(item["original_record_ref"], set()) | aliases.get(str(target), set())
                        if len(candidates) > 1:
                            raise ValueError("来源路径关联到多条执行，无法唯一匹配")
                        if candidates:
                            key = next(iter(candidates))
                        else:
                            # 尚未登记路径时才读取原文件定位；不按问题文字匹配。
                            try:
                                history = read_history_file(target, namespace=namespace, data_version=data_version)
                            except FileNotFoundError:
                                unmatched.append({**location, "reason": "原路径失效且没有已登记关联"})
                                continue
                            key = history.task.task_id
                            if key in hashes and history.source_sha256 != hashes[key]:
                                raise ValueError("来源日志已变化，请先核查历史导入冲突")
                        if key not in records:
                            unmatched.append({**location, "reason": "该执行尚未导入当前范围"})
                            continue
                        history = records[key]
                        actual = {
                            "baseline_id": history["task"]["data_version"],
                            "session_id": history["session_id"],
                            "turn_index": history["turn_index"],
                            "mode": history.get("mode"),
                            "experiment_id": history.get("experiment_id"),
                        }
                        for field, value in item["expected_identity"].items():
                            if value is not None and value != actual[field]:
                                raise ValueError(f"评价与原日志的 {field} 不一致")
                        grouped[key].append(item)
                        linked += 1
                    except (OSError, ValueError) as error:
                        errors.append({**location, "reason": str(error)})

            profiles = {
                key: _profile(key, records[key], grouped[key], root)
                for key in sorted(records)
            }
            for statement in SCHEMA:
                connection.execute(statement)
            # 来源校验有错误时，仅保存报告，保留所有旧的当前档案。
            if not errors:
                for key, profile in profiles.items():
                    payload = json.dumps(
                        profile, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
                    digest = hashlib.sha256(payload.encode()).hexdigest()
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO evaluation_versions VALUES (?, ?, ?, ?)",
                        (key, digest, payload, now),
                    )
                    inserted += cursor.rowcount
                    old = connection.execute(
                        "SELECT profile_sha256 FROM evaluation_current WHERE execution_id = ?",
                        (key,),
                    ).fetchone()
                    if old is None or old[0] != digest:
                        connection.execute(
                            "INSERT INTO evaluation_current VALUES (?, ?) "
                            "ON CONFLICT(execution_id) DO UPDATE "
                            "SET profile_sha256 = excluded.profile_sha256",
                            (key, digest),
                        )
                        refreshed += 1

            conflicts = sum(bool(p["signal_conflicts"]) for p in profiles.values())
            report = {
                "run_id": uuid4().hex,
                "created_at": now,
                "namespace": namespace,
                "data_version": data_version,
                "source_root": str(root),
                "status": "blocked" if errors else (
                    "completed_with_issues" if unmatched or conflicts else "completed"
                ),
                "scanned_evaluation_files": len(paths),
                "records_in_scope": len(records),
                "linked_source_items": linked,
                "records_without_external_evidence": sum(not e for e in grouped.values()),
                "new_profile_versions": inserted,
                "refreshed_current_profiles": refreshed,
                "signal_conflict_records": conflicts,
                "review_hint_records": sum(bool(p["review_hints"]) for p in profiles.values()),
                "unmatched": unmatched,
                "errors": errors,
                "warnings": warnings,
                "admission_changes": 0,
                "usage": {
                    # 统计至报告生成，不含最终提交和终端输出。
                    "elapsed_seconds": round(perf_counter() - started, 6),
                    "model_call_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "embedding_input_tokens": 0,
                },
            }
            if record_runs:
                connection.execute(
                    "INSERT INTO evaluation_runs VALUES (?, ?)",
                    (report["run_id"], json.dumps(report, ensure_ascii=False)),
                )
    return report
