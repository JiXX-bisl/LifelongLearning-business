"""SQLite 协调层：版本化评价、批量排序与原子反馈。"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .quality_models import Feedback, POLICY_VERSION, canonical, digest, ranking_key
from .quality_policy import evaluate_candidate, original_content
from ..history.source_checks import inspect_source
from ..pool.pool_store import invalidate_pool_members


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS quality_contents (
        execution_id TEXT NOT NULL REFERENCES history_records(execution_id),
        content_version TEXT NOT NULL, content_json TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY(execution_id, content_version))""",
    """CREATE TABLE IF NOT EXISTS quality_heads (
        execution_id TEXT PRIMARY KEY, content_version TEXT NOT NULL,
        FOREIGN KEY(execution_id, content_version)
          REFERENCES quality_contents(execution_id, content_version))""",
    """CREATE TABLE IF NOT EXISTS quality_feedback (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
        execution_id TEXT NOT NULL, target_version TEXT NOT NULL,
        payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
        FOREIGN KEY(execution_id, target_version)
          REFERENCES quality_contents(execution_id, content_version))""",
    """CREATE TABLE IF NOT EXISTS quality_versions (
        execution_id TEXT NOT NULL, content_version TEXT NOT NULL, decision_sha256 TEXT NOT NULL,
        decision_json TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY(execution_id, content_version, decision_sha256),
        FOREIGN KEY(execution_id, content_version)
          REFERENCES quality_contents(execution_id, content_version))""",
    """CREATE TABLE IF NOT EXISTS quality_current (
        execution_id TEXT PRIMARY KEY, content_version TEXT NOT NULL, decision_sha256 TEXT NOT NULL,
        FOREIGN KEY(execution_id, content_version, decision_sha256)
          REFERENCES quality_versions(execution_id, content_version, decision_sha256))""",
    """CREATE TABLE IF NOT EXISTS quality_runs (
        run_id TEXT PRIMARY KEY, report_json TEXT NOT NULL)""",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _usage(started):
    # 截至报告生成；不含最终事务提交、终端输出。
    return dict(elapsed_seconds=round(perf_counter() - started, 6), model_call_count=0,
                prompt_tokens=0, completion_tokens=0, embedding_input_tokens=0)


class SQLiteQualityStore:
    def __init__(self, path, *, resolve_path=None, record_runs=True):
        self.path = Path(path).resolve()
        self.resolve_path = resolve_path or (lambda value: Path(value).expanduser().resolve())
        self.record_runs = record_runs

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=30)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise ValueError("请先建立兼容的历史候选库")
                for statement in SCHEMA:
                    connection.execute(statement)
                yield connection
        finally:
            connection.close()

    def initialize(self) -> dict:
        """建立质量组件表；空候选库无需生成评价结论或调用模型。"""
        started = perf_counter()
        with self._transaction():
            pass
        return {
            "operation": "initialize_quality",
            "store_path": str(self.path),
            "initialized": True,
            "usage": _usage(started),
        }

    def _load(self, c, execution_id):
        row = c.execute("SELECT record_json, source_sha256, import_state FROM history_records "
                        "WHERE execution_id = ?", (execution_id,)).fetchone()
        if row is None:
            raise ValueError("经验不存在")
        record = json.loads(row[0])
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        profile, evidence_version = {}, None
        if "evaluation_current" in tables:
            evidence = c.execute(
                "SELECT v.profile_json, v.profile_sha256 FROM evaluation_current e "
                "JOIN evaluation_versions v ON v.execution_id=e.execution_id "
                "AND v.profile_sha256=e.profile_sha256 WHERE e.execution_id=?", (execution_id,)
            ).fetchone()
            if evidence:
                profile, evidence_version = json.loads(evidence[0]), evidence[1]
        locations = [r[0] for r in c.execute(
            "SELECT source_ref FROM source_locations WHERE execution_id=? AND source_sha256=?",
            (execution_id, row[1]))]
        source = inspect_source(record, row[1], row[2], self.resolve_path, locations)
        return record, profile, evidence_version, source

    def _put_content(self, c, content):
        version = digest(content)
        c.execute("INSERT OR IGNORE INTO quality_contents VALUES (?, ?, ?, ?)",
                  (content["execution_id"], version, canonical(content), _now()))
        return version

    def _content(self, c, execution_id, version):
        row = c.execute("SELECT content_json FROM quality_contents WHERE execution_id=? "
                        "AND content_version=?", (execution_id, version)).fetchone()
        if row is None:
            raise ValueError("目标内容版本不存在")
        return json.loads(row[0])

    def _ensure_head(self, c, record, profile):
        key = record["task"]["task_id"]
        row = c.execute("SELECT content_version FROM quality_heads WHERE execution_id=?", (key,)).fetchone()
        if row:
            return row[0]
        version = self._put_content(c, original_content(record, profile))
        c.execute("INSERT INTO quality_heads VALUES (?, ?)", (key, version))
        return version

    def _events(self, c, key):
        return [dict(json.loads(raw), event_id=event_id) for event_id, raw in c.execute(
            "SELECT event_id, payload_json FROM quality_feedback WHERE execution_id=? ORDER BY sequence", (key,))]

    def _insert_event(self, c, feedback):
        payload = feedback.payload()
        event_id = feedback.event_id or "feedback:" + digest(payload)
        old = c.execute("SELECT payload_json FROM quality_feedback WHERE event_id=?", (event_id,)).fetchone()
        if old:
            if old[0] != canonical(payload):
                raise ValueError("相同 event_id 对应不同反馈内容")
            return event_id, False
        c.execute("INSERT INTO quality_feedback(event_id,execution_id,target_version,payload_json,created_at) "
                  "VALUES (?, ?, ?, ?, ?)", (event_id, feedback.execution_id, feedback.target_version,
                                             canonical(payload), _now()))
        return event_id, True

    def _sync_legacy_feedback(self, c, key):
        # 扫描已保存的评价版本，避免此前的负面反馈因文件被覆盖而丢失。
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "evaluation_versions" not in tables:
            return
        original = next(json.loads(r[0]) for r in c.execute(
            "SELECT content_json FROM quality_contents WHERE execution_id=? ORDER BY created_at,content_version",
            (key,)) if json.loads(r[0])["kind"] == "original")
        base_version = digest(original)
        for payload, in c.execute("SELECT profile_json FROM evaluation_versions WHERE execution_id=? "
                                  "ORDER BY created_at,profile_sha256", (key,)).fetchall():
            profile = json.loads(payload)
            for evidence in profile.get("evidence", []):
                if evidence.get("source_kind") != "feedback":
                    continue
                rating = evidence.get("signals", {}).get("user_feedback")
                if rating not in ("helpful", "problem"):
                    continue
                details = evidence.get("details", {})
                item = Feedback(key, base_version, rating,
                                comment=details.get("comment") or "", actor="legacy_feedback",
                                event_id="legacy:" + digest([key, evidence]))
                self._insert_event(c, item)
            # 兼容只有汇总字段的旧评价，仍保留负面反馈的停用效力。
            if profile.get("user_feedback") in ("helpful", "problem") and not any(
                e.get("source_kind") == "feedback" for e in profile.get("evidence", [])
            ):
                self._insert_event(c, Feedback(key, base_version, profile["user_feedback"],
                    actor="legacy_feedback", event_id="legacy-summary:" + digest([key, profile])))

    def _save_decision(self, c, decision, *, current=True):
        payload = decision.to_dict()
        fingerprint = digest(payload)

        inserted = c.execute(
            "INSERT OR IGNORE INTO quality_versions VALUES (?, ?, ?, ?, ?)",
            (
                decision.execution_id,
                decision.content_version,
                fingerprint,
                canonical(payload),
                _now(),
            ),
        ).rowcount

        old = c.execute(
            """
            SELECT content_version, decision_sha256
            FROM quality_current WHERE execution_id=?
            """,
            (decision.execution_id,),
        ).fetchone()

        changed = current and old != (
            decision.content_version,
            fingerprint,
        )

        if changed:
            c.execute(
                """
                INSERT INTO quality_current VALUES (?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    content_version=excluded.content_version,
                    decision_sha256=excluded.decision_sha256
                """,
                (
                    decision.execution_id,
                    decision.content_version,
                    fingerprint,
                ),
            )

        pool_removed = 0
        if current:
            eligible = decision.admission_state == "eligible"
            pool_removed = invalidate_pool_members(
                c,
                decision.execution_id,
                keep_version=decision.content_version if eligible else None,
                reason=(
                    "content_version_replaced"
                    if eligible else "admission_suspended"
                ),
                record_runs=self.record_runs,
            )

        return inserted, int(changed), pool_removed

    def _evaluate(self, c, loaded, version):
        record, profile, evidence_version, source = loaded
        key = record["task"]["task_id"]
        return evaluate_candidate(record, profile, self._content(c, key, version), source=source,
                                  evidence_version=evidence_version, feedback=self._events(c, key))

    def assess_batch(self, *, namespace, data_version, execution_ids=None):
        started = perf_counter()
        with self._transaction() as c:
            keys = [r[0] for r in c.execute("SELECT execution_id FROM history_records "
                "WHERE namespace=? AND data_version=? ORDER BY execution_id", (namespace, data_version))]
            if execution_ids is not None:
                requested = set(execution_ids)
                if not requested or not requested <= set(keys):
                    raise ValueError("指定执行不属于当前候选范围")
                keys = [key for key in keys if key in requested]
            if not keys:
                raise ValueError("指定范围没有候选记录")
            decisions, inserted, refreshed, pool_removed = [], 0, 0, 0
            for key in keys:
                loaded = self._load(c, key)
                version = self._ensure_head(c, loaded[0], loaded[1])
                self._sync_legacy_feedback(c, key)
                decision = self._evaluate(c, loaded, version)
                counts = self._save_decision(c, decision)
                inserted += counts[0]
                refreshed += counts[1]
                pool_removed += counts[2]
                decisions.append(decision)
            ranked = sorted((d for d in decisions if d.admission_state == "eligible"), key=ranking_key)
            report = dict(
                run_id=uuid4().hex, operation="assess", policy_version=POLICY_VERSION,
                namespace=namespace, data_version=data_version, records_in_scope=len(keys),
                states={name: sum(d.admission_state == name for d in decisions)
                        for name in ("eligible", "suspended")},
                reason_counts=dict(sorted(Counter(r for d in decisions for r in d.reasons).items())),
                source_status_counts=dict(Counter(d.source["status"] for d in decisions)),
                new_quality_versions=inserted, refreshed_current_decisions=refreshed,
                ranked_execution_ids=[d.execution_id for d in ranked],
                decisions=[d.to_dict() for d in decisions],
                admission_changes=pool_removed,
                pool_removed_count=pool_removed,
                usage=_usage(started),
                created_at=_now(),
            )
            if self.record_runs:
                c.execute("INSERT INTO quality_runs VALUES (?, ?)", (report["run_id"], canonical(report)))
        return report

    def submit_feedback(self, feedback: Feedback):
        feedback.validate()
        started = perf_counter()
        with self._transaction() as c:
            loaded = self._load(c, feedback.execution_id)
            head = self._ensure_head(c, loaded[0], loaded[1])
            self._sync_legacy_feedback(c, feedback.execution_id)
            event_id = feedback.event_id or "feedback:" + digest(feedback.payload())
            old_event = c.execute("SELECT payload_json FROM quality_feedback WHERE event_id=?", (event_id,)).fetchone()
            if old_event and old_event[0] != canonical(feedback.payload()):
                raise ValueError("相同 event_id 对应不同反馈内容")
            if not old_event and head != feedback.target_version:
                raise ValueError("目标版本已不是当前版本，请读取最新版本后再提交")
            target = self._content(c, feedback.execution_id, feedback.target_version)
            if feedback.confirmed and feedback.corrected_sql.strip() == target["sql"].strip():
                raise ValueError("修正 SQL 与旧 SQL 相同；不能用相同内容解除已知错误")
            event_id, inserted = self._insert_event(c, feedback)
            pool_removed = 0
            if inserted:
                old_decision = self._evaluate(c, loaded, head)
                pool_removed += self._save_decision(c, old_decision)[2]
                if feedback.confirmed:
                    content = dict(
                        execution_id=feedback.execution_id, kind="correction", parent_version=head,
                        question=target["question"], context_text=target["context_text"],
                        context_available=target["context_available"], sql=feedback.corrected_sql.strip(),
                        operation={}, confirmed_by_user=True, confirmation_event_id=event_id,
                        source_sha256=target["source_sha256"],
                        notes=feedback.notes or old_decision.content.get("notes", ""),
                        applicability=feedback.applicability or old_decision.content.get("applicability", ""),
                        error_reason=feedback.error_reason or feedback.comment or old_decision.content.get("error_reason", ""),
                    )
                    head = self._put_content(c, content)
                    c.execute("UPDATE quality_heads SET content_version=? WHERE execution_id=?",
                              (head, feedback.execution_id))
            decision = self._evaluate(c, loaded, head)
            pool_removed += self._save_decision(c, decision)[2]
            report = dict(run_id=uuid4().hex, operation="feedback", event_id=event_id,
                          replayed=not inserted, decision=decision.to_dict(), admission_changes=pool_removed, 
                          pool_removed_count=pool_removed, usage=_usage(started), created_at=_now())
            if self.record_runs:
                c.execute("INSERT INTO quality_runs VALUES (?, ?)", (report["run_id"], canonical(report)))
        return report
