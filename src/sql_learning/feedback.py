"""Standard turns enter the existing history → quality → pool → index pipeline."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from src.file_lock import lock_exclusive_nonblocking
from src.psu_memory.models import CompletedTask
from src.psu_memory.history.history_adapter import HistoricalRecord
from src.psu_memory.history.history_store import SQLiteHistoryStore
from src.psu_memory.history.source_checks import inspect_source
from src.psu_memory.evaluation.evaluation_evidence import SCHEMA, _profile
from src.psu_memory.evaluation.quality_store import SQLiteQualityStore
from src.psu_memory.pool.pool_models import PoolConfig
from src.psu_memory.pool.pool_store import SQLitePoolStore
from src.psu_memory.pool.maintenance import run_pool_maintenance
from src.psu_memory.retrieval.indexing import OfflineIndexBuilder
from src.psu_memory.integration.direct_update import DirectUpdateError, DirectUpdateIntegrityError


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def history_from_turn(record, source):
    """No fabricated business session log or caller-provided operation identity."""
    operation = []
    if record.get("sql"):
        operation = [{"kind": "sql", "query_id": record["turn_id"] + "_result",
                      "sql": record["sql"], "columns": record["columns"],
                      "rows": record["rows"], "row_count": len(record["rows"]),
                      "success": record["execution_success"],
                      "is_complete": record["is_complete"]}]
    context = list(record.get("context", []))
    if record.get("constraints"):
        context.append({"role": "user", "content": record["constraints"]})
    task = CompletedTask(
        task_id=record["turn_id"], question=record["question"],
        answer=record.get("answer", ""), namespace=record["namespace"],
        data_version=record["data_version"], status=record["status"],
        context_text=_json(context) if context else "",
        evidence_text=_json(operation), source_ref=str(source),
    )
    return HistoricalRecord(
        task=task, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        original_source_ref=str(source), session_id=record["turn_id"], turn_index=1,
        mode="sql_learning_v2", experiment_id=None, created_at=None,
        reference_date=None, original_metrics={},
        provenance={"original_data_version": task.data_version,
                    "target_data_version": task.data_version, "source_version_known": True},
    )


class TurnFeedback:
    """One work store per run; submit once, evaluate/re-evaluate by turn_id.

    All writers using this interface share a process lock. A full SQLite backup
    restores history, quality, pool and index together if any stage fails.
    Immutable turn files remain available for retries and delayed evaluation.
    """

    def __init__(self, retriever, scene, *, pool_config=None):
        self.retriever, self.scene = retriever, scene
        self.path = Path(retriever.index.pool_store.path).resolve()
        self.pool = SQLitePoolStore(self.path, record_runs=False)
        self.config = pool_config or PoolConfig()
        self.history = SQLiteHistoryStore(self.path, record_runs=False)
        self.quality = SQLiteQualityStore(self.path, resolve_path=Path, record_runs=False)
        self.records = self.path.parent / (self.path.stem + "_turns")
        self.records.mkdir(exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS sql_learning_turns ("
                               "turn_id TEXT PRIMARY KEY, source_ref TEXT NOT NULL, "
                               "source_sha256 TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS sql_learning_evaluations ("
                               "turn_id TEXT PRIMARY KEY, verdict TEXT NOT NULL, "
                               "policy_version TEXT NOT NULL)")

    @contextmanager
    def _transaction(self):
        with self.path.with_suffix(self.path.suffix + ".turn.lock").open("a+b") as lock:
            lock_exclusive_nonblocking(lock)
            with closing(sqlite3.connect(":memory:")) as before:
                with closing(sqlite3.connect(self.path)) as source:
                    source.backup(before)
                expected = tuple(before.iterdump())
                try:
                    yield
                except BaseException as error:
                    try:
                        with closing(sqlite3.connect(self.path)) as target:
                            before.backup(target)
                            if tuple(target.iterdump()) != expected:
                                raise ValueError("working store was not fully restored")
                    except Exception as recovery:
                        raise DirectUpdateIntegrityError("轮次回传失败且无法确认工作库恢复") from recovery
                    if not isinstance(error, Exception):
                        raise
                    raise DirectUpdateError("轮次回传失败，工作库已回滚，可重试") from error

    def _validate(self, record):
        for name in ("turn_id", "question", "namespace", "data_version"):
            if not isinstance(record.get(name), str) or not record[name].strip():
                raise ValueError(f"invalid {name}")
        if (record["namespace"], record["data_version"]) != (self.scene.namespace, self.scene.data_version):
            raise ValueError("turn belongs to a different memory scope")
        if record.get("status") not in ("completed", "failed", "limit_reached"):
            raise ValueError("invalid turn status")
        for name in ("execution_success", "is_complete"):
            if type(record.get(name)) is not bool:
                raise ValueError(f"invalid {name}")
        columns, rows = record.get("columns"), record.get("rows")
        if (not isinstance(columns, list) or any(not isinstance(c, str) for c in columns)
                or not isinstance(rows, list) or any(not isinstance(r, list) or len(r) != len(columns) for r in rows)):
            raise ValueError("invalid structured result")
        if record.get("sql") is not None and not isinstance(record["sql"], str):
            raise ValueError("invalid SQL")
        if record["execution_success"] and (not record.get("sql") or not columns):
            raise ValueError("successful result requires SQL and columns")
        if not isinstance(record.get("answer", ""), str) or not isinstance(record.get("constraints", ""), str):
            raise ValueError("invalid answer or constraints")
        context = record.get("context", [])
        if not isinstance(context, list) or any(not isinstance(m, dict) or m.get("role") not in
                ("user", "assistant") or not isinstance(m.get("content"), str) for m in context):
            raise ValueError("invalid public context")

    def submit(self, turn):
        record = turn.to_dict() if hasattr(turn, "to_dict") else json.loads(_json(turn))
        self._validate(record)
        payload = _json(record).encode()
        sha = hashlib.sha256(payload).hexdigest()
        name = hashlib.sha256(record["turn_id"].encode()).hexdigest() + ".json"
        source = self.records / name
        with self._transaction():
            if source.exists():
                if source.read_bytes() != payload:
                    raise ValueError("turn_id already binds a different immutable result")
            else:
                # Atomic publication; a failed write never leaves a partial turn.
                import os
                import tempfile
                fd, temporary = tempfile.mkstemp(dir=self.records)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(payload)
                    os.link(temporary, source)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            historical = history_from_turn(record, source)
            with closing(sqlite3.connect(self.path)) as connection:
                existing = connection.execute("SELECT source_ref, source_sha256 FROM sql_learning_turns "
                                              "WHERE turn_id=?", (record["turn_id"],)).fetchone()
            if existing is not None:
                if existing != (str(source), sha):
                    raise ValueError("turn source binding differs")
                return self._result(record["turn_id"])
            report = self.history.import_batch(
                [historical], namespace=self.scene.namespace, data_version=self.scene.data_version,
                source_root=str(self.records), scanned_files=1, errors=[], started_at=perf_counter())
            if report["errors"] or report["conflicts"]:
                raise ValueError("turn history import conflict")
            with closing(sqlite3.connect(self.path)) as connection, connection:
                connection.execute("INSERT OR IGNORE INTO sql_learning_turns VALUES (?, ?, ?)",
                                   (record["turn_id"], str(source), sha))
            self._refresh(record["turn_id"])
        return self._result(record["turn_id"])

    def evaluate(self, turn_id, verdict, *, policy_version="row_multiset_v1"):
        """Trusted evaluator submits PASS/FAIL/UNKNOWN; gold never enters memory."""
        if verdict not in ("PASS", "FAIL", "UNKNOWN"):
            raise ValueError("verdict must be PASS, FAIL or UNKNOWN")
        if not isinstance(policy_version, str) or not policy_version.strip():
            raise ValueError("policy_version is required")
        with self._transaction():
            with closing(sqlite3.connect(self.path)) as connection:
                row = connection.execute("SELECT source_ref, source_sha256 FROM sql_learning_turns "
                                         "WHERE turn_id=?", (turn_id,)).fetchone()
            if row is None:
                raise ValueError("unknown turn_id")
            source = Path(row[0])
            if hashlib.sha256(source.read_bytes()).hexdigest() != row[1]:
                raise ValueError("immutable turn source changed")
            record = json.loads(source.read_bytes())
            self._validate(record)
            if verdict == "PASS" and not (record["execution_success"] and record["is_complete"]
                                           and record["status"] == "completed"):
                raise ValueError("PASS requires a completed turn and successful complete SQL")
            historical = history_from_turn(record, source)
            with closing(sqlite3.connect(self.path)) as connection:
                previous = connection.execute("SELECT verdict, policy_version FROM sql_learning_evaluations "
                                              "WHERE turn_id=?", (turn_id,)).fetchone()
            if previous == (verdict, policy_version):
                return self._result(turn_id)
            self._attach(historical, verdict, policy_version)
            self._refresh(turn_id)
            # Publish the retry shortcut only after every pipeline stage succeeds.
            with closing(sqlite3.connect(self.path)) as connection, connection:
                connection.execute("INSERT INTO sql_learning_evaluations VALUES (?, ?, ?) ON CONFLICT(turn_id) "
                                   "DO UPDATE SET verdict=excluded.verdict, policy_version=excluded.policy_version",
                                   (turn_id, verdict, policy_version))
        return self._result(turn_id)

    def _attach(self, historical, verdict, policy_version):
        task = historical.task
        record = asdict(historical)
        if inspect_source(record, historical.source_sha256, "held", Path)["status"] != "verified":
            raise ValueError("unverified turn source")
        qid = task.task_id + "_result"
        evidence = [{"source_kind": "standard_turn_evaluation", "source_ref": task.source_ref,
                     "source_sha256": historical.source_sha256,
                     "signals": {"reference_result_match": {"PASS": True, "FAIL": False, "UNKNOWN": None}[verdict]},
                     "details": {"comparison_policy": policy_version,
                                 "legacy_b1": {"query_id": qid}}}]
        if verdict == "PASS":
            evidence[0]["signals"]["independent_review"] = "pass"
            evidence[0]["details"]["independent_review"] = {
                "scope": "sql", "query_id": qid, "verdict": "pass",
                "admission_policy": "sql_result_direct_v1", "actor": "standard-turn-evaluator-v2"}
        profile = _profile(task.task_id, record, evidence, self.records)
        payload = _json(profile)
        sha = hashlib.sha256(payload.encode()).hexdigest()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            for statement in SCHEMA:
                connection.execute(statement)
            connection.execute("INSERT OR IGNORE INTO evaluation_versions VALUES (?, ?, ?, datetime('now'))",
                               (task.task_id, sha, payload))
            connection.execute("INSERT INTO evaluation_current VALUES (?, ?) ON CONFLICT(execution_id) "
                               "DO UPDATE SET profile_sha256=excluded.profile_sha256", (task.task_id, sha))

    def _refresh(self, turn_id):
        scope = dict(namespace=self.scene.namespace, data_version=self.scene.data_version)
        self.quality.assess_batch(**scope, execution_ids=(turn_id,))
        maintenance = run_pool_maintenance(quality_store=self.quality, pool_store=self.pool, **scope,
                                           config=self.config, execution_ids=(turn_id,), refresh_quality=False)
        if maintenance["error_count"]:
            raise ValueError("turn pool maintenance failed")
        index = OfflineIndexBuilder(self.retriever.index, self.retriever.prefilter.embeddings).build(**scope)
        if index["error_count"] or not index["index_complete"]:
            raise ValueError("turn index publication failed")

    def _result(self, turn_id):
        active = self.pool.load_active_members(namespace=self.scene.namespace, data_version=self.scene.data_version)
        return {"turn_id": turn_id, "stored": True,
                "admitted": any(m["identity"]["execution_id"] == turn_id for m in active)}
