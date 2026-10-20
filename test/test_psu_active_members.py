"""有效成员读取验证：复用生产接口，不调用模型或业务数据库。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from src.project_paths import PROJECT_ROOT
from src.psu_memory.history.history_adapter import read_history_file
from src.psu_memory.history.history_store import SQLiteHistoryStore
from src.psu_memory.evaluation.quality_models import Feedback, QualityDecision
from src.psu_memory.evaluation.quality_store import SQLiteQualityStore
from src.psu_memory.evaluation.evaluation_evidence import attach_sql_result_pass
from src.psu_memory.pool._common import canonical_json, sha256_text
from src.psu_memory.pool.classification import classify, apply_group_override
from src.psu_memory.pool.pool_models import (
    GROUP_IDS,
    DIFFICULTY_PROMPT_VERSION,
    ContentKey,
    DifficultyAssessment,
    DifficultyResult,
    DifficultyUsage,
    GroupingInput,
    PoolCandidate,
    PoolConfig,
    PoolScope,
)
from src.psu_memory.pool.pool_store import SQLitePoolStore


def database_snapshot(path: Path) -> dict:
    """用于测试及验收命令；同时检查表结构和内容，支持向量 BLOB。"""

    result = {}
    with closing(sqlite3.connect(
        path.as_uri() + "?mode=ro", uri=True
    )) as connection, connection:
        result["user_version"] = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]

        tables = connection.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type='table' ORDER BY name
            """
        ).fetchall()

        for name, definition in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            rows = connection.execute(
                f"SELECT * FROM {quoted}"
            ).fetchall()

            serializable = [
                [
                    {"blob_hex": value.hex()}
                    if isinstance(value, bytes)
                    else value
                    for value in row
                ]
                for row in rows
            ]
            serializable.sort(key=canonical_json)

            result[name] = {
                "definition": definition,
                "row_count": len(rows),
                "content_sha256": sha256_text(
                    canonical_json(serializable)
                ),
            }

    return result


class ActiveMemberTests(unittest.TestCase):
    def setUp(self):
        base = PROJECT_ROOT / "data/derived/psu_memory/validation"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix="active-members-unit-", dir=base
        )
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "history.sqlite3"

        self.pool = SQLitePoolStore(self.db)
        self.quality = SQLiteQualityStore(self.db)
        self.initialized = False
        self.sequence = 0

        self.detail_keys = [
            self._add_sample("列出正常营业的企业名单。"),
            self._add_sample("列出建筑行业的企业名单。"),
            self._add_sample("列出物流运输行业的企业名单。"),
        ]
        self.aggregate_key = self._add_sample(
            "统计正常营业的企业数量。",
            sql="SELECT COUNT(*) AS enterprise_count FROM enterprise_basic;",
        )
        self.pool.sync(namespace="unit", data_version="v1")

    def _add_sample(
        self,
        question,
        *,
        namespace="unit",
        data_version="v1",
        annotate=True,
        sql="SELECT enterprise_id FROM enterprise_basic;",
        context_messages=None,
    ):
        self.sequence += 1
        log = self.root / f"log-{self.sequence}.json"
        raw = {
            "baseline_id": data_version,
            "session_id": f"fixture-{self.sequence}",
            "turn_index": 1,
            "mode": "psu",
            "experiment_id": "active-member-test",
            "question": question,
            "answer": "测试回答",
            "status": "completed",
            "evaluation": {"verdict": "pass"},
            "messages": [
                *(context_messages or []),
                {"role": "user", "content": question},
            ],
            "operations": [{
                "kind": "sql",
                "query_id": f"query-{self.sequence}",
                "sql": sql,
                "success": True,
                "is_complete": True,
                "columns": ["value"],
                "rows": [[1]],
                "scope": {
                    "enterprise": "测试企业",
                    "time_range": "当前",
                },
            }],
            "metrics": {},
        }
        log.write_text(
            json.dumps(raw, ensure_ascii=False),
            encoding="utf-8",
        )
        record = read_history_file(
            log,
            namespace=namespace,
            data_version=data_version,
        )
        SQLiteHistoryStore(self.db).import_batch(
            [record],
            namespace=namespace,
            data_version=data_version,
            source_root=str(self.root),
            scanned_files=1,
            errors=[],
            started_at=perf_counter(),
        )

        attach_sql_result_pass(self.db, record, f"query-{self.sequence}")
        report = self.quality.assess_batch(
            namespace=namespace,
            data_version=data_version,
        )
        decision = next(
            value for value in report["decisions"]
            if value["execution_id"] == record.task.task_id
        )

        if not self.initialized:
            self.pool.initialize()
            self.initialized = True

        accepted = self.quality.submit_feedback(Feedback(
            record.task.task_id,
            decision["content_version"],
            "helpful",
            actor="unit-test",
        ))["decision"]
        self.assertEqual(accepted["admission_state"], "eligible")
        self.assertTrue(accepted["direct_sql_result_verified"])

        if annotate:
            self._annotate(
                record.task.task_id,
                namespace=namespace,
                data_version=data_version,
            )

        return record.task.task_id

    def _annotate(
        self,
        execution_id,
        *,
        namespace="unit",
        data_version="v1",
    ):
        source = next(
            entry
            for entry in self.pool.load_inputs(
                namespace=namespace,
                data_version=data_version,
            )
            if entry["identity"]["execution_id"] == execution_id
        )
        content = source["quality"].content
        grouping = classify(GroupingInput(
            **source["identity"],
            question=content["question"],
            context_text=content.get("context_text", ""),
            sql=content["sql"],
        ))

        # 明确的测试标注，不模拟一次真实模型调用。
        assessment = DifficultyAssessment(
            cache_key=sha256_text(
                execution_id + ":" + source["identity"]["content_version"]
            ),
            prompt_version=DIFFICULTY_PROMPT_VERSION,
            prompt_sha256=sha256_text("active-member-test-fixture"),
            model_signature="no-model-test-fixture",
            status="assessed",
            level="L1",
            reason="仅供有效成员读取测试使用。",
            missing_information=(),
            origin_usage=DifficultyUsage(),
        )
        self.pool.put(assessment)
        difficulty = DifficultyResult(
            **source["identity"],
            assessment=assessment,
            cache_hit=False,
            usage=DifficultyUsage(),
        )
        self.pool.save_annotation(
            source,
            grouping=grouping,
            difficulty=difficulty,
        )

    def _members(self, namespace="unit", data_version="v1"):
        return self.pool.load_active_members(
            namespace=namespace,
            data_version=data_version,
        )

    def _member_row(self, execution_id):
        with self.pool.connection() as connection:
            return connection.execute(
                """
                SELECT * FROM psu_pool_members
                WHERE namespace='unit' AND data_version='v1'
                  AND execution_id=?
                """,
                (execution_id,),
            ).fetchone()

    def _restore_stale_member(self, row):
        # 只在测试库中模拟残留成员，检查读取入口不会直接信任它。
        with self.pool.connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO psu_pool_members
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def test_views_snapshots_and_read_only(self):
        before = database_snapshot(self.db)
        first = self._members()
        second = self._members()

        self.assertEqual(len(first), 4)
        self.assertEqual(
            [item["candidate"].key for item in first],
            [item["candidate"].key for item in second],
        )

        for item in first:
            self.assertIsInstance(item["candidate"], PoolCandidate)
            self.assertIsInstance(item["quality"], QualityDecision)
            self.assertEqual(
                item["candidate"].key,
                ContentKey(
                    item["quality"].execution_id,
                    item["quality"].content_version,
                ),
            )
            self.assertEqual(item["candidate"].admission_state, "eligible")
            self.assertTrue(item["quality"].content["question"])
            self.assertTrue(item["quality"].content["sql"])

        for group in GROUP_IDS:
            expected = tuple(
                item["candidate"].key
                for item in first
                if item["candidate"].group_id == group
            )
            snapshot = self.pool.active_snapshot(
                PoolScope("unit", "v1", group)
            )
            self.assertEqual(snapshot.members, expected)

        self.assertEqual(database_snapshot(self.db), before)

    def test_scope_isolation(self):
        other = self._add_sample(
            "列出企业名单。",
            namespace="other",
        )
        versioned = self._add_sample(
            "列出企业名单。",
            data_version="v2",
        )
        self.pool.sync(namespace="other", data_version="v1")
        self.pool.sync(namespace="unit", data_version="v2")

        self.assertEqual(len(self._members()), 4)
        self.assertEqual(
            [item["candidate"].execution_id for item in self._members("other")],
            [other],
        )
        self.assertEqual(
            [
                item["candidate"].execution_id
                for item in self._members(data_version="v2")
            ],
            [versioned],
        )
        self.assertEqual(self._members("missing"), [])

        with self.assertRaises(ValueError):
            self._members(namespace=" ")

    def test_eligible_without_membership_is_excluded(self):
        extra = self._add_sample(
            "列出其他企业名单。",
            annotate=False,
        )
        source = next(
            item
            for item in self.pool.load_inputs(
                namespace="unit", data_version="v1"
            )
            if item["identity"]["execution_id"] == extra
        )
        self.assertEqual(source["quality"].admission_state, "eligible")
        self.assertNotIn(
            extra,
            {item["candidate"].execution_id for item in self._members()},
        )

    def test_negative_feedback_and_stale_member(self):
        key = self.detail_keys[0]
        stale = self._member_row(key)
        member = next(
            item for item in self._members()
            if item["candidate"].execution_id == key
        )
        self.quality.submit_feedback(Feedback(
            key,
            member["candidate"].content_version,
            "problem",
            error_reason="测试负面反馈",
        ))

        self.assertNotIn(
            key,
            {item["candidate"].execution_id for item in self._members()},
        )

        self._restore_stale_member(stale)
        self.assertNotIn(
            key,
            {item["candidate"].execution_id for item in self._members()},
        )
        self.assertNotIn(
            member["candidate"].key,
            self.pool.active_snapshot(
                PoolScope("unit", "v1", "detail")
            ).members,
        )

    def test_correction_requires_current_annotation(self):
        key = self.detail_keys[0]
        stale = self._member_row(key)
        old = next(
            item for item in self._members()
            if item["candidate"].execution_id == key
        )
        corrected_sql = (
            "SELECT DISTINCT enterprise_id FROM enterprise_basic;"
        )
        corrected = self.quality.submit_feedback(Feedback(
            key,
            old["candidate"].content_version,
            "correction",
            corrected_sql=corrected_sql,
            confirmed=True,
            notes="测试修正版。",
        ))["decision"]

        self.assertEqual(corrected["admission_state"], "eligible")
        self._restore_stale_member(stale)
        self.assertNotIn(
            key,
            {item["candidate"].execution_id for item in self._members()},
        )

        # 人工确认的修正版没有对应的实际 SQL PASS，不能进入当前实验池。
        with self.assertRaises(RuntimeError):
            self._annotate(key)
        self.pool.sync(namespace="unit", data_version="v1")
        self.assertNotIn(key, {item["candidate"].execution_id for item in self._members()})

    def test_stale_annotation_is_excluded(self):
        key = self.detail_keys[0]
        stale = self._member_row(key)
        source = next(
            item
            for item in self.pool.load_inputs(
                namespace="unit", data_version="v1"
            )
            if item["identity"]["execution_id"] == key
        )
        grouping = apply_group_override(
            source["grouping"],
            "detail",
            reason="测试同组标注版本更新。",
            actor="unit-test",
        )
        self.pool.save_annotation(
            source,
            grouping=grouping,
            difficulty=source["difficulty"],
        )

        self._restore_stale_member(stale)
        self.assertNotIn(
            key,
            {item["candidate"].execution_id for item in self._members()},
        )

    def test_capacity_is_applied_on_read(self):
        config = PoolConfig(capacities={
            "detail": 1,
            "aggregate": 0,
            "compare": 8,
            "rank": 8,
        })
        with self.pool.connection(write=True) as connection:
            connection.execute(
                """
                UPDATE psu_pool_scopes SET config_json=?
                WHERE namespace='unit' AND data_version='v1'
                """,
                (canonical_json(config.to_dict()),),
            )

        members = self._members()
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["candidate"].group_id, "detail")
        self.assertEqual(
            self.pool.active_snapshot(
                PoolScope("unit", "v1", "aggregate")
            ).members,
            (),
        )
        self.assertTrue(all(
            item["quality"].admission_state == "eligible"
            for item in self.pool.load_inputs(
                namespace="unit", data_version="v1"
            )
        ))
        with self.pool.connection() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM psu_pool_members"
                ).fetchone()[0],
                4,
            )

    def test_one_transaction_snapshot(self):
        # WAL 允许另一个连接在读取期间提交反馈。
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")

        key = self.detail_keys[0]
        version = next(
            item["candidate"].content_version
            for item in self._members()
            if item["candidate"].execution_id == key
        )
        original_read_config = self.pool._read_config

        def submit_during_read(connection, namespace, data_version):
            config = original_read_config(
                connection, namespace, data_version
            )
            self.quality.submit_feedback(Feedback(
                key, version, "problem",
                error_reason="测试读取期间提交反馈。",
            ))
            return config

        with patch.object(
            self.pool,
            "_read_config",
            side_effect=submit_during_read,
        ):
            snapshot_members = self._members()

        # 本次读取保持同一个事务快照。
        self.assertIn(
            key,
            {item["candidate"].execution_id for item in snapshot_members},
        )
        # 下一次读取立即看到已经提交的停用。
        self.assertNotIn(
            key,
            {item["candidate"].execution_id for item in self._members()},
        )


if __name__ == "__main__":
    unittest.main()
