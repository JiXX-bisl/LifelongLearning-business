"""由用户独立运行：直接成功经验准入、私有标准隔离及无流水失败回滚。"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.psu_memory.history.history_adapter import history_from_query_result
from src.psu_memory.integration.bootstrap import initialize_empty_store
from src.psu_memory.integration.direct_update import (
    DirectExperienceUpdater, DirectUpdateError, DirectUpdateIntegrityError,
)
from src.psu_memory.pool.pool_models import DifficultyUsage, ModelReply, PoolConfig
from src.psu_memory.pool.pool_store import SQLitePoolStore
from src.psu_memory.retrieval.embedding_service import EmbeddingService
from src.psu_memory.retrieval.index_store import SQLiteRetrievalIndex
from src.psu_memory.retrieval.indexing import OfflineIndexBuilder
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.vector_cache import SQLiteVectorCache


class FakeDifficultyClient:
    signature = "direct-update-unit-client"

    def __init__(self):
        self.calls = 0
        self.closed = False

    def generate(self, **_):
        self.calls += 1
        return ModelReply(
            text=json.dumps({"level": "L2", "reason": "单表查询", "missing_information": []}),
            usage=DifficultyUsage(model_call_count=1, prompt_tokens=10, completion_tokens=8),
        )

    def close(self):
        self.closed = True


class FakeEncoder:
    def __init__(self, config):
        self.config = config

    def count_tokens(self, text, *, usage):
        return len(text)

    def encode(self, texts, *, usage):
        usage.embedding_input_tokens += sum(len(text) for text in texts)
        usage.encoded_text_count += len(texts)
        return [(1.0, 0.0) for _ in texts]


def database_rows(path):
    with closing(sqlite3.connect(path)) as connection, connection:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {table: sorted(connection.execute('SELECT * FROM "' + table + '"').fetchall(), key=repr)
                for table in tables}


class DirectUpdateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "history.sqlite3"
        self.pool_config = PoolConfig(capacities={"detail": 1, "aggregate": 8, "compare": 8, "rank": 8})
        self.retrieval_config = RetrievalConfig(device="cpu", embedding_dimension=2)
        initialize_empty_store(self.database, namespace="unit", data_version="v1",
                               retrieval_config=self.retrieval_config, pool_config=self.pool_config,
                               record_runs=False)
        self.pool = SQLitePoolStore(self.database, record_runs=False)
        cache = SQLiteVectorCache(self.pool)
        self.index = SQLiteRetrievalIndex(cache, self.retrieval_config)
        embeddings = EmbeddingService(self.retrieval_config, cache, encoder=FakeEncoder(self.retrieval_config))
        retriever = SimpleNamespace(index=self.index, prefilter=SimpleNamespace(embeddings=embeddings))
        self.updater = DirectExperienceUpdater(retriever, "unit", "v1", "unused.yaml", self.pool_config)
        self.client = FakeDifficultyClient()
        self.updater._difficulty_client = self.client
        self.addCleanup(self.updater.close)

    def record(self, case_id="q1", question="列出甲公司在2026年1月份的全部收入明细"):
        case = {"id": case_id, "question": question, "history": [],
                "reference_sql": "PRIVATE_REFERENCE_CANARY"}
        operation = {"kind": "sql", "query_id": case_id + "-sql", "sql": "SELECT 1 AS amount",
                     "columns": ["amount"], "rows": [[1]], "row_count": 1,
                     "success": True, "is_complete": True,
                     "scope": {"enterprise": "甲公司", "time_range": "2026-01"},
                     "notes": "PRIVATE_OPERATION_CANARY"}
        record = {"baseline_id": "v1", "session_id": case_id, "turn_index": 1,
                  "question": question, "answer": "实际回答", "status": "completed",
                  "mode": "standard", "experiment_id": "unit-run", "metrics": {},
                  "messages": [{"role": "user", "content": "PRIVATE_MESSAGES_CANARY"}],
                  "operations": [operation], "expected": "PRIVATE_EXPECTED_CANARY"}
        path = self.root / (case_id + ".json")
        path.write_text(json.dumps({"case_id": case_id, "condition": "adaptive", "question": question,
                                    "query_id": operation["query_id"], "sql": operation["sql"],
                                    "actual": {"columns": ["amount"], "rows": [[1]]},
                                    "expected": "PRIVATE_FILE_CANARY", "verdict": "PASS"},
                                   ensure_ascii=False), encoding="utf-8")
        return case, record, path, operation["query_id"]

    def test_memory_adapter_filters_private_fields_and_uses_only_public_history(self):
        case, record, path, query_id = self.record()
        case["history"] = [{"role": "user", "content": "公开前问", "expected": "PRIVATE_HISTORY_CANARY"}]
        record["operations"].insert(0, {"kind": "sql", "query_id": "earlier",
                                           "sql": "PRIVATE_EARLIER_CANARY"})
        history = history_from_query_result(case, record, path, query_id, namespace="unit", data_version="v1")
        exported = json.dumps(asdict(history), ensure_ascii=False)
        self.assertNotIn("PRIVATE_", exported)
        self.assertEqual(json.loads(history.task.context_text), [{"role": "user", "content": "公开前问"}])
        self.assertEqual(history.task.evaluation_verdict, "unknown")
        self.assertEqual(len(json.loads(history.task.evidence_text)), 1)

    def test_duplicate_query_id_and_incomplete_result_cannot_be_admitted(self):
        for invalid in ("duplicate", "incomplete", "private_history"):
            with self.subTest(invalid=invalid):
                case, record, path, query_id = self.record(invalid)
                if invalid == "duplicate":
                    record["operations"].append(deepcopy(record["operations"][0]))
                elif invalid == "incomplete":
                    record["operations"][0]["is_complete"] = False
                else:
                    case["history"] = [{"role": "assistant", "content": "不能导入私有历史"}]
                before = database_rows(self.database)
                with self.assertRaises(DirectUpdateError):
                    self.updater.apply(case, record, path, query_id)
                self.assertEqual(database_rows(self.database), before)
        self.assertEqual(self.client.calls, 0)

    def test_pass_admits_without_internal_evaluation_or_sidecar_reports(self):
        case, record, path, query_id = self.record()
        original = path.read_bytes()
        self.assertEqual(self.updater.apply(case, record, path, query_id), {"admitted": True})
        members = self.pool.load_active_members(namespace="unit", data_version="v1")
        self.assertEqual(len(members), 1)
        self.assertTrue(members[0]["quality"].execution_verified)
        self.assertEqual(members[0]["quality"].admission_state, "eligible")
        self.assertTrue(self.index.inspect(namespace="unit", data_version="v1")["complete"])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["history.sqlite3", "q1.json"])
        rows = database_rows(self.database)
        for table in ("import_runs", "evaluation_runs", "quality_runs", "psu_pool_events"):
            self.assertEqual(rows[table], [])
        for table in ("history_records", "evaluation_versions", "quality_contents", "quality_versions"):
            self.assertNotIn("PRIVATE_", repr(rows[table]))
        self.updater.apply(case, record, path, query_id)
        self.assertEqual(self.client.calls, 1)
        self.assertEqual(len(self.pool.load_active_members(namespace="unit", data_version="v1")), 1)

    def test_index_failure_restores_evicted_member_and_all_functional_state(self):
        self.updater.apply(*self.record())
        before = database_rows(self.database)
        arguments = self.record("q2", "列出乙公司")
        original_build = OfflineIndexBuilder.build

        def publish_then_fail(builder, **scope):
            report = original_build(builder, **scope)
            current = self.pool.load_active_members(namespace="unit", data_version="v1")
            self.assertEqual(len(current), 1)
            self.assertEqual(current[0]["quality"].content["question"], "列出乙公司")
            self.assertTrue(report["index_complete"])
            raise RuntimeError("模拟发布后异常")

        with patch.object(OfflineIndexBuilder, "build", new=publish_then_fail):
            with self.assertRaises(DirectUpdateError):
                self.updater.apply(*arguments)
        self.assertEqual(database_rows(self.database), before)
        self.assertTrue(self.index.inspect(namespace="unit", data_version="v1")["complete"])

    def test_sql_group_is_independent_of_question_keywords(self):
        case, record, path, query_id = self.record(question="看看目前的情况")
        record["operations"][0]["sql"] = "SELECT COUNT(*) AS total FROM enterprise_basic"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["sql"] = record["operations"][0]["sql"]
        path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        self.updater.apply(case, record, path, query_id)
        members = self.pool.load_active_members(namespace="unit", data_version="v1")
        self.assertEqual(members[0]["candidate"].group_id, "aggregate")

    def test_fifo_evicts_oldest_and_replayed_pass_does_not_reenter(self):
        config = PoolConfig(strategy="fifo", fifo_capacity=1)
        self.pool.configure(namespace="unit", data_version="v1", config=config)
        self.updater.config = config
        first = self.record("old", "短问题")
        second = self.record("new", "较长的问题不应因预览长度较长而在FIFO被淘汰")
        self.updater.apply(*first)
        self.updater.apply(*second)
        self.updater.apply(*first)
        members = self.pool.load_active_members(namespace="unit", data_version="v1")
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0]["quality"].content["question"], second[0]["question"])

    def test_fifo_failed_publication_restores_queue_seen_and_all_functional_rows(self):
        config = PoolConfig(strategy="fifo", fifo_capacity=1)
        self.pool.configure(namespace="unit", data_version="v1", config=config)
        self.updater.config = config
        first = self.record("fifo-old", "原有经验")
        second = self.record("fifo-new", "新提交的经验")
        self.updater.apply(*first)
        before = database_rows(self.database)
        self.assertEqual(len(before["psu_pool_fifo_seen"]), 1)
        original_build = OfflineIndexBuilder.build

        def publish_then_fail(builder, **scope):
            report = original_build(builder, **scope)
            members = self.pool.load_active_members(namespace="unit", data_version="v1")
            self.assertEqual([entry["quality"].content["question"] for entry in members],
                             [second[0]["question"]])
            self.assertEqual(len(database_rows(self.database)["psu_pool_fifo_seen"]), 2)
            self.assertTrue(report["index_complete"])
            raise RuntimeError("模拟FIFO淘汰并发布索引后的失败")

        with patch.object(OfflineIndexBuilder, "build", new=publish_then_fail):
            with self.assertRaises(DirectUpdateError):
                self.updater.apply(*second)
        self.assertEqual(database_rows(self.database), before)
        self.assertTrue(self.index.inspect(namespace="unit", data_version="v1")["complete"])

        # 回滚还原seen标记后，同一新经验可以正常重新提交，而非被误当作已消费。
        self.assertEqual(self.updater.apply(*second), {"admitted": True})
        members = self.pool.load_active_members(namespace="unit", data_version="v1")
        self.assertEqual([entry["quality"].content["question"] for entry in members],
                         [second[0]["question"]])
        self.assertEqual(len(database_rows(self.database)["psu_pool_fifo_seen"]), 2)

    def test_failed_recovery_verification_requires_stopping_condition(self):
        with patch.object(OfflineIndexBuilder, "build", side_effect=RuntimeError("模拟更新失败")):
            with patch.object(self.index, "inspect", side_effect=RuntimeError("模拟恢复复核失败")):
                with self.assertRaises(DirectUpdateIntegrityError):
                    self.updater.apply(*self.record())


if __name__ == "__main__":
    unittest.main()
