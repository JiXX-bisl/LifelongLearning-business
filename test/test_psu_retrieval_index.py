"""离线索引测试：复用有效成员夹具和模拟编码器。"""

from __future__ import annotations

import sqlite3
import unittest
from dataclasses import replace
from unittest.mock import patch

from test import test_psu_active_members as member_fixtures
from test.test_psu_embedding import FakeEncoder

from src.psu_memory.evaluation.quality_models import Feedback
from src.psu_memory.retrieval.embedding_service import EmbeddingService
from src.psu_memory.retrieval.index_store import SQLiteRetrievalIndex
from src.psu_memory.retrieval.indexing import OfflineIndexBuilder
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.vector_cache import SQLiteVectorCache


class RetrievalIndexTests(unittest.TestCase):
    def setUp(self):
        self.fixture = member_fixtures.ActiveMemberTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

        self.pool = self.fixture.pool
        self.config = RetrievalConfig()
        self.cache = SQLiteVectorCache(self.pool)
        self.index = SQLiteRetrievalIndex(self.cache, self.config)
        self.index.initialize()

        self.encoder = FakeEncoder(self.config)
        self.embeddings = EmbeddingService(
            self.config, self.cache, encoder=self.encoder
        )
        self.builder = OfflineIndexBuilder(
            self.index, self.embeddings
        )

    def build(self):
        return self.builder.build(
            namespace="unit", data_version="v1"
        )

    def inspect(self):
        return self.index.inspect(
            namespace="unit", data_version="v1"
        )

    def test_first_build_and_repeat(self):
        first = self.build()
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["member_count"], 4)
        self.assertEqual(first["ready_binding_count"], 8)
        self.assertEqual(first["unique_cache_key_count"], 4)
        self.assertEqual(first["usage"]["encoded_text_count"], 4)

        before = member_fixtures.database_snapshot(
            self.fixture.db
        )["psu_retrieval_index_members"]
        with (
            patch.object(self.embeddings, "encode_many", side_effect=AssertionError(
                "unchanged index must skip encoding"
            )),
            patch.object(self.index, "publish", side_effect=AssertionError(
                "unchanged index must skip publication"
            )),
        ):
            repeat = self.build()

        self.assertEqual(repeat["status"], "ok")
        self.assertEqual(repeat["publication"]["reason"], "index_unchanged")
        self.assertFalse(repeat["publication"]["published"])
        self.assertEqual(repeat["usage"]["encoded_text_count"], 0)
        self.assertEqual(repeat["usage"]["embedding_input_tokens"], 0)
        self.assertEqual(repeat["publication"]["inserted_count"], 0)
        self.assertEqual(repeat["publication"]["updated_count"], 0)
        self.assertEqual(repeat["publication"]["unchanged_count"], 8)
        self.assertEqual(
            member_fixtures.database_snapshot(
                self.fixture.db
            )["psu_retrieval_index_members"],
            before,
        )
        self.assertTrue(self.inspect()["complete"])

    def test_incremental_and_duplicate_question(self):
        self.build()

        self.fixture._add_sample("列出新增行业的企业名单。")
        self.pool.sync(namespace="unit", data_version="v1")
        self.assertFalse(self.inspect()["complete"])

        added = self.build()
        self.assertEqual(added["member_count"], 5)
        self.assertEqual(added["publication"]["inserted_count"], 2)
        self.assertEqual(added["usage"]["encoded_text_count"], 1)

        self.fixture._add_sample("列出正常营业的企业名单。")
        self.pool.sync(namespace="unit", data_version="v1")
        duplicate = self.build()

        self.assertEqual(duplicate["member_count"], 6)
        self.assertEqual(duplicate["publication"]["inserted_count"], 2)
        self.assertEqual(duplicate["usage"]["encoded_text_count"], 0)
        self.assertEqual(duplicate["unique_cache_key_count"], 5)

    def test_corrected_sql_requires_new_pass_before_reusing_vector(self):
        self.build()
        old = next(
            member for member in self.pool.load_active_members(namespace="unit", data_version="v1")
            if member["candidate"].execution_id == self.fixture.detail_keys[0]
        )
        key = old["candidate"].execution_id
        old_bindings = {
            item["representation"]: item
            for item in self.inspect()["bindings"]
            if item["identity"]["execution_id"] == key
        }

        corrected = self.fixture.quality.submit_feedback(Feedback(
            key,
            old["candidate"].content_version,
            "correction",
            corrected_sql=(
                "SELECT DISTINCT enterprise_id FROM enterprise_basic;"
            ),
            confirmed=True,
            notes="测试去重修正",
        ))["decision"]

        self.assertFalse(corrected["execution_verified"])
        self.assertFalse(corrected["direct_sql_result_verified"])
        with self.assertRaisesRegex(RuntimeError, "准入状态已变化"):
            self.fixture._annotate(key)
        self.pool.sync(namespace="unit", data_version="v1")

        remaining = self.inspect()
        self.assertTrue(remaining["complete"])
        self.assertEqual(remaining["member_count"], 3)
        self.assertEqual(remaining["ready_binding_count"], 6)
        self.assertNotIn(key, {item["identity"]["execution_id"] for item in remaining["bindings"]})

        # 测试夹具另建一次完整执行及直接PASS记录，不能把人工修正当作执行证据。
        new_key = self.fixture._add_sample(
            old["quality"].content["question"], sql=corrected["content"]["sql"],
        )
        self.assertNotEqual(new_key, key)
        self.pool.sync(namespace="unit", data_version="v1")
        self.assertFalse(self.inspect()["complete"])

        report = self.build()
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["usage"]["encoded_text_count"], 0)
        self.assertEqual(report["publication"]["inserted_count"], 2)
        self.assertEqual(report["member_count"], 4)
        self.assertEqual(report["ready_binding_count"], 8)

        selected = [
            item for item in report["bindings"]
            if item["identity"]["execution_id"] == new_key
        ]
        self.assertEqual(len(selected), 2)
        for item in selected:
            previous = old_bindings[item["representation"]]
            self.assertEqual(item["cache_key"], previous["cache_key"])
            self.assertNotEqual(item["index_key"], previous["index_key"])
        member = next(
            item for item in self.pool.load_active_members(namespace="unit", data_version="v1")
            if item["candidate"].execution_id == new_key
        )
        self.assertTrue(member["quality"].execution_verified)
        self.assertTrue(member["quality"].direct_sql_result_verified)
        self.assertIn("reference_result_match", member["quality"].basis)
        self.assertNotIn(key, {item["identity"]["execution_id"] for item in report["bindings"]})
        self.assertEqual(self.index.info()["stored_binding_count"], 10)

    def test_negative_feedback_excludes_old_index(self):
        self.build()
        member = self.pool.load_active_members(
            namespace="unit", data_version="v1"
        )[0]

        self.fixture.quality.submit_feedback(Feedback(
            member["candidate"].execution_id,
            member["candidate"].content_version,
            "problem",
            comment="测试立即停用",
        ))
        current = self.inspect()

        self.assertTrue(current["complete"])
        self.assertEqual(current["member_count"], 3)
        self.assertEqual(current["ready_binding_count"], 6)
        self.assertNotIn(
            member["candidate"].execution_id,
            {
                value["identity"]["execution_id"]
                for value in current["bindings"]
            },
        )
        self.assertEqual(self.index.info()["stored_binding_count"], 8)

    def test_deterministic_exclusions_are_processed(self):
        self.fixture._add_sample("查看这些企业的情况。")
        self.fixture._add_sample("列出企业名单。" * 200)
        self.pool.sync(namespace="unit", data_version="v1")

        report = self.build()
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["index_complete"])
        self.assertEqual(report["member_count"], 6)
        self.assertEqual(report["ready_binding_count"], 10)
        self.assertEqual(report["excluded_binding_count"], 2)
        self.assertEqual(report["error_count"], 0)

        reasons = {
            item["reason"] for item in report["bindings"]
            if item["status"] == "excluded"
        }
        self.assertEqual(reasons, {"question_too_long"})

    def test_member_change_blocks_stale_publication(self):
        member = self.pool.load_active_members(
            namespace="unit", data_version="v1"
        )[0]
        original_publish = self.index.publish

        def change_then_publish(**kwargs):
            self.fixture.quality.submit_feedback(Feedback(
                member["candidate"].execution_id,
                member["candidate"].content_version,
                "problem",
                comment="编码完成后提交反馈",
            ))
            return original_publish(**kwargs)

        with patch.object(
            self.index, "publish", side_effect=change_then_publish
        ):
            report = self.build()

        self.assertEqual(report["status"], "stale")
        self.assertFalse(report["publication"]["published"])
        self.assertEqual(self.index.info()["stored_binding_count"], 0)

        retry = self.build()
        self.assertEqual(retry["status"], "ok")
        self.assertEqual(retry["member_count"], 3)
        self.assertEqual(retry["usage"]["encoded_text_count"], 0)

    def test_corrupted_vector_is_not_ready_then_repaired(self):
        self.build()
        key = next(iter(self.inspect()["vectors"]))

        with self.pool.connection(write=True) as connection:
            connection.execute(
                """
                UPDATE psu_retrieval_vectors SET vector_blob=?
                WHERE cache_key=?
                """,
                (b"x", key),
            )

        current = self.inspect()
        self.assertFalse(current["complete"])
        self.assertTrue(any(
            item["reason"] == "cache_corrupted"
            for item in current["pending"]
        ))

        repaired = self.build()
        self.assertEqual(repaired["status"], "ok")
        self.assertEqual(repaired["usage"]["repaired_cache_count"], 1)
        self.assertEqual(repaired["usage"]["encoded_text_count"], 1)
        self.assertEqual(self.cache.info()["vector_count"], 4)

    def test_publication_transaction_rolls_back(self):
        members = self.pool.load_active_members(
            namespace="unit", data_version="v1"
        )
        rejected = self.index.descriptor(
            members[1], "default"
        )["index_key"]

        with self.pool.connection(write=True) as connection:
            connection.execute(f"""
                CREATE TRIGGER reject_index_binding
                BEFORE INSERT ON psu_retrieval_index_members
                WHEN NEW.index_key='{rejected}'
                BEGIN
                    SELECT RAISE(ABORT, 'index rollback test');
                END
            """)

        report = self.build()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["publication"]["published"])
        self.assertEqual(self.index.info()["stored_binding_count"], 0)
        self.assertEqual(self.cache.info()["vector_count"], 4)

    def test_profile_and_scope_isolation(self):
        self.build()

        changed = replace(self.config, max_context_questions=1)
        changed_index = SQLiteRetrievalIndex(self.cache, changed)
        changed_service = EmbeddingService(
            changed, self.cache, encoder=FakeEncoder(changed)
        )
        self.assertFalse(changed_index.inspect(
            namespace="unit", data_version="v1"
        )["complete"])

        rebuilt = OfflineIndexBuilder(
            changed_index, changed_service
        ).build(namespace="unit", data_version="v1")
        self.assertEqual(rebuilt["status"], "ok")
        self.assertEqual(rebuilt["usage"]["encoded_text_count"], 0)

        other = self.fixture._add_sample(
            "列出正常营业的企业名单。",
            namespace="other",
        )
        self.pool.sync(namespace="other", data_version="v1")
        self.assertFalse(self.index.inspect(
            namespace="other", data_version="v1"
        )["complete"])

        report = self.builder.build(
            namespace="other", data_version="v1"
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["member_count"], 1)
        self.assertTrue(all(
            item["identity"]["execution_id"] == other
            for item in report["bindings"]
        ))


if __name__ == "__main__":
    unittest.main()
