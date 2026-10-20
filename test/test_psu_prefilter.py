"""语义初筛回归；使用已知向量，不调用真实模型。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from test import test_psu_retrieval_index as index_fixtures
from test.test_psu_embedding import FakeEncoder

from src.psu_memory.models import RetrievalRequest
from src.psu_memory.evaluation.quality_models import Feedback
from src.psu_memory.pool.pool_models import RankValue
from src.psu_memory.retrieval.embedding_service import EmbeddingService
from src.psu_memory.retrieval.indexing import OfflineIndexBuilder
from src.psu_memory.retrieval.prefilter import SemanticPrefilter
from src.psu_memory.retrieval.retrieval_models import validate_dense_vector


BASE = "列出正常营业的企业名单。"
BUILDING = "列出建筑行业的企业名单。"
LOGISTICS = "列出物流运输行业的企业名单。"
NEW_QUERY = "列出目前正常营业的公司名单。"


class ControlledEncoder(FakeEncoder):
    def encode(self, texts, *, usage):
        fallback = super().encode(texts, usage=usage)
        mapping = {
            BASE: (1.0, 0.0, 0.0, 0.0, 0.0),
            BUILDING: (0.8, 0.6, 0.0, 0.0, 0.0),
            LOGISTICS: (0.6, 0.0, 0.8, 0.0, 0.0),
            "统计正常营业的企业数量。": (0.0, 0.0, 0.0, 1.0, 0.0),
            NEW_QUERY: (1.0, 0.0, 0.0, 0.0, 0.0),
            "请帮我处理这件事。": (1.0, 0.0, 0.0, 0.0, 0.0),
            "查询排名前3的企业。": (1.0, 0.0, 0.0, 0.0, 0.0),
            "列出企业名单。": (0.0, 0.0, 0.0, 0.0, 1.0),
            "那北京呢？": (1.0, 0.0, 0.0, 0.0, 0.0),
        }

        vectors = []
        for text, original in zip(texts, fallback):
            values = (
                (0.0, 1.0, 0.0, 0.0, 0.0)
                if text.startswith("历史用户问题")
                else mapping.get(text)
            )
            if values is None:
                vectors.append(original)
                continue

            vector = [
                *values,
                *([0.0] * (self.config.embedding_dimension - len(values))),
            ]
            vectors.append(validate_dense_vector(
                vector, self.config.embedding_dimension
            ))
        return vectors


class FavorPolicy:
    version = "unit-favor-policy"

    def __init__(self, key):
        self.key = key

    def rank(self, candidate):
        return RankValue(
            key=(0 if candidate.execution_id == self.key else 1,),
            reasons=("仅供排序策略接口测试",),
        )


class PrefilterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = index_fixtures.RetrievalIndexTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

        self.members = self.fixture.fixture
        self.pool = self.fixture.pool
        self.index = self.fixture.index
        self.cache = self.fixture.cache
        self.config = self.fixture.config
        self.encoder = ControlledEncoder(self.config)
        self.embeddings = EmbeddingService(
            self.config, self.cache, encoder=self.encoder
        )
        self.builder = OfflineIndexBuilder(
            self.index, self.embeddings
        )
        self.assertEqual(self.builder.build(
            namespace="unit", data_version="v1"
        )["status"], "ok")
        self.service = SemanticPrefilter(
            self.index, self.embeddings
        )

    def request(self, question=NEW_QUERY, **kwargs):
        return RetrievalRequest(
            task_id="prefilter-unit",
            question=question,
            namespace=kwargs.pop("namespace", "unit"),
            data_version=kwargs.pop("data_version", "v1"),
            **kwargs,
        )

    def test_full_pool_top3_and_query_cache(self):
        first = self.service.prefilter(self.request())
        self.assertEqual(first.status, "ok")
        self.assertEqual(first.routing["search_scope"], "all_groups")
        self.assertEqual(len(first.candidates), 3)
        self.assertEqual(first.diagnostics["scored_member_count"], 4)

        scores = [item["similarity"] for item in first.candidates]
        for actual, expected in zip(scores, (1.0, 0.8, 0.6)):
            self.assertAlmostEqual(actual, expected, places=6)

        self.assertEqual(first.usage.encoded_text_count, 1)
        repeat = self.service.prefilter(self.request())
        self.assertEqual(repeat.usage.embedding_batch_count, 0)
        self.assertEqual(repeat.usage.embedding_input_tokens, 0)
        self.assertEqual(repeat.usage.cache_hit_count, 1)
        self.assertEqual(
            [item["binding"]["index_key"] for item in first.candidates],
            [item["binding"]["index_key"] for item in repeat.candidates],
        )
        self.assertTrue(first.to_dict()["candidates"][0]["content"]["sql"])

    def test_question_wording_never_gates_action_groups(self):
        for question in ("请帮我处理这件事。", "查询排名前3的企业。"):
            with self.subTest(question=question):
                result = self.service.prefilter(self.request(question))
                self.assertEqual(result.status, "ok")
                self.assertEqual(result.routing["search_scope"], "all_groups")
                self.assertEqual(result.diagnostics["scored_member_count"], 4)
                self.assertEqual(result.routing["searched_groups"], ["detail", "aggregate"])

    def test_aggregate_candidate_can_be_selected_across_full_pool(self):
        result = self.service.prefilter(self.request("统计正常营业的企业数量。"))
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.diagnostics["scored_member_count"], 4)
        self.assertEqual(result.candidates[0]["member"]["candidate"].execution_id,
                         self.members.aggregate_key)
        self.assertTrue(any(item["member"]["candidate"].group_id == "detail"
                            for item in result.candidates))

    def test_no_members_and_disabled(self):
        before = self.encoder.calls
        for kwargs in (
            {"namespace": "missing"},
            {"data_version": "missing"},
        ):
            result = self.service.prefilter(self.request(**kwargs))
            self.assertEqual(result.status, "no_members")

        for kwargs in ({"limit": 0}, {"max_context_chars": 0}):
            result = self.service.prefilter(self.request(**kwargs))
            self.assertEqual(result.status, "disabled")

        self.assertEqual(self.encoder.calls, before)

    def test_missing_index_does_not_rebuild(self):
        self.members._add_sample("列出新增行业的企业名单。")
        self.pool.sync(namespace="unit", data_version="v1")

        calls = self.encoder.calls
        vectors = self.cache.info()["vector_count"]
        result = self.service.prefilter(self.request())

        self.assertEqual(result.status, "index_not_ready")
        self.assertEqual(result.candidates, [])
        self.assertEqual(self.encoder.calls, calls)
        self.assertEqual(self.cache.info()["vector_count"], vectors)
        self.assertEqual(self.index.info()["stored_binding_count"], 8)
        self.assertEqual(result.routing["search_scope"], "all_groups")

    def test_corrupted_history_does_not_repair_online(self):
        inspection = self.index.inspect(
            namespace="unit", data_version="v1", group_id="detail"
        )
        key = inspection["bindings"][0]["cache_key"]
        with self.pool.connection(write=True) as connection:
            connection.execute(
                "UPDATE psu_retrieval_vectors SET vector_blob=? WHERE cache_key=?",
                (b"x", key),
            )

        calls = self.encoder.calls
        result = self.service.prefilter(self.request(BASE))
        self.assertEqual(result.status, "index_not_ready")
        self.assertEqual(self.encoder.calls, calls)

        with self.pool.connection() as connection:
            self.assertEqual(connection.execute(
                "SELECT vector_blob FROM psu_retrieval_vectors WHERE cache_key=?",
                (key,),
            ).fetchone()[0], b"x")

    def test_context_overflow_uses_question_only_on_both_sides(self):
        key = self.members._add_sample(
            "那北京呢？",
            context_messages=[
                {"role": "user", "content": BUILDING},
            ],
        )
        self.pool.sync(namespace="unit", data_version="v1")
        self.assertEqual(self.builder.build(
            namespace="unit", data_version="v1"
        )["status"], "ok")

        long_context = json.dumps([
            {"role": "user", "content": BUILDING * 200}
        ], ensure_ascii=False)
        result = self.service.prefilter(self.request(
            "那北京呢？", context_text=long_context
        ))

        self.assertEqual(result.status, "ok")
        self.assertEqual(
            result.diagnostics["comparison_representation"], "question_only"
        )
        self.assertTrue(all(
            item["binding"]["representation"] == "question_only"
            for item in result.candidates
        ))
        dependent = next(
            item for item in result.candidates
            if item["member"]["candidate"].execution_id == key
        )
        self.assertAlmostEqual(dependent["similarity"], 1.0, places=6)

    def test_invalid_context_and_overlong_question(self):
        calls = self.encoder.calls
        for context_text, reason in (
            ("not-json", "invalid_context"),
        ):
            result = self.service.prefilter(self.request(
                "那北京呢？", context_text=context_text
            ))
            self.assertEqual(result.status, "query_unavailable")
            self.assertEqual(result.reason, reason)

        result = self.service.prefilter(self.request(
            "列出企业名单。" * 200
        ))
        self.assertEqual(result.status, "query_unavailable")
        self.assertEqual(result.reason, "question_too_long")
        self.assertEqual(self.encoder.calls, calls)

    def test_relevance_first_then_quality_strategy(self):
        favored = self.members.detail_keys[2]
        service = SemanticPrefilter(
            self.index,
            self.embeddings,
            ranking_policy=FavorPolicy(favored),
        )

        relevant = service.prefilter(self.request())
        self.assertEqual(
            relevant.candidates[0]["member"]["candidate"].execution_id,
            self.members.detail_keys[0],
        )
        tied = service.prefilter(self.request("列出企业名单。"))
        self.assertEqual(
            tied.candidates[0]["member"]["candidate"].execution_id,
            favored,
        )
        self.assertTrue(all(
            item["similarity"] == 0 for item in tied.candidates
        ))

    def test_feedback_during_encoding_filters_current_members(self):
        original_encode = self.embeddings.encode_many
        key = self.members.detail_keys[0]
        member = next(
            item for item in self.pool.load_active_members(
                namespace="unit", data_version="v1"
            )
            if item["candidate"].execution_id == key
        )

        def encode_then_feedback(*args, **kwargs):
            batch = original_encode(*args, **kwargs)
            self.members.quality.submit_feedback(Feedback(
                key,
                member["candidate"].content_version,
                "problem",
                comment="初筛期间收到负面反馈",
            ))
            return batch

        with patch.object(
            self.embeddings, "encode_many",
            side_effect=encode_then_feedback,
        ):
            result = self.service.prefilter(self.request())

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.candidates), 3)
        self.assertEqual(result.diagnostics["filtered_member_count"], 1)
        self.assertNotIn(
            key,
            {
                item["member"]["candidate"].execution_id
                for item in result.candidates
            },
        )

    def test_member_added_during_encoding_returns_not_ready(self):
        original_encode = self.embeddings.encode_many

        def encode_then_add(*args, **kwargs):
            batch = original_encode(*args, **kwargs)
            self.members._add_sample("列出新增行业的企业名单。")
            self.pool.sync(namespace="unit", data_version="v1")
            return batch

        with patch.object(
            self.embeddings, "encode_many",
            side_effect=encode_then_add,
        ):
            result = self.service.prefilter(self.request())

        self.assertEqual(result.status, "index_not_ready")
        self.assertEqual(result.candidates, [])

    def test_encoding_failure_has_no_automatic_retry(self):
        self.encoder.fail = True
        before = self.encoder.calls
        result = self.service.prefilter(self.request())

        self.assertEqual(result.status, "query_unavailable")
        self.assertEqual(result.reason, "simulated_encoding_failure")
        self.assertEqual(result.candidates, [])
        self.assertEqual(self.encoder.calls, before + 1)
        self.assertEqual(self.embeddings.config.device, "mps")

    def test_fully_indexed_new_member_returns_stale(self):
        original_encode = self.embeddings.encode_many

        def encode_then_publish_new_member(*args, **kwargs):
            batch = original_encode(*args, **kwargs)
            self.members._add_sample("列出新增行业的企业名单。")
            self.pool.sync(namespace="unit", data_version="v1")

            # 模拟另一个离线任务完成新成员索引。
            # 独立服务避免再次进入当前测试的补丁。
            other_embeddings = EmbeddingService(
                self.config,
                self.cache,
                encoder=ControlledEncoder(self.config),
            )
            report = OfflineIndexBuilder(
                self.index, other_embeddings
            ).build(namespace="unit", data_version="v1")
            self.assertEqual(report["status"], "ok")
            return batch

        with patch.object(
            self.embeddings,
            "encode_many",
            side_effect=encode_then_publish_new_member,
        ):
            result = self.service.prefilter(self.request())

        self.assertEqual(result.status, "stale")
        self.assertEqual(result.reason, "comparison_scope_changed")
        self.assertEqual(result.candidates, [])
        self.assertEqual(
            result.diagnostics["uncompared_member_count"], 1
        )

    def test_corrected_sql_is_retrieved_only_after_new_pass(self):
        key = self.members.detail_keys[0]
        before = self.service.prefilter(self.request(BASE))
        old = next(
            item for item in before.candidates
            if item["member"]["candidate"].execution_id == key
        )
        corrected_sql = (
            "SELECT DISTINCT enterprise_id FROM enterprise_basic;"
        )

        corrected = self.members.quality.submit_feedback(Feedback(
            key,
            old["member"]["candidate"].content_version,
            "correction",
            corrected_sql=corrected_sql,
            confirmed=True,
            notes="此类企业名单应注意去重。",
        ))["decision"]

        self.assertFalse(corrected["execution_verified"])
        self.assertFalse(corrected["direct_sql_result_verified"])
        with self.assertRaisesRegex(RuntimeError, "准入状态已变化"):
            self.members._annotate(key)
        self.pool.sync(namespace="unit", data_version="v1")

        remaining = self.service.prefilter(self.request(BASE))
        self.assertEqual(remaining.status, "ok")
        self.assertNotIn(key, {
            item["member"]["candidate"].execution_id for item in remaining.candidates
        })

        # 模拟重新执行修正SQL并直接核验PASS，作为独立的新执行记录入池。
        new_key = self.members._add_sample(BASE, sql=corrected_sql)
        self.assertNotEqual(new_key, key)
        self.pool.sync(namespace="unit", data_version="v1")

        calls = self.encoder.calls
        not_ready = self.service.prefilter(self.request(BASE))
        self.assertEqual(not_ready.status, "index_not_ready")
        self.assertEqual(not_ready.candidates, [])
        self.assertEqual(self.encoder.calls, calls)

        rebuilt = self.builder.build(
            namespace="unit", data_version="v1"
        )
        self.assertEqual(rebuilt["status"], "ok")
        self.assertEqual(rebuilt["usage"]["encoded_text_count"], 0)

        result = self.service.prefilter(self.request(BASE))
        self.assertEqual(result.status, "ok")
        current = next(
            item for item in result.candidates
            if item["member"]["candidate"].execution_id == new_key
        )
        quality = current["member"]["quality"]

        self.assertNotEqual(
            quality.content_version,
            corrected["content_version"],
        )
        self.assertEqual(quality.content["sql"], corrected_sql)
        self.assertEqual(quality.content["operation"]["sql"], corrected_sql)
        self.assertNotIn("user_confirmed_correction", quality.basis)
        self.assertIn("reference_result_match", quality.basis)
        self.assertTrue(quality.execution_verified)
        self.assertTrue(quality.direct_sql_result_verified)
        self.assertEqual(current["binding"]["cache_key"], old["binding"]["cache_key"])
        self.assertNotEqual(current["binding"]["index_key"], old["binding"]["index_key"])
        self.assertNotIn(key, {
            item["member"]["candidate"].execution_id for item in result.candidates
        })

    def test_zero_capacity_stops_retrieval_without_suspending(self):
        from src.psu_memory.pool.pool_models import GROUP_IDS, PoolConfig

        before = self.encoder.calls
        self.pool.sync(
            namespace="unit",
            data_version="v1",
            config=PoolConfig(capacities={
                group: 0 for group in GROUP_IDS
            }),
        )

        result = self.service.prefilter(self.request())
        self.assertEqual(result.status, "no_members")
        self.assertEqual(result.candidates, [])
        self.assertEqual(self.encoder.calls, before)
        self.assertEqual(
            self.index.info()["stored_binding_count"], 8
        )
        self.assertTrue(all(
            entry["quality"].admission_state == "eligible"
            for entry in self.pool.load_inputs(
                namespace="unit", data_version="v1"
            )
        ))

    def test_data_advice_does_not_restrict_full_pool(self):
        ordinary = self.service.prefilter(self.request())
        advised = self.service.prefilter(self.request(
            data_advice="统计符合条件的企业数量。"
        ))

        self.assertEqual(ordinary.status, "ok")
        self.assertEqual(advised.status, "ok")
        self.assertEqual(ordinary.routing, advised.routing)
        self.assertEqual(
            [item["binding"]["index_key"] for item in ordinary.candidates],
            [item["binding"]["index_key"] for item in advised.candidates],
        )
        self.assertEqual(
            ordinary.query["cache_key"],
            advised.query["cache_key"],
        )
        self.assertEqual(
            ordinary.query["prepared"] if "prepared" in ordinary.query
            else ordinary.query["matching"],
            advised.query["prepared"] if "prepared" in advised.query
            else advised.query["matching"],
        )
        self.assertEqual(advised.usage.encoded_text_count, 0)

if __name__ == "__main__":
    unittest.main()
