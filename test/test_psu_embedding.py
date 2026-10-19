"""步骤二回归：模拟编码器 + 实际 SQLite，不加载模型。"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from src.project_paths import PROJECT_ROOT
from src.psu_memory.history.history_store import SCHEMA as HISTORY_SCHEMA
from src.psu_memory.evaluation.quality_store import SCHEMA as QUALITY_SCHEMA
from src.psu_memory.pool._common import sha256_text
from src.psu_memory.pool.pool_store import SQLitePoolStore
from src.psu_memory.retrieval.embedding import (
    EmbeddingUnavailable,
    LocalBgeM3Encoder,
)
from src.psu_memory.retrieval.embedding_service import EmbeddingService
from src.psu_memory.retrieval.retrieval_models import (
    EmbeddingUsage,
    RetrievalConfig,
)
from src.psu_memory.retrieval.text import build_matching_text
from src.psu_memory.retrieval.vector_cache import (
    SQLiteVectorCache,
    vector_cache_key,
)


class FakeEncoder:
    def __init__(self, config):
        self.config = config
        self.calls = 0
        self.fail = False

    def count_tokens(self, text, *, usage):
        # 人工长度规则，只用于确定性验证预算分支。
        return len(text) + 2

    def encode(self, texts, *, usage):
        self.calls += 1
        if self.fail:
            raise EmbeddingUnavailable("simulated_encoding_failure")

        usage.embedding_batch_count += 1
        usage.encoded_text_count += len(texts)
        usage.embedding_input_tokens += sum(
            len(text) + 2 for text in texts
        )

        vectors = []
        for text in texts:
            vector = [0.0] * self.config.embedding_dimension
            index = int(sha256_text(text)[:8], 16) % len(vector)
            vector[index] = 1.0
            vectors.append(tuple(vector))
        return vectors


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        root = PROJECT_ROOT / "data/derived/psu_memory/validation"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(
            prefix="embedding-unit-", dir=root
        )
        self.addCleanup(self.temp.cleanup)
        self.db = root / self.temp.name / "history.sqlite3"

        with sqlite3.connect(self.db) as connection:
            for statement in (*HISTORY_SCHEMA, *QUALITY_SCHEMA):
                connection.execute(statement)
            connection.execute("PRAGMA user_version=1")

        self.pool = SQLitePoolStore(self.db)
        self.pool.initialize()
        self.cache = SQLiteVectorCache(self.pool)
        self.cache.initialize()

        self.config = RetrievalConfig()
        self.encoder = FakeEncoder(self.config)
        self.service = EmbeddingService(
            self.config, self.cache, encoder=self.encoder
        )

    def encode(self, texts, **kwargs):
        return self.service.encode_many(
            texts,
            namespace="test",
            data_version="v1",
            **kwargs,
        )

    def test_cache_and_duplicate_reuse(self):
        text = build_matching_text("统计企业数量")
        first = self.encode([text, text])
        self.assertTrue(all(item.ready for item in first.items))
        self.assertEqual(first.usage.encoded_text_count, 1)
        self.assertEqual(first.usage.deduplicated_count, 1)

        second = self.encode([text, text])
        self.assertEqual(second.usage.cache_hit_count, 2)
        self.assertEqual(second.usage.embedding_batch_count, 0)
        self.assertEqual(second.usage.embedding_input_tokens, 0)
        self.assertEqual(first.items[0].vector, second.items[0].vector)
        self.assertEqual(self.cache.info()["vector_count"], 1)

    def test_new_service_uses_persistent_cache(self):
        text = build_matching_text("查询企业名称")
        first = self.encode([text])

        encoder = FakeEncoder(self.config)
        encoder.fail = True
        service = EmbeddingService(
            self.config,
            SQLiteVectorCache(SQLitePoolStore(self.db)),
            encoder=encoder,
        )
        second = service.encode_many(
            [text], namespace="test", data_version="v1"
        )
        self.assertTrue(second.items[0].cache_hit)
        self.assertEqual(encoder.calls, 0)
        self.assertEqual(first.items[0].vector, second.items[0].vector)

    def test_scope_and_signature_isolation(self):
        text = build_matching_text("统计企业数量")
        self.encode([text])

        scoped = self.service.encode_many(
            [text], namespace="other", data_version="v1"
        )
        versioned = self.service.encode_many(
            [text], namespace="test", data_version="v2"
        )
        self.assertFalse(scoped.items[0].cache_hit)
        self.assertFalse(versioned.items[0].cache_hit)

        config = replace(self.config, embedding_revision="test-revision-2")
        service = EmbeddingService(
            config, self.cache, encoder=FakeEncoder(config)
        )
        changed = service.encode_many(
            [text], namespace="test", data_version="v1"
        )
        self.assertFalse(changed.items[0].cache_hit)
        self.assertEqual(self.cache.info()["vector_count"], 4)

    def test_overflow_falls_back_to_question(self):
        history = json.dumps([
            {"role": "user", "content": "统计企业数量。" * 200}
        ], ensure_ascii=False)
        text = build_matching_text("那北京呢？", history)
        first = self.encode([text])

        item = first.items[0]
        self.assertTrue(item.ready)
        self.assertEqual(item.prepared.mode, "question_only")
        self.assertEqual(
            item.prepared.matching.reason_code,
            "context_overflow_question_only",
        )
        self.assertEqual(item.prepared.matching.text, "那北京呢？")
        self.assertGreater(item.prepared.original_token_count, 1024)

        second = self.encode([text], question_only=True)
        self.assertTrue(second.items[0].cache_hit)
        self.assertEqual(item.cache_key, second.items[0].cache_key)

    def test_token_boundary_and_empty_context(self):
        accepted = self.encode([build_matching_text("企" * 1022)])
        rejected = self.encode([build_matching_text("企" * 1023)])
        missing = self.encode([build_matching_text("那北京呢？")])

        self.assertTrue(accepted.items[0].ready)
        self.assertFalse(rejected.items[0].ready)
        self.assertEqual(
            rejected.items[0].prepared.matching.reason_code,
            "question_too_long",
        )
        self.assertEqual(rejected.usage.embedding_batch_count, 0)
        self.assertTrue(missing.items[0].ready)
        self.assertEqual(
            missing.items[0].prepared.matching.reason_code,
            "question_only",
        )

    def test_corrupted_cache_is_repaired(self):
        text = build_matching_text("统计企业数量")
        first = self.encode([text])

        with self.pool.connection(write=True) as connection:
            connection.execute(
                """
                UPDATE psu_retrieval_vectors SET vector_blob=?
                WHERE cache_key=?
                """,
                (b"x", first.items[0].cache_key),
            )

        second = self.encode([text])
        self.assertTrue(second.items[0].ready)
        self.assertFalse(second.items[0].cache_hit)
        self.assertEqual(second.usage.corrupted_cache_count, 1)
        self.assertEqual(second.usage.repaired_cache_count, 1)
        self.assertEqual(self.cache.info()["vector_count"], 1)

    def test_write_transaction_rolls_back(self):
        texts = [
            build_matching_text("统计企业数量"),
            build_matching_text("查询企业名称"),
        ]
        second_key = vector_cache_key(
            "test", "v1", texts[1].input_sha256,
            self.service.signature_json,
        )

        with self.pool.connection(write=True) as connection:
            connection.execute(f"""
                CREATE TRIGGER reject_second_vector
                BEFORE INSERT ON psu_retrieval_vectors
                WHEN NEW.cache_key='{second_key}'
                BEGIN
                    SELECT RAISE(ABORT, 'validation rollback');
                END
            """)

        with self.assertRaises(sqlite3.IntegrityError):
            self.encode(texts)

        self.assertEqual(self.cache.info()["vector_count"], 0)

    def test_mps_unavailable_has_no_cpu_fallback(self):
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False)
            )
        )
        encoder = LocalBgeM3Encoder(self.config)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(
                EmbeddingUnavailable, "mps_unavailable"
            ):
                encoder.encode(
                    ["统计企业数量"],
                    usage=EmbeddingUsage(embedding_input_tokens=0),
                )

        self.assertEqual(encoder.config.device, "mps")

    def test_schema_and_repeated_initialization(self):
        self.encode([build_matching_text("统计企业数量")])
        result = self.cache.initialize()

        self.assertEqual(result["history_schema_version"], 1)
        self.assertEqual(result["pool_schema_version"], 2)
        self.assertEqual(result["retrieval_schema_version"], 1)
        self.assertEqual(result["vector_count"], 1)


if __name__ == "__main__":
    unittest.main()