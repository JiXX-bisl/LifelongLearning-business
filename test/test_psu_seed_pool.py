"""由用户独立运行：共同种子初始化的集合、顺序、隔离和失败发布边界。"""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.psu_memory.pool.pool_models import (
    DIFFICULTY_PROMPT_VERSION, DifficultyUsage, ModelReply, PoolConfig,
)
from src.psu_memory.pool.pool_store import SQLitePoolStore
from src.psu_memory.retrieval.embedding_service import EmbeddingService
from src.psu_memory.retrieval.index_store import SQLiteRetrievalIndex
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.vector_cache import SQLiteVectorCache
from src.psu_memory.validation import seed_pool
from src.psu_memory.validation.authored_formation import DATA_VERSION, NAMESPACE


class FakeDifficultyClient:
    signature = "seed-unit-difficulty-v1"

    def __init__(self, fail=False, level="L2"):
        self.fail = fail
        self.level = level
        self.calls = 0
        self.closed = False
        self.payloads = []

    def generate(self, *, system_prompt, user_text):
        self.calls += 1
        self.payloads.append(json.loads(user_text))
        return ModelReply(
            json.dumps({"level": "undetermined" if self.fail else self.level, "reason": "独立测试返回",
                        "missing_information": ["测试失败路径"] if self.fail else []}),
            DifficultyUsage(model_call_count=1, prompt_tokens=None if self.fail else 7,
                            completion_tokens=None if self.fail else 4,
                            token_usage_complete=not self.fail),
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


class SeedPoolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.business = self.root / "business.yaml"
        self.business.write_text("model: {}\n", encoding="utf-8")
        self.retrieval = RetrievalConfig(device="cpu", embedding_dimension=2)
        self.retrieval_file = self.root / "retrieval.json"
        self.retrieval_file.write_text(json.dumps(self.retrieval.to_dict()))
        self.pool_file = self.root / "pool.json"
        self.pool_file.write_text(json.dumps(PoolConfig().to_dict()))
        entries = []
        for i in (3, 1, 2, 6, 4, 5, 9, 7, 8, 12, 10, 11, 15, 13, 14, 18, 16, 17):
            if i <= 5:
                group = "detail"
                sql = ("SELECT registered_capital AS n FROM enterprise_basic "
                       f"WHERE enterprise_name='企业{i}'")
            elif i <= 13:
                group = "aggregate"
                sql = ("SELECT COUNT(*) AS n FROM enterprise_basic "
                       f"WHERE enterprise_name='企业{i}'")
            else:
                group = "compare"
                sql = ("SELECT a.operating_revenue - b.operating_revenue AS n "
                       "FROM enterprise_monthly_operation a JOIN enterprise_monthly_operation b "
                       "ON a.enterprise_id=b.enterprise_id "
                       f"WHERE a.enterprise_id={i} AND a.stat_month='2026-06-01' "
                       "AND b.stat_month='2026-07-01'")
            row = {
                "case_id": f"q{i}", "question": f"查询第{i}家企业的{group}数据", "context": [],
                "sql": sql,
                "actual": {"columns": ["n"], "rows": [[i]], "row_count": 1, "is_complete": True},
                "session_id": f"single-q{i}", "turn": 1, "query_id": f"q{i}-authored",
                "verdict": "pass", "executed": True, "data_version": DATA_VERSION,
                "cost": {"database_seconds": 0.1, "comparison_seconds": 0.01},
            }
            source = self.root / f"q{i}.json"
            source.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            entries.append({"case_id": row["case_id"], "group": group, "record": row,
                            "source": {"path": str(source),
                                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}})
        self.bundle = {"version": "psu_initial_seed_v1", "data_version": DATA_VERSION,
                       "namespace": NAMESPACE, "entries": entries}
        self.seeds = self.root / "seeds.json"
        self.seeds.write_text(json.dumps(self.bundle, ensure_ascii=False), encoding="utf-8")

    def execute(self, client=None, embedding_side_effect=None, *, run=None):
        client = client or FakeDifficultyClient()

        def embeddings(config, cache):
            if embedding_side_effect is not None:
                raise embedding_side_effect
            return EmbeddingService(config, cache, encoder=FakeEncoder(config))

        with patch("src.psu_memory.validation.initial_seeds.load_seed_bundle", return_value=deepcopy(self.bundle)), \
                patch.object(seed_pool, "_schema_context", return_value=("共享公开Schema", {})), \
                patch.object(seed_pool, "_make_difficulty_client", return_value=client), \
                patch.object(seed_pool, "_make_embeddings", side_effect=embeddings):
            result = seed_pool.initialize_seed_pool(
                run_dir=run or self.run, seeds_path=self.seeds, business_config=self.business,
                retrieval_config=self.retrieval_file, pool_config=self.pool_file,
            )
        self.assertTrue(client.closed)
        return result, client

    def test_selected_order_and_both_target_strategies_keep_identical_members(self):
        summary, client = self.execute()
        self.assertTrue(summary["ready"])
        case_ids = [entry["case_id"] for entry in self.bundle["entries"]]
        self.assertEqual(summary["case_ids"], case_ids)
        self.assertEqual(summary["selected_count"], 18)
        self.assertEqual(summary["member_count"], 18)
        self.assertEqual(summary["group_counts"], {"detail": 5, "aggregate": 8, "compare": 5, "rank": 0})
        self.assertEqual(summary["model_difficulty_counts"], {"L1": 0, "L2": 18, "L3": 0, "L4": 0})
        self.assertEqual(client.calls, 18)
        self.assertEqual(summary["cost"]["difficulty_and_pool"]["model_call_count"], 18)
        self.assertEqual(summary["difficulty_prompt_version"], DIFFICULTY_PROMPT_VERSION)
        self.assertEqual([row["case_id"] for row in summary["difficulty_results"]], case_ids)
        self.assertTrue(all(row["status"] == "assessed" and not row["cache_hit"]
                            for row in summary["difficulty_results"]))
        for entry, payload in zip(self.bundle["entries"], client.payloads):
            self.assertEqual(payload["question"], entry["record"]["question"])
            self.assertIn(entry["record"]["sql"], payload["data_advice"])
            self.assertEqual(payload["schema_context"], "共享公开Schema")
            self.assertNotIn("actual", payload)
            self.assertNotIn("expected", payload)
        for name in ("pool_models.py", "difficulty.py", f"{DIFFICULTY_PROMPT_VERSION}.txt"):
            paths = [Path(path) for path in summary["source_sha256"] if Path(path).name == name]
            self.assertEqual(len(paths), 1)
            self.assertEqual(summary["source_sha256"][str(paths[0])],
                             hashlib.sha256(paths[0].read_bytes()).hexdigest())
        self.assertGreater(summary["cost"]["embedding_index"]["embedding_input_tokens"], 0)
        self.assertEqual(sorted(path.name for path in self.run.iterdir()), ["seed.sqlite3", "summary.json"])
        store = self.run / "seed.sqlite3"
        pool = SQLitePoolStore(store, record_runs=False)
        scope = {"namespace": NAMESPACE, "data_version": DATA_VERSION}
        members = pool.load_active_members(**scope)
        self.assertEqual([item["position"] for item in members], list(range(18)))
        self.assertEqual([item["quality"].content["question"] for item in members],
                         [entry["record"]["question"] for entry in self.bundle["entries"]])
        keys = {item["candidate"].key for item in members}
        index = SQLiteRetrievalIndex(SQLiteVectorCache(pool), self.retrieval)
        self.assertTrue(index.inspect(**scope)["complete"])
        pool.configure(**scope, config=PoolConfig(strategy="adaptive"))
        self.assertEqual({item["candidate"].key for item in pool.load_active_members(**scope)}, keys)
        self.assertTrue(index.inspect(**scope)["complete"])
        with closing(sqlite3.connect(store)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM history_records").fetchone()[0], 18)
            for record_json, source_sha in connection.execute("SELECT record_json, source_sha256 FROM history_records"):
                task = json.loads(record_json)["task"]
                self.assertEqual(source_sha, hashlib.sha256(Path(task["source_ref"]).read_bytes()).hexdigest())
                self.assertNotIn("expected", task["evidence_text"])

    def test_failed_difficulty_keeps_partial_and_unknown_tokens(self):
        summary, _ = self.execute(FakeDifficultyClient(fail=True))
        self.assertFalse(summary["ready"])
        self.assertIsNone(summary["seed_store"])
        self.assertFalse((self.run / "seed.sqlite3").exists())
        self.assertTrue((self.run / "seed.partial.sqlite3").is_file())
        self.assertIsNone(summary["cost"]["difficulty_and_pool"]["prompt_tokens"])
        self.assertEqual(summary["error"]["stage"], "difficulty_and_pool")
        self.assertEqual(summary["preparation_error_count"], 18)
        self.assertEqual(len(summary["difficulty_results"]), 18)
        self.assertTrue(all(row["status"] == "undetermined" and row["level"] is None
                            for row in summary["difficulty_results"]))

    def test_embedding_failure_does_not_publish_seed(self):
        summary, _ = self.execute(embedding_side_effect=RuntimeError("encoder failed"))
        self.assertFalse(summary["ready"])
        self.assertFalse((self.run / "seed.sqlite3").exists())
        self.assertTrue((self.run / "summary.json").is_file())
        self.assertEqual(summary["error"]["stage"], "embedding_index")
        self.assertIsNone(summary["cost"]["embedding_index"])

    def test_existing_run_is_not_overwritten(self):
        self.run.mkdir()
        with self.assertRaises(FileExistsError):
            self.execute()
        self.assertEqual(list(self.run.iterdir()), [])

    def test_input_change_during_initialization_prevents_publication(self):
        client = FakeDifficultyClient()
        generate = client.generate

        def change_source(**kwargs):
            result = generate(**kwargs)
            if client.calls == 1:
                source = Path(self.bundle["entries"][0]["source"]["path"])
                source.write_bytes(source.read_bytes() + b"\n")
            return result

        client.generate = change_source
        summary, _ = self.execute(client)
        self.assertFalse(summary["ready"])
        self.assertFalse((self.run / "seed.sqlite3").exists())
        self.assertEqual(summary["error"]["stage"], "publication")
        self.assertIn("来源发生变化", summary["error"]["message"])

    def prepare_manual_review(self):
        original, _ = self.execute(FakeDifficultyClient(level="L3"))
        self.assertTrue(original["ready"])
        review = {
            "version": "psu_seed_difficulty_review_v1", "actor": "assistant",
            "namespace": NAMESPACE, "data_version": DATA_VERSION,
            "source_run": {"path": str(self.run), "summary_sha256": seed_pool._sha(self.run / "summary.json"),
                           "seed_sha256": seed_pool._sha(self.run / "seed.sqlite3")},
            "changes": [{"case_id": case_id, "from_level": "L3", "to_level": "L2",
                         "reason": "人工复核：必要任务为简单派生或筛选，不计实现写法。"}
                        for case_id in ("q3", "q6", "q15")],
        }
        path = self.root / "review.json"
        path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
        return original, review, path

    def apply_manual_review(self, path):
        with patch("src.psu_memory.validation.initial_seeds.load_seed_bundle", return_value=deepcopy(self.bundle)), \
                patch.object(seed_pool, "_make_difficulty_client", side_effect=AssertionError("不应调用模型")), \
                patch.object(seed_pool, "_make_embeddings", side_effect=AssertionError("不应调用嵌入")):
            return seed_pool.apply_difficulty_review(
                run_dir=self.root / "reviewed", review_path=path,
                seeds_path=self.seeds, retrieval_config=self.retrieval_file,
            )

    def test_manual_review_preserves_model_evidence_fifo_and_vectors(self):
        original, review, path = self.prepare_manual_review()
        protected = {item: seed_pool._sha(item) for item in self.run.iterdir()}
        old_pool = SQLitePoolStore(self.run / "seed.sqlite3", record_runs=False)
        scope = {"namespace": NAMESPACE, "data_version": DATA_VERSION}
        previous = old_pool.load_active_members(**scope)
        old_index = SQLiteRetrievalIndex(SQLiteVectorCache(old_pool), self.retrieval).inspect(**scope)
        result = self.apply_manual_review(path)
        self.assertTrue(result["ready"])
        self.assertTrue(result["manual_review"]["applied"])
        self.assertEqual(result["effective_difficulty_counts"], {"L1": 0, "L2": 3, "L3": 15, "L4": 0})
        self.assertEqual(result["model_difficulty_counts"], original["model_difficulty_counts"])
        self.assertEqual(result["cost"], original["cost"])
        self.assertEqual(result["manual_review"]["cost"]["model_call_count"], 0)
        self.assertEqual(result["manual_review"]["cost"]["embedding_input_tokens"], 0)
        self.assertEqual([{key: value for key, value in row.items() if key != "effective_level"}
                          for row in result["difficulty_results"]], original["difficulty_results"])
        pool = SQLitePoolStore(Path(result["seed_store"]), record_runs=False)
        current = pool.load_active_members(**scope)
        self.assertEqual([member["position"] for member in current], list(range(18)))
        changed = {item["case_id"] for item in review["changes"]}
        for seed, before, after in zip(self.bundle["entries"], previous, current):
            prior, assessment = before["difficulty"].assessment, after["difficulty"].assessment
            self.assertEqual(after["identity"], before["identity"])
            self.assertEqual(after["quality"], before["quality"])
            self.assertEqual(after["grouping"], before["grouping"])
            self.assertEqual(pool.get(prior.cache_key), prior)
            if seed["case_id"] in changed:
                self.assertTrue(assessment.cache_key.startswith("manual:"))
                self.assertEqual(assessment.model_signature, "manual:assistant_static_review_v1")
                self.assertEqual(assessment.level, "L2")
                self.assertEqual(assessment.origin_usage, DifficultyUsage())
            else:
                self.assertEqual(assessment, prior)
        index = SQLiteRetrievalIndex(SQLiteVectorCache(pool), self.retrieval).inspect(**scope)
        self.assertTrue(index["complete"])
        self.assertEqual(index["vectors"], old_index["vectors"])
        self.assertEqual(index["ready_binding_count"], old_index["ready_binding_count"])
        for item, value in protected.items():
            self.assertEqual(seed_pool._sha(item), value)

    def test_crlf_prompt_uses_text_digest_and_keeps_exact_source_binding(self):
        from src.psu_memory.pool import difficulty, pool_models
        from src.psu_memory.pool._common import sha256_text

        sources = self.root / "pool_sources"
        (sources / "prompts").mkdir(parents=True)
        prompt = sources / "prompts" / f"{DIFFICULTY_PROMPT_VERSION}.txt"
        original_prompt = Path(pool_models.__file__).parent / "prompts" / prompt.name
        text = original_prompt.read_text(encoding="utf-8")
        prompt.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
        module_files = {}
        for module in (difficulty, pool_models):
            copied = sources / Path(module.__file__).name
            copied.write_bytes(Path(module.__file__).read_bytes())
            module_files[module] = str(copied)
        with patch.object(difficulty, "__file__", module_files[difficulty]), \
                patch.object(pool_models, "__file__", module_files[pool_models]):
            original, _, review_path = self.prepare_manual_review()
            self.assertEqual(original["source_sha256"][str(prompt)], seed_pool._sha(prompt))
            self.assertNotEqual(seed_pool._sha(prompt), sha256_text(text))
            pool = SQLitePoolStore(self.run / "seed.sqlite3", record_runs=False)
            members = pool.load_active_members(namespace=NAMESPACE, data_version=DATA_VERSION)
            self.assertTrue(all(member["difficulty"].assessment.prompt_sha256 == sha256_text(text)
                                for member in members))
            self.assertTrue(self.apply_manual_review(review_path)["ready"])
            # Even a newline-only source change must still invalidate byte provenance.
            prompt.write_bytes(text.encode("utf-8"))
            with self.assertRaisesRegex(ValueError, "来源发生变化"):
                self.apply_manual_review(review_path)

    def test_manual_review_rejects_wrong_source_or_original_level(self):
        _, review, path = self.prepare_manual_review()
        for field in ("hash", "level"):
            with self.subTest(field=field):
                changed = deepcopy(review)
                if field == "hash":
                    changed["source_run"]["seed_sha256"] = "0" * 64
                else:
                    changed["changes"][0]["from_level"] = "L1"
                path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.apply_manual_review(path)
                self.assertFalse((self.root / "reviewed").exists())

    def test_manual_review_index_failure_keeps_partial_only(self):
        _, _, path = self.prepare_manual_review()
        protected = {item: seed_pool._sha(item) for item in self.run.iterdir()}
        with patch.object(SQLiteRetrievalIndex, "publish", side_effect=ValueError("index rejected")):
            result = self.apply_manual_review(path)
        self.assertFalse(result["ready"])
        self.assertFalse(result["manual_review"]["applied"])
        self.assertEqual(result["error"]["stage"], "rebind_index")
        self.assertFalse((self.root / "reviewed" / "seed.sqlite3").exists())
        self.assertTrue((self.root / "reviewed" / "seed.partial.sqlite3").is_file())
        self.assertTrue((self.root / "reviewed" / "summary.json").is_file())
        for item, value in protected.items():
            self.assertEqual(seed_pool._sha(item), value)

    def test_manual_review_changed_input_prevents_publication(self):
        _, _, path = self.prepare_manual_review()
        publish = SQLiteRetrievalIndex.publish

        def change_review(index, **kwargs):
            result = publish(index, **kwargs)
            path.write_bytes(path.read_bytes() + b"\n")
            return result

        with patch.object(SQLiteRetrievalIndex, "publish", new=change_review):
            result = self.apply_manual_review(path)
        self.assertFalse(result["ready"])
        self.assertEqual(result["error"]["stage"], "publication")
        self.assertIn("来源发生变化", result["error"]["message"])
        self.assertFalse((self.root / "reviewed" / "seed.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
