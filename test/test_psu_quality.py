"""只使用临时文件与临时候选库；不访问真实业务数据库或模型。"""
from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from src.psu_memory.evaluation.evaluation_evidence import attach_evaluations
from src.psu_memory.history.history_adapter import read_history_file
from src.psu_memory.history.history_store import SQLiteHistoryStore
from src.psu_memory.evaluation.quality_models import Feedback, digest, ranking_key
from src.psu_memory.evaluation.quality_policy import evaluate_candidate, original_content
from src.psu_memory.evaluation.quality_store import SQLiteQualityStore


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "sessions/s1/0001.json"
        self.log.parent.mkdir(parents=True)
        self.db = self.root / "history.sqlite3"
        self.raw = {
            "baseline_id": "v1", "session_id": "s1", "turn_index": 1,
            "mode": "psu", "experiment_id": "local",
            "question": "查询未缴社保企业数量", "answer": "2 家", "status": "completed",
            "evaluation": {"verdict": "pass"},
            "messages": [{"role": "user", "content": "查询未缴社保企业数量"}],
            "operations": [{"kind": "sql", "query_id": "q1", "sql": "SELECT COUNT(enterprise_id) FROM staff;",
                            "success": True, "is_complete": True, "columns": ["count"], "rows": [[2]],
                            "scope": {"enterprise": "全部企业", "time_range": "当前"}}],
            "metrics": {},
        }
        self.log.write_text(json.dumps(self.raw, ensure_ascii=False))
        self.record = read_history_file(self.log, namespace="test", data_version="v1")
        self.key = self.record.task.task_id
        self._import(self.record)
        self.acceptance = {
            "dataset_id": "test", "settings": {"data": {"baseline_id": "v1"},
                "learning": {"mode": "psu", "experiment_id": "local"}},
            "results": [{"record_path": str(self.log), "task_id": "Q1",
                         "b1": {"matched": True, "reason": "matched", "query_id": "q1"},
                         "independent_review": {"verdict": "pending"}}],
        }
        self._attach()
        self.store = SQLiteQualityStore(self.db)

    def _import(self, record):
        return SQLiteHistoryStore(self.db).import_batch(
            [record], namespace="test", data_version="v1", source_root=str(self.root),
            scanned_files=1, errors=[], started_at=perf_counter())

    def _attach(self):
        (self.root / "acceptance.json").write_text(json.dumps(self.acceptance, ensure_ascii=False))
        return attach_evaluations(self.db, self.root, namespace="test", data_version="v1", resolve_path=Path)

    def _profile(self):
        with sqlite3.connect(self.db) as c:
            return json.loads(c.execute("SELECT v.profile_json FROM evaluation_versions v "
                "JOIN evaluation_current e USING(execution_id,profile_sha256)").fetchone()[0])

    def _batch(self):
        return self.store.assess_batch(namespace="test", data_version="v1")

    def _decision(self):
        return self._batch()["decisions"][0]

    def _state(self):
        with sqlite3.connect(self.db) as c:
            return json.loads(c.execute("SELECT v.decision_json FROM quality_versions v JOIN quality_current q "
                "USING(execution_id,content_version,decision_sha256)").fetchone()[0])

    def _pure(self, record=None, profile=None):
        record = record or asdict(self.record)
        profile = profile if profile is not None else self._profile()
        content = original_content(record, profile)
        return evaluate_candidate(record, profile, content, source={"status": "verified"})

    def test_reference_pass_and_repeat(self):
        first = self._batch()
        second = self._batch()
        self.assertEqual(first["states"], {"eligible": 1, "suspended": 0})
        self.assertEqual(second["new_quality_versions"], 0)
        self.assertEqual(second["refreshed_current_decisions"], 0)
        self.assertEqual(first["ranked_execution_ids"], second["ranked_execution_ids"])
        self.assertEqual(first["usage"]["model_call_count"], 0)
        self.assertEqual(first["admission_changes"], 0)

    def test_internal_and_positive_must_both_exist(self):
        profile = {"user_feedback": None}
        self.assertEqual(self._pure(profile=profile).admission_state, "suspended")
        profile["user_feedback"] = "helpful"
        accepted = self._pure(profile=profile)
        self.assertEqual(accepted.admission_state, "eligible")
        self.assertIsNone(accepted.effective_correctness)
        record = asdict(self.record)
        record["task"]["evaluation_verdict"] = "unknown"
        self.assertEqual(self._pure(record=record, profile=profile).admission_state, "suspended")

    def test_no_evaluation_suspends(self):
        self.assertEqual(self._pure(profile={}).admission_state, "suspended")

    def test_explicit_error_dominates_positive(self):
        for failure in ("reference", "execution", "source", "review"):
            with self.subTest(failure=failure):
                record, profile = asdict(self.record), self._profile()
                profile["user_feedback"] = "helpful"
                source = {"status": "verified"}
                if failure == "reference":profile["reference_result_match"] = False
                if failure == "execution":record["task"]["status"] = "failed"
                if failure == "source":source = {"status": "conflict", "conflicts": ["source_content_changed"]}
                if failure == "review":
                    profile["evidence"][0]["details"]["independent_review"] = {
                        "scope": "sql", "query_id": "q1", "verdict": "fail"}
                decision = evaluate_candidate(record, profile, original_content(record, profile), source=source)
                self.assertEqual(decision.admission_state, "suspended")

    def test_scoped_review_and_ranking(self):
        review = {"verdict": "pass", "scope": "sql", "query_id": "q1"}
        profile = self._profile()
        profile["evidence"][0]["details"]["independent_review"] = review
        combined = self._pure(profile=profile)
        self.assertEqual(combined.validation_level, 3)
        reference_only = self._pure()
        self.assertLess(ranking_key(combined), ranking_key(reference_only))
        profile["reference_result_match"] = None
        profile["evidence"][0]["signals"]["reference_result_match"] = None
        self.assertEqual(self._pure(profile=profile).admission_state, "eligible")
        review.pop("scope")
        self.assertEqual(self._pure(profile=profile).admission_state, "suspended")

    def test_uses_exact_reference_query(self):
        record = asdict(self.record)
        operations = json.loads(record["task"]["evidence_text"])
        other = copy.deepcopy(operations[0]);other["query_id"] = "q2";other["sql"] = "SELECT 999;"
        record["task"]["evidence_text"] = json.dumps([*operations, other])
        result = self._pure(record=record)
        self.assertEqual(result.admission_state, "eligible")
        self.assertEqual(result.content["operation"]["query_id"], "q1")
        profile = self._profile();profile["evidence"][0]["details"]["legacy_b1"]["query_id"] = "missing"
        self.assertEqual(self._pure(profile=profile).admission_state, "suspended")

    def test_incomplete_sql_cannot_use_reference_pass(self):
        record = asdict(self.record)
        operations = json.loads(record["task"]["evidence_text"]);operations[0]["is_complete"] = False
        record["task"]["evidence_text"] = json.dumps(operations)
        self.assertEqual(self._pure(record=record).admission_state, "suspended")

    def test_missing_original_file_preserves_evidence(self):
        self.log.unlink()
        attached = self._attach()
        self.assertEqual(attached["status"], "completed")
        self.assertTrue(attached["warnings"])
        result = self._decision()
        self.assertEqual(result["admission_state"], "eligible")
        self.assertEqual(result["source"]["status"], "unverifiable")

    def test_known_hash_change_is_blocked(self):
        self.log.write_text(self.log.read_text() + " ")
        result = self._decision()
        self.assertEqual(result["admission_state"], "suspended")
        self.assertIn("source_content_changed", result["reasons"])
        self.assertEqual(self._attach()["status"], "blocked")

    def test_missing_source_version_and_known_identity(self):
        again = read_history_file(self.log, namespace="test", data_version="v1")
        self.assertEqual(again.task.task_id, self.record.task.task_id)
        raw = copy.deepcopy(self.raw);raw.pop("baseline_id")
        path = self.root / "unversioned.json";path.write_text(json.dumps(raw))
        converted = read_history_file(path, namespace="legacy", data_version="target")
        self.assertIsNone(converted.provenance["original_data_version"])
        self.assertEqual(converted.task.data_version, "target")
        from src.psu_memory.history.source_checks import inspect_source
        source = inspect_source(asdict(converted), converted.source_sha256, "pending_review", Path)
        self.assertEqual(source["status"], "verified")
        self.assertIn("original_version_unknown", source["warnings"])
        imported = SQLiteHistoryStore(self.root / "legacy.sqlite3").import_batch(
            [converted], namespace="legacy", data_version="target", source_root=str(self.root),
            scanned_files=1, errors=[], started_at=perf_counter())
        self.assertEqual(imported["inserted"], 1)
        with self.assertRaises(ValueError):read_history_file(self.log, namespace="test", data_version="v2")

    def test_negative_is_immediate_and_like_does_not_restore(self):
        current = self._decision()
        problem = Feedback(self.key, current["content_version"], "problem", comment="没有去重")
        self.store.submit_feedback(problem)
        self.assertEqual(self._state()["admission_state"], "suspended")
        self.assertIs(self._state()["effective_correctness"], False)
        self.store.submit_feedback(Feedback(self.key, current["content_version"], "helpful"))
        self.assertEqual(self._decision()["admission_state"], "suspended")

    def test_correction_versions_notes_and_replay(self):
        original = self._decision()
        negative = Feedback(self.key, original["content_version"], "problem", comment="企业重复",
                            notes="企业数量按企业唯一标识去重", applicability="统计企业数量")
        self.store.submit_feedback(negative)
        corrected = Feedback(self.key, original["content_version"], "correction",
            corrected_sql="SELECT COUNT(DISTINCT enterprise_id) FROM staff;", confirmed=True, actor="tester")
        result = self.store.submit_feedback(corrected)["decision"]
        self.assertEqual(result["admission_state"], "eligible")
        self.assertNotEqual(result["content_version"], original["content_version"])
        self.assertFalse(result["execution_verified"])
        self.assertEqual(result["basis"], ["user_confirmed_correction"])
        self.assertIn("去重", result["content"]["notes"])
        self.assertEqual(self._decision()["content_version"], result["content_version"])
        replay = self.store.submit_feedback(corrected)
        self.assertTrue(replay["replayed"])
        self.assertEqual(self.store.submit_feedback(negative)["decision"]["admission_state"], "eligible")
        with sqlite3.connect(self.db) as c:
            old = json.loads(c.execute("SELECT decision_json FROM quality_versions WHERE content_version=? "
                "ORDER BY created_at DESC LIMIT 1", (original["content_version"],)).fetchone()[0])
            self.assertEqual(old["admission_state"], "suspended")
            self.assertEqual(c.execute("SELECT count(*) FROM quality_contents").fetchone()[0], 2)

    def test_unconfirmed_proposal_does_not_restore(self):
        original = self._decision()
        feedback = Feedback(self.key, original["content_version"], "problem",
            corrected_sql="SELECT COUNT(DISTINCT enterprise_id) FROM staff;", notes="注意去重")
        result = self.store.submit_feedback(feedback)["decision"]
        self.assertEqual(result["admission_state"], "suspended")
        self.assertEqual(result["content_version"], original["content_version"])

    def test_stale_version_and_event_id_collision(self):
        original = self._decision()
        self.store.submit_feedback(Feedback(self.key, original["content_version"], "correction",
            corrected_sql="SELECT COUNT(DISTINCT enterprise_id) FROM staff;", confirmed=True, event_id="fix-1"))
        with self.assertRaises(ValueError):
            self.store.submit_feedback(Feedback(self.key, original["content_version"], "problem"))
        with self.assertRaises(ValueError):
            self.store.submit_feedback(Feedback(self.key, original["content_version"], "problem", event_id="fix-1"))

    def test_same_sql_cannot_restore_error(self):
        original = self._decision()
        with self.assertRaises(ValueError):
            self.store.submit_feedback(Feedback(self.key, original["content_version"], "correction",
                corrected_sql=original["content"]["sql"], confirmed=True))

    def test_feedback_transaction_rolls_back(self):
        original = self._decision()
        with patch.object(self.store, "_save_decision", side_effect=RuntimeError("simulated failure")):
            with self.assertRaises(RuntimeError):
                self.store.submit_feedback(Feedback(self.key, original["content_version"], "problem"))
        self.assertEqual(self._state()["admission_state"], "eligible")
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute("SELECT count(*) FROM quality_feedback").fetchone()[0], 0)

    def test_legacy_feedback_history_is_sticky(self):
        path = self.root / "feedback/s1/0001.json";path.parent.mkdir(parents=True)
        feedback = {"source_record_path": str(self.log), "rating": "problem", "comment": "未去重"}
        path.write_text(json.dumps(feedback));self._attach()
        feedback["rating"] = "helpful";path.write_text(json.dumps(feedback));self._attach()
        self.assertEqual(self._decision()["admission_state"], "suspended")
        with sqlite3.connect(self.db) as c:
            count = c.execute("SELECT count(*) FROM quality_feedback").fetchone()[0]
        self._batch()
        with sqlite3.connect(self.db) as c:
            self.assertEqual(c.execute("SELECT count(*) FROM quality_feedback").fetchone()[0], count)

    def test_import_compatibility_and_original_tables_unchanged(self):
        with sqlite3.connect(self.db) as c:
            before = c.execute("SELECT * FROM history_records").fetchall()
            evidence_before = c.execute("SELECT * FROM evaluation_versions").fetchall()
        current = self._decision()
        self.store.submit_feedback(Feedback(self.key, current["content_version"], "problem"))
        self.assertEqual(self._import(self.record)["skipped"], 1)
        with sqlite3.connect(self.db) as c:
            self.assertEqual(before, c.execute("SELECT * FROM history_records").fetchall())
            self.assertEqual(evidence_before, c.execute("SELECT * FROM evaluation_versions").fetchall())
            self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], 1)

    def test_wrong_namespace_is_not_silently_used(self):
        with self.assertRaises(ValueError):self.store.assess_batch(namespace="other", data_version="v1")

    def test_does_not_create_missing_database(self):
        missing = self.root / "typo.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            SQLiteQualityStore(missing).assess_batch(namespace="test", data_version="v1")
        self.assertFalse(missing.exists())

    def test_negative_on_corrected_version_is_not_masked(self):
        original = self._decision()
        fixed = self.store.submit_feedback(Feedback(self.key, original["content_version"], "correction",
            corrected_sql="SELECT COUNT(DISTINCT enterprise_id) FROM staff;", confirmed=True))["decision"]
        self.store.submit_feedback(Feedback(self.key, fixed["content_version"], "problem", comment="修正版仍有问题"))
        final = self._decision()
        self.assertEqual(final["admission_state"], "suspended")
        self.assertIs(final["effective_correctness"], False)

    def test_sorting_uses_completeness_then_length_and_id(self):
        full = self._pure()
        less = copy.deepcopy(full);less.completeness["query_scope"] = False;less.preview_text = "短"
        self.assertLess(ranking_key(full), ranking_key(less))
        shorter = copy.deepcopy(full);shorter.preview_text = "短"
        self.assertLess(ranking_key(shorter), ranking_key(full))
        a = copy.deepcopy(full);b = copy.deepcopy(full);a.execution_id = "a";b.execution_id = "b"
        self.assertLess(ranking_key(a), ranking_key(b))


if __name__ == "__main__":
    unittest.main()
