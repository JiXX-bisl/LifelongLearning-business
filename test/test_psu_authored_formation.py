"""由用户独立运行：预编写 SQL、只读快照、候选隔离和输入绑定。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from src.psu_memory.validation import authored_formation as formation


class FakeConnection:
    def __init__(self):
        self.live = True
        self.transactions = []
        self.rolled_back = False

    def start_transaction(self, **options):
        self.transactions.append(options)

    def is_connected(self):
        return self.live

    def rollback(self):
        self.rolled_back = True


class FakeEnvironment:
    def __init__(self, responses):
        self.original = self.connection = FakeConnection()
        self.responses = list(responses)
        self.queries = []
        self.closed = False
        self.counts = None

    def verify_record_counts(self, counts):
        self.counts = counts

    def execute_query(self, sql):
        self.queries.append(sql)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            self.connection = None
            raise response
        if response.get("is_complete") is False:
            self.connection = None
        return deepcopy(response)

    def close(self):
        self.closed = True


class AuthoredFormationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.expected = {"columns": ["amount"], "rows": [["12.30"]]}
        cases = [{"id": "q" + str(i), "question": "查询企业金额" + str(i),
                  "history": [], "session": "single-q" + str(i), "turn": 1}
                 for i in range(1, 4)]
        self.bundle = formation.AuthoredBundle(
            path=self.root / "bundle.json", sha256="bundle-hash", namespace=formation.NAMESPACE,
            cases=cases, statements={case["id"]: "SELECT 12.30 AS amount FROM enterprise_basic LIMIT 1"
                                    for case in cases},
            standards={case["id"]: deepcopy(self.expected) for case in cases},
            record_counts={"enterprise_basic": 120}, source_hashes={"ground_truth": "private-hash"},
        )

    def actual(self, rows=None, **overrides):
        rows = [["12.300"]] if rows is None else rows
        return {"columns": ["different_alias"], "rows": rows, "row_count": len(rows),
                "is_complete": True, **overrides}

    def execute(self, responses):
        environment = FakeEnvironment(responses)
        with patch.object(formation, "load_bundle", return_value=self.bundle), \
                patch.object(formation, "_make_environment", return_value=environment) as factory:
            summary = formation.run_authored_formation(run_dir=self.run)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(environment.closed)
        self.assertEqual(environment.original.transactions, [{"isolation_level": "REPEATABLE READ",
                                                            "consistent_snapshot": True, "readonly": True}])
        return summary, environment

    def test_only_actual_direct_pass_enters_candidate_without_pool_or_model(self):
        # 列别名不同和Decimal尾零不影响现行直接比较；重复行不会被丢弃。
        summary, environment = self.execute([self.actual(), self.actual([["12.30"], ["12.30"]]),
                                             self.actual([[None]])])
        self.assertEqual((summary["pass_count"], summary["fail_count"], summary["candidate_count"]),
                         (1, 2, 1))
        self.assertTrue(summary["candidate_save_complete"])
        self.assertTrue(summary["execution_complete"])
        self.assertEqual(summary["cost"]["model_call_count"], 0)
        self.assertEqual(summary["cost"]["prompt_tokens"], 0)
        self.assertEqual(len(environment.queries), 3)
        self.assertTrue(environment.original.rolled_back)
        self.assertEqual(sorted(path.name for path in self.run.iterdir()),
                         ["candidates.sqlite3", "results", "summary.json"])
        with sqlite3.connect(self.run / "candidates.sqlite3") as connection:
            records = connection.execute("SELECT record_json, source_sha256 FROM history_records").fetchall()
            self.assertEqual(len(records), 1)
            historical = json.loads(records[0][0])
            self.assertEqual(historical["task"]["answer"], "")
            evidence = json.loads(historical["task"]["evidence_text"])[0]
            self.assertEqual(evidence["rows"], [["12.300"]])
            self.assertEqual(evidence["columns"], ["different_alias"])
            self.assertNotIn("expected", historical["task"]["evidence_text"])
            source = self.run / "results/q1.json"
            self.assertEqual(records[0][1], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotIn("expected", json.loads(source.read_text()))
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("psu_pool_members", tables)
            self.assertNotIn("psu_retrieval_index_members", tables)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0], 0)

    def test_connection_loss_does_not_reconnect_or_invent_results(self):
        summary, environment = self.execute([self.actual(), RuntimeError("lost snapshot")])
        self.assertEqual(len(environment.queries), 2)
        self.assertEqual((summary["executed_count"], summary["not_executed_count"]), (2, 1))
        self.assertEqual(summary["candidate_count"], 1)
        self.assertFalse(summary["execution_complete"])
        last = json.loads((self.run / "results/q3.json").read_text())
        self.assertEqual(last["verdict"], "fail")
        self.assertFalse(last["executed"])
        self.assertIsNone(last["actual"])
        self.assertEqual(last["error"]["reason"], "not_executed")
        self.assertEqual(last["error"]["cause"], "snapshot_connection_lost")

    def test_truncated_output_cannot_be_saved_as_success(self):
        summary, environment = self.execute([self.actual(is_complete=False)])
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(summary["fail_count"], 3)
        self.assertEqual(len(environment.queries), 1)
        first = json.loads((self.run / "results/q1.json").read_text())
        self.assertEqual(first["error"]["reason"], "incomplete_result")

    def test_database_setup_failure_records_every_unexecuted_case(self):
        with patch.object(formation, "load_bundle", return_value=self.bundle), \
                patch.object(formation, "_make_environment", side_effect=RuntimeError("no password")):
            summary = formation.run_authored_formation(run_dir=self.run)
        self.assertEqual(summary["executed_count"], 0)
        self.assertEqual(summary["not_executed_count"], 3)
        self.assertEqual(summary["candidate_count"], 0)
        self.assertEqual(len(list((self.run / "results").glob("*.json"))), 3)

    def test_existing_directory_is_never_overwritten(self):
        self.run.mkdir()
        with patch.object(formation, "load_bundle", return_value=self.bundle), \
                patch.object(formation, "_make_environment") as factory:
            with self.assertRaises(FileExistsError):
                formation.run_authored_formation(run_dir=self.run)
        factory.assert_not_called()

    def test_bundle_checks_source_and_sql_hashes_before_database(self):
        # 测试输入绑定只涉及文件，不调用数据库；分区语义由已有split测试覆盖。
        project = self.root / "project"
        project.mkdir()
        documents = {name: {} for name in formation.SOURCE_NAMES}
        documents["questions"] = {"data_version": formation.DATA_VERSION, "reviewed_by_user": True,
                                  "cases": self.bundle.cases}
        documents["ground_truth"] = {
            "data_version": formation.DATA_VERSION, "reviewed_by_user": True,
            "cases": [{"id": case["id"], "expected_result": self.expected,
                       "reference_sql": "PRIVATE_SQL_MUST_NOT_EXECUTE"} for case in self.bundle.cases]}
        documents["business_context"] = {"data_version": formation.DATA_VERSION}
        documents["dataset_manifest"] = {
            "baseline_id": formation.DATA_VERSION, "schema_version": "3.0",
            "mysql": {"database": formation.DATABASE},
            "record_counts": {"table_" + str(i): i for i in range(7)}}
        bindings = {}
        for name, document in documents.items():
            source = project / (name + ".json")
            source.write_text(json.dumps(document), encoding="utf-8")
            bindings[name] = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        sql = project / "authored.sql"
        sql.write_text("SELECT 12.30 AS amount", encoding="utf-8")
        authored = [{"case_id": case["id"], "sql_file": sql.name,
                     "sql_sha256": hashlib.sha256(sql.read_bytes()).hexdigest()} for case in self.bundle.cases]
        bundle_file = project / "bundle.json"
        bundle_file.write_text(json.dumps({"version": formation.VERSION, "data_version": formation.DATA_VERSION,
                                          "namespace": formation.NAMESPACE, "sources": bindings, "cases": authored}))
        with patch.object(formation, "resolve_project_path", side_effect=lambda value: (project / value).resolve()), \
                patch.object(formation, "select_cases", return_value=self.bundle.cases):
            loaded = formation.load_bundle(bundle_file)
            self.assertNotIn("PRIVATE_SQL", repr(loaded))
            sql.write_text("SELECT 999 AS amount", encoding="utf-8")
            self.assertEqual(loaded.statements["q1"], "SELECT 12.30 AS amount")
            with self.assertRaisesRegex(ValueError, "预编写 SQL 已变化"):
                formation.load_bundle(bundle_file)
            source = project / "questions.json"
            source.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "来源文件已变化"):
                formation.load_bundle(bundle_file)


if __name__ == "__main__":
    unittest.main()
