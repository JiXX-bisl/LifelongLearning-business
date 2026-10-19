"""用户独立运行：直接比较入口、静态输入、单题输出和中断边界。

查询、检索、经验更新和SQLite完成标记均使用模拟对象，不发起真实请求。
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from src.psu_memory.cli.validate_local import main
from src.psu_memory.validation import reporting, runner
from src.psu_memory.validation.common import POLICY, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


class DirectRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "run"
        self.cases = [
            {"id": "q1", "session": "session", "turn": 1, "question": "甲公司怎么样？", "history": []},
            {"id": "q2", "session": "session", "turn": 2, "question": "再看一下人数", "history": [
                {"role": "user", "content": "甲公司怎么样？"}]},
        ]
        self.standards = {"q1": {"columns": ["value"], "rows": [[1]]},
                          "q2": {"columns": ["value"], "rows": [[2]]}}
        self.public = [{"role": "user", "content": "当前数据基准日为2026-08-31。"}]
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        write_json(self.baseline / "manifest.json", {
            "baseline_id": "fixture-v1", "schema_version": "3.0",
            "mysql": {"database": "fixture_db", "version": "8.0"},
        })
        (self.baseline / "database_schema.sql").write_text("CREATE TABLE fixture(value DECIMAL(20,2));", encoding="utf-8")
        self.rules_path, self.mapping_path = self.root / "rules.txt", self.root / "mapping.json"
        self.rules_path.write_text("使用公开字段完成查询。", encoding="utf-8")
        write_json(self.mapping_path, {"fields": {"value": "数值"}})
        self.questions_path = self.root / "questions.json"
        write_json(self.questions_path, {"data_version": "fixture-v1", "status": "REVIEWED",
                                        "reviewed_by_user": True, "cases": [*self.cases, {
                                            "id": "clarify", "session": "clarify", "turn": 1,
                                            "question": "是哪一家？", "history": []}]})
        self.truth_path = self.root / "ground_truth.json"
        write_json(self.truth_path, {"data_version": "fixture-v1", "reviewed_by_user": True,
                                    "cases": [{"id": case_id, "response_type": "answer", "expected_result": expected}
                                              for case_id, expected in self.standards.items()] + [
                                                  {"id": "clarify", "response_type": "clarify"}]})
        self.business_path = self.root / "business.yaml"
        self.business_path.write_text(yaml.safe_dump({"data": {
            "baseline_dir": str(self.baseline), "expected_baseline_id": "fixture-v1",
            "expected_schema_version": "3.0"}, "prompt": {
                "profile": "basic_field_mapping_v1", "rules_file": str(self.rules_path),
                "field_mapping_file": str(self.mapping_path)}}), encoding="utf-8")
        public_path = self.root / "public.json"
        write_json(public_path, {"data_version": "fixture-v1", "messages": self.public})
        self.config = {"policy": POLICY, "namespace": "direct-test", "data_version": "fixture-v1",
                       "conditions": ["baseline", "fixed", "adaptive"], "initial_state": "empty",
                       "source_store": None, "questions": str(self.questions_path),
                       "ground_truth": str(self.truth_path), "public_context": str(public_path),
                       "business_config": str(self.business_path),
                       "retrieval_config": str(ROOT / "src/psu_memory/retrieval/retrieval_config.example.json"),
                       "pool_config": str(ROOT / "src/psu_memory/pool/pool_config.example.json"),
                       "exclude_case_ids": ["clarify"], "require_dataset_acceptance": False}
        self.config_path = self.root / "source_config.json"
        write_json(self.config_path, self.config)

    def prepared_fixture(self):
        self.run.mkdir()
        for mode in self.config["conditions"]:
            folder = self.run / mode
            folder.mkdir()
            (folder / "history.sqlite3").write_bytes(b"opaque mocked store")
        write_json(self.run / "config.json", {
            "policy": POLICY, "run_id": "fixture-run", "namespace": "direct-test",
            "data_version": "fixture-v1", "conditions": self.config["conditions"],
            "initial_state": "empty", "cases": self.cases, "standards": self.standards,
            "public_context": self.public, "business_config": str(self.business_path),
            "retrieval_config": self.config["retrieval_config"], "pool_config": self.config["pool_config"],
        })

    @staticmethod
    def record(query_id, value):
        return {"status": "completed", "result_query_id": query_id, "answer": "实际回答",
                "operations": [{"kind": "sql", "query_id": query_id, "sql": "SELECT " + str(value),
                                "success": True, "is_complete": True,
                                "columns": ["actual_value"], "rows": [[value]]}]}

    @contextmanager
    def runtime(self, records):
        service, retriever, updater = Mock(), Mock(), Mock()
        histories = []
        pending = iter(records)

        def query(question):
            histories.append(deepcopy(service._history.model_dump(mode="json")["messages"]))
            item = next(pending)
            if isinstance(item, BaseException):
                raise item
            return item

        service.query.side_effect = query
        completion = {}
        with patch.object(runner, "_make_service", return_value=service) as make_service, \
             patch.object(runner, "_make_retriever", return_value=retriever) as make_retriever, \
             patch.object(runner, "_make_updater", return_value=updater) as make_updater, \
             patch.object(runner, "_set_completed", side_effect=lambda path, state: completion.__setitem__(Path(path), state)), \
             patch.object(runner, "condition_completed", side_effect=lambda path: completion.get(Path(path), False)), \
             patch.object(reporting, "condition_completed", side_effect=lambda path: completion.get(Path(path), False)), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield service, retriever, updater, make_service, make_retriever, make_updater, histories

    def test_prepare_snapshots_selected_cases_and_standards_without_querying(self):
        def initialize(path, **kwargs):
            Path(path).write_bytes(b"same mocked initial store")

        with patch("src.psu_memory.integration.bootstrap.initialize_empty_store", side_effect=initialize) as bootstrap, \
             patch("src.psu_memory.integration.common.copy_database", side_effect=shutil.copyfile), \
             patch.object(runner, "_set_completed") as mark_completed, \
             patch.object(runner, "_make_service", side_effect=AssertionError("prepare must not query")):
            result = runner.prepare(self.config_path, self.run)
            with self.assertRaises(FileExistsError):
                runner.prepare(self.config_path, self.run)
        self.assertEqual(result["case_count"], 2)
        self.assertFalse(bootstrap.call_args.kwargs["record_runs"])
        saved = read_json(self.run / "config.json")
        self.assertEqual(saved["cases"], self.cases)
        self.assertEqual(saved["standards"], self.standards)
        self.assertEqual(saved["public_context"], self.public)
        copies = [(self.run / mode / "history.sqlite3").read_bytes() for mode in self.config["conditions"]]
        self.assertEqual(len(set(copies)), 1)
        self.assertEqual(mark_completed.call_count, len(self.config["conditions"]))
        self.assertTrue(all(call.args[1] is False for call in mark_completed.call_args_list))
        business = yaml.safe_load((self.run / "business.yaml").read_text())
        self.assertNotIn("ground_truth", business)
        self.assertEqual(business["data"]["baseline_dir"], str(self.run / "inputs"))
        self.assertEqual(Path(business["prompt"]["rules_file"]).read_bytes(), self.rules_path.read_bytes())
        snapshots = {path.name: path.read_bytes() for path in (self.run / "inputs").iterdir()}
        self.rules_path.write_text("后来修改的公共规则。", encoding="utf-8")
        self.mapping_path.unlink()
        (self.baseline / "database_schema.sql").unlink()
        write_json(self.truth_path, {"cases": []})
        write_json(self.questions_path, {"cases": []})
        _, loaded, cases = runner.load_run(self.run)
        self.assertEqual(cases, self.cases)
        self.assertEqual(loaded["standards"], self.standards)
        self.assertEqual({path.name: path.read_bytes() for path in (self.run / "inputs").iterdir()}, snapshots)

    def test_missing_standard_and_partial_session_exclusion_fail_before_creating_run(self):
        truth = read_json(self.truth_path)
        del truth["cases"][0]["expected_result"]
        write_json(self.truth_path, truth)
        with self.assertRaises(ValueError):
            runner.prepare(self.config_path, self.run)
        self.assertFalse(self.run.exists())
        truth["cases"][0]["expected_result"] = self.standards["q1"]
        write_json(self.truth_path, truth)
        write_json(self.config_path, {**self.config, "exclude_case_ids": ["clarify", "q1"]})
        with self.assertRaises(ValueError):
            runner.prepare(self.config_path, self.run)
        self.assertFalse(self.run.exists())

    def test_baseline_writes_one_record_per_case_and_keeps_public_history(self):
        self.prepared_fixture()
        with self.runtime([self.record("actual-q1", 1), RuntimeError("query unavailable")]) as runtime:
            service, _, _, _, make_retriever, make_updater, histories = runtime
            result = runner.run_condition(self.run, "baseline")
        self.assertEqual((result["pass"], result["fail"], result["recorded"]), (1, 1, 2))
        self.assertEqual(result["pass_rate"], .5)
        make_retriever.assert_not_called()
        make_updater.assert_not_called()
        service.close.assert_called_once()
        self.assertEqual(histories, [self.public, [*self.public, *self.cases[1]["history"]]])
        for case in self.cases:
            row = read_json(self.run / "baseline" / "results" / (case["id"] + ".json"))
            self.assertEqual(set(row), {"case_id", "condition", "question", "query_id", "sql", "actual", "expected", "verdict"})
        self.assertEqual({path.name for path in (self.run / "baseline").iterdir()}, {"history.sqlite3", "results"})

    def test_adaptive_updates_only_passed_sql_and_complete_group_is_not_requeried(self):
        self.prepared_fixture()
        records = [self.record("actual-q1", 1), self.record("actual-q2", 999)]
        with self.runtime(records) as runtime:
            service, retriever, updater, make_service, _, _, _ = runtime
            result = runner.run_condition(self.run, "adaptive")
            again = runner.run_condition(self.run, "adaptive")
        self.assertEqual(result, again)
        make_service.assert_called_once()
        self.assertEqual(service.query.call_count, 2)
        updater.apply.assert_called_once()
        call = updater.apply.call_args.kwargs
        self.assertEqual(call["query_id"], "actual-q1")
        self.assertEqual(call["record"], records[0])
        self.assertNotIn("expected", call["record"])
        self.assertNotIn("expected_result", call["case"])
        self.assertEqual(call["record_path"], self.run / "adaptive" / "results" / "q1.json")
        updater.close.assert_called_once()
        retriever.close.assert_called_once()

    def test_fixed_retrieves_but_never_updates(self):
        self.prepared_fixture()
        with self.runtime([self.record("actual-q1", 1), self.record("actual-q2", 2)]) as runtime:
            _, _, _, _, make_retriever, make_updater, _ = runtime
            result = runner.run_condition(self.run, "fixed")
        self.assertTrue(result["completed"])
        make_retriever.assert_called_once()
        make_updater.assert_not_called()

    def test_safely_rolled_back_update_failure_allows_the_next_case(self):
        from src.psu_memory.integration.direct_update import DirectUpdateError

        self.prepared_fixture()
        with self.runtime([self.record("actual-q1", 1), self.record("actual-q2", 999)]) as runtime:
            runtime[2].apply.side_effect = DirectUpdateError("rolled back")
            result = runner.run_condition(self.run, "adaptive")
        self.assertTrue(result["completed"])
        self.assertEqual(result["recorded"], 2)

    def test_update_integrity_failure_stops_the_group(self):
        from src.psu_memory.integration.direct_update import DirectUpdateIntegrityError

        self.prepared_fixture()
        with self.runtime([self.record("actual-q1", 1), self.record("actual-q2", 2)]) as runtime:
            runtime[2].apply.side_effect = DirectUpdateIntegrityError("unknown pool state")
            with self.assertRaises(DirectUpdateIntegrityError):
                runner.run_condition(self.run, "adaptive")
            self.assertEqual(runtime[0].query.call_count, 1)
            partial = reporting.summarize(self.run)["conditions"]["adaptive"]
            self.assertFalse(partial["completed"])
            self.assertIsNone(partial["pass_rate"])

    def test_interrupted_group_rejects_restart_and_incomplete_summary_has_no_rate(self):
        self.prepared_fixture()
        with self.runtime([self.record("actual-q1", 1), KeyboardInterrupt()]) as runtime:
            service = runtime[0]
            with self.assertRaises(KeyboardInterrupt):
                runner.run_condition(self.run, "baseline")
            partial = reporting.summarize(self.run)["conditions"]["baseline"]
            self.assertEqual(partial["recorded"], 1)
            self.assertIsNone(partial["pass_rate"])
            with self.assertRaises(ValueError):
                runner.run_condition(self.run, "baseline")
            self.assertEqual(service.query.call_count, 2)

    def test_old_directory_is_rejected_and_listing_never_constructs_query_service(self):
        old = self.root / "old_run"
        old.mkdir()
        write_json(old / "manifest.json", {"version": "psu_local_validation_v2"})
        with self.assertRaises(ValueError):
            runner.load_run(old)
        with patch.object(runner, "_make_service", side_effect=AssertionError("listing must not query")), \
             redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["list-questions", "--config", str(self.config_path)]), 0)
        self.assertIn("SQL测试题数：2", stream.getvalue())
        self.assertNotIn("是哪一家？", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
