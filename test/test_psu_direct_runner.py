"""用户独立运行：直接比较入口、静态输入、单题输出和中断边界。

查询、检索、经验更新和SQLite完成标记均使用模拟对象，不发起真实请求。
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
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
        self.comparison_contexts = {
            case["id"]: {"enterprises": [], "amount_unit": "元"} for case in self.cases}
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
                       "conditions": ["baseline", "fifo", "adaptive", "fixed"], "initial_state": "empty",
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
            "implementation_version": runner.IMPLEMENTATION_VERSION,
            "comparison_version": runner.COMPARISON_VERSION,
            "comparison_contexts": self.comparison_contexts,
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
        configurations = {call.args[0].parent.name: call.kwargs["pool_config"]
                          for call in bootstrap.call_args_list}
        self.assertEqual(configurations["fifo"].strategy, "fifo")
        self.assertEqual(configurations["fifo"].fifo_capacity, 18)
        self.assertEqual(configurations["adaptive"].strategy, "adaptive")
        self.assertEqual(sum(configurations["adaptive"].capacities.values()), 32)
        saved = read_json(self.run / "config.json")
        self.assertEqual(saved["cases"], self.cases)
        self.assertEqual(saved["standards"], self.standards)
        self.assertEqual(saved["public_context"], self.public)
        self.assertEqual(saved["implementation_version"], "dev_v0.1")
        self.assertEqual(saved["comparison_version"], runner.COMPARISON_VERSION)
        self.assertEqual(saved["comparison_contexts"], self.comparison_contexts)
        self.assertEqual(saved["pool_configs"]["fifo"]["strategy"], "fifo")
        copies = [(self.run / mode / "history.sqlite3").read_bytes() for mode in self.config["conditions"]]
        self.assertEqual(len(set(copies)), 1)
        self.assertEqual(mark_completed.call_count, len(self.config["conditions"]))
        self.assertTrue(all(call.args[1] is False for call in mark_completed.call_args_list))
        business = yaml.safe_load((self.run / "business.yaml").read_text())
        self.assertNotIn("ground_truth", business)
        self.assertNotIn("comparison_contexts", business)
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

    def test_prepare_freezes_private_enterprise_identity_context_from_static_data(self):
        enterprises = [
            {"enterprise_id": "E1", "enterprise_name": "甲公司", "industry": "不传入核验背景"},
            {"enterprise_id": "E2", "enterprise_name": "乙公司", "industry": "不传入核验背景"},
        ]
        write_json(self.baseline / "tables.json", {"enterprise_basic": enterprises})
        truth = read_json(self.truth_path)
        truth["cases"][0]["enterprise_ids"] = ["E1"]
        truth["cases"][1]["enterprise_ids"] = ["E2", "E1"]
        write_json(self.truth_path, truth)

        def initialize(path, **_):
            Path(path).write_bytes(b"mocked empty store")

        with patch("src.psu_memory.integration.bootstrap.initialize_empty_store", side_effect=initialize), \
                patch.object(runner, "_set_completed"), \
                patch.object(runner, "_make_service", side_effect=AssertionError("prepare must not query")):
            runner.prepare(self.config_path, self.run)
        saved = read_json(self.run / "config.json")
        first = {"enterprise_id": "E1", "enterprise_name": "甲公司"}
        second = {"enterprise_id": "E2", "enterprise_name": "乙公司"}
        expected = {"q1": {"enterprises": [first], "amount_unit": "元"},
                    "q2": {"enterprises": [second, first], "amount_unit": "元"}}
        self.assertEqual(saved["comparison_contexts"], expected)
        self.assertEqual(saved["standards"], self.standards)
        self.assertEqual(saved["public_context"], self.public)
        self.assertEqual(saved["cases"], self.cases)
        self.assertFalse((self.run / "inputs" / "tables.json").exists())
        self.assertNotIn("comparison_contexts", yaml.safe_load((self.run / "business.yaml").read_text()))
        (self.baseline / "tables.json").unlink()
        write_json(self.truth_path, {"cases": []})
        self.assertEqual(runner.load_run(self.run)[1]["comparison_contexts"], expected)

    def test_missing_or_mismatched_enterprise_definitions_fail_before_store_preparation(self):
        original_truth = read_json(self.truth_path)
        valid = {"enterprise_id": "E1", "enterprise_name": "甲公司"}
        tables_path = self.baseline / "tables.json"
        with patch("src.psu_memory.integration.bootstrap.initialize_empty_store") as initialize, \
                patch("src.psu_memory.integration.common.copy_database") as copy_store, \
                patch.object(runner, "_set_completed") as mark_completed:
            for problem in ("missing_file", "missing_id", "empty_name", "duplicate_id", "mismatched_name"):
                with self.subTest(problem=problem):
                    truth = deepcopy(original_truth)
                    truth["cases"][0]["enterprise_ids"] = ["E1"]
                    if problem == "mismatched_name":
                        truth["cases"][0]["expected_result"] = {
                            "columns": ["enterprise_name"], "rows": [["其他公司"]]}
                    write_json(self.truth_path, truth)
                    enterprises = [valid]
                    if problem == "missing_id":
                        enterprises = []
                    elif problem == "empty_name":
                        enterprises = [{**valid, "enterprise_name": ""}]
                    elif problem == "duplicate_id":
                        enterprises = [valid, valid]
                    write_json(tables_path, {"enterprise_basic": enterprises})
                    if problem == "missing_file":
                        tables_path.unlink()
                    run = self.root / problem
                    with self.assertRaisesRegex(ValueError, "企业静态定义"):
                        runner.prepare(self.config_path, run)
                    self.assertFalse(run.exists())
            initialize.assert_not_called()
            copy_store.assert_not_called()
            mark_completed.assert_not_called()

    def test_case_limit_shares_questions_and_denominator_across_conditions(self):
        extra = {"id": "q3", "session": "another-session", "turn": 1,
                 "question": "另一家公司呢？", "history": []}
        questions = read_json(self.questions_path)
        questions["cases"].append(extra)
        write_json(self.questions_path, questions)
        truth = read_json(self.truth_path)
        truth["cases"].append({"id": "q3", "response_type": "answer",
                               "expected_result": {"columns": ["value"], "rows": [[3]]}})
        write_json(self.truth_path, truth)

        def initialize(path, **_):
            Path(path).write_bytes(b"same mocked initial store")

        with patch("src.psu_memory.integration.bootstrap.initialize_empty_store", side_effect=initialize), \
                patch.object(runner, "_set_completed"), \
                patch.object(runner, "_make_service", side_effect=AssertionError("prepare must not query")):
            prepared = runner.prepare(self.config_path, self.run, case_limit=2)
        saved = read_json(self.run / "config.json")
        self.assertEqual(prepared["case_count"], 2)
        self.assertEqual(saved["cases"], self.cases)
        self.assertEqual(saved["standards"], self.standards)
        self.assertEqual(saved["comparison_contexts"], self.comparison_contexts)
        for mode in ("baseline", "fifo", "adaptive"):
            with self.subTest(mode=mode), self.runtime([
                    self.record("actual-q1", 1), self.record("actual-q2", 2)]) as runtime:
                result = runner.run_condition(self.run, mode)
                self.assertEqual([call.args[0] for call in runtime[0].query.call_args_list],
                                 [case["question"] for case in self.cases])
                self.assertEqual((result["total"], result["recorded"], result["pass"]), (2, 2, 2))
                self.assertEqual(result["pass_rate"], 1.0)

    def test_invalid_limit_or_truncated_session_fails_before_store_preparation(self):
        with patch("src.psu_memory.integration.bootstrap.initialize_empty_store") as initialize, \
                patch("src.psu_memory.integration.common.copy_database") as copy_store, \
                patch.object(runner, "_set_completed") as mark_completed:
            for index, limit in enumerate((0, -1, 3, True, "2", 1)):
                with self.subTest(limit=limit):
                    run = self.root / f"invalid-limit-{index}"
                    with self.assertRaisesRegex(ValueError, "--case-limit"):
                        runner.prepare(self.config_path, run, case_limit=limit)
                    self.assertFalse(run.exists())
            initialize.assert_not_called()
            copy_store.assert_not_called()
            mark_completed.assert_not_called()

    def add_split_fixture(self):
        formation = {"id": "formation", "session": "formation", "turn": 1,
                     "question": "用于形成经验的问题", "history": []}
        questions = read_json(self.questions_path)
        questions["cases"].insert(0, formation)
        write_json(self.questions_path, questions)
        truth = read_json(self.truth_path)
        truth["cases"].insert(0, {"id": "formation", "response_type": "answer",
                                  "expected_result": {"columns": ["value"], "rows": [[99]]}})
        write_json(self.truth_path, truth)
        report = self.root / "reference_validation.json"
        write_json(report, {"verdict": "pass", "approved": True, "data_version": "fixture-v1",
                            "pass_count": 3, "fail_count": 0, "excluded_ids": ["clarify"],
                            "ground_truth_sha256_after": hashlib.sha256(self.truth_path.read_bytes()).hexdigest()})
        split = {
            "version": "psu_formation_test_split_v1", "data_version": "fixture-v1",
            "sources": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in (("questions", self.questions_path), ("ground_truth", self.truth_path),
                                           ("reference_validation", report))},
            "formation_case_ids": ["formation"], "test_case_ids": ["q1", "q2"],
            "excluded_case_ids": ["clarify"],
        }
        path = self.root / "split.json"
        write_json(path, split)
        self.config.update({"split_file": str(path), "case_partition": "test"})
        write_json(self.config_path, self.config)
        return path, split

    def test_prepare_uses_only_approved_test_partition_and_keeps_whole_session(self):
        self.add_split_fixture()

        def initialize(path, **_):
            Path(path).write_bytes(b"mocked empty store")

        with patch("src.psu_memory.integration.bootstrap.initialize_empty_store", side_effect=initialize), \
                patch.object(runner, "_set_completed"), \
                patch.object(runner, "_make_service", side_effect=AssertionError("prepare must not query")):
            result = runner.prepare(self.config_path, self.run)
        saved = read_json(self.run / "config.json")
        self.assertEqual(result["case_count"], len(self.cases))
        self.assertEqual(saved["cases"], self.cases)
        self.assertEqual(saved["standards"], self.standards)
        self.assertNotIn("formation", saved["standards"])

    def test_split_source_change_or_split_session_is_rejected_before_preparation(self):
        path, split = self.add_split_fixture()
        for problem in ("source_hash", "session"):
            with self.subTest(problem=problem):
                changed = deepcopy(split)
                if problem == "source_hash":
                    changed["sources"]["questions"]["sha256"] = "0" * 64
                else:
                    changed["formation_case_ids"].append("q2")
                    changed["test_case_ids"].remove("q2")
                write_json(path, changed)
                run = self.root / problem
                with self.assertRaises(ValueError):
                    runner.prepare(self.config_path, run)
                self.assertFalse(run.exists())

    def test_shell_rejects_existing_run_without_launching_python(self):
        self.run.mkdir()
        previous = b'{"case_count":70,"initial_state":"empty"}\n'
        (self.run / "config.json").write_bytes(previous)
        binaries = self.root / "bin"
        binaries.mkdir()
        marker = self.root / "python_called"
        fake_python = binaries / "python"
        fake_python.write_text('#!/bin/sh\nprintf called > "$PSU_TEST_MARKER"\nexit 91\n', encoding="utf-8")
        fake_python.chmod(0o755)
        environment = {**os.environ, "PSU_DIRECT_RUN": str(self.run), "PSU_TEST_MARKER": str(marker),
                       "PATH": str(binaries) + os.pathsep + os.environ.get("PATH", "")}
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/enterprise_query_bench/run_direct_sql_validation.sh")],
            env=environment, capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())
        self.assertEqual((self.run / "config.json").read_bytes(), previous)

    @staticmethod
    def snapshot_seeds():
        from src.psu_memory.pool._common import sha256_text
        from src.psu_memory.pool.pool_models import DIFFICULTY_PROMPT_VERSION, GROUP_RULE_VERSION

        prompt_path = ROOT / "src/psu_memory/pool/prompts" / f"{DIFFICULTY_PROMPT_VERSION}.txt"
        prompt_sha256 = sha256_text(prompt_path.read_text(encoding="utf-8"))
        return [
            {"identity": {"execution_id": key, "content_version": "v1"},
             "quality": SimpleNamespace(admission_state="eligible"),
             "grouping": SimpleNamespace(rule_version=GROUP_RULE_VERSION),
             "difficulty": SimpleNamespace(assessment=SimpleNamespace(
                 prompt_version=DIFFICULTY_PROMPT_VERSION, prompt_sha256=prompt_sha256))}
            for key in ("seed-one", "seed-two")
        ]

    def test_snapshot_copies_identical_seeds_before_configuring_each_strategy(self):
        source = self.root / "seed.sqlite3"
        source.write_bytes(b"same selected seed members and index")
        write_json(self.config_path, {**self.config, "initial_state": "snapshot",
                                     "source_store": str(source)})
        seeds = self.snapshot_seeds()
        initial_pool = Mock()
        initial_pool.load_active_members.return_value = seeds
        initial_pool.load_inputs.return_value = seeds
        condition_pools = {mode: Mock() for mode in self.config["conditions"]}

        def pool_for(path, **_):
            path = Path(path)
            return initial_pool if path.name == ".initial.sqlite3" else condition_pools[path.parent.name]

        index = Mock()
        index.inspect.return_value = {"complete": True, "member_count": 2, "ready_binding_count": 4}
        with (
            patch("src.psu_memory.integration.bootstrap.initialize_empty_store") as bootstrap,
            patch("src.psu_memory.integration.common.copy_database", side_effect=shutil.copyfile) as copier,
            patch("src.psu_memory.pool.pool_store.SQLitePoolStore", side_effect=pool_for),
            patch("src.psu_memory.retrieval.vector_cache.SQLiteVectorCache"),
            patch("src.psu_memory.retrieval.index_store.SQLiteRetrievalIndex", return_value=index),
            patch.object(runner, "_set_completed"),
            patch.object(runner, "_make_service", side_effect=AssertionError("prepare must not query")),
        ):
            runner.prepare(self.config_path, self.run)

        bootstrap.assert_not_called()
        self.assertEqual(copier.call_count, 1 + len(self.config["conditions"]))
        self.assertEqual(Path(copier.call_args_list[0].args[0]), source)
        initial = self.run / ".initial.sqlite3"
        for mode, pool in condition_pools.items():
            target = self.run / mode / "history.sqlite3"
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertTrue(any(Path(call.args[0]) == initial and Path(call.args[1]) == target
                                for call in copier.call_args_list[1:]))
            pool.configure.assert_called_once()
            config = pool.configure.call_args.kwargs["config"]
            self.assertEqual(config.strategy, "fifo" if mode == "fifo" else "adaptive")
            self.assertEqual(config.total_capacity, 18 if mode == "fifo" else 32)
            pool.sync.assert_not_called()
        self.assertFalse(initial.exists())
        self.assertEqual(source.read_bytes(), b"same selected seed members and index")

    def test_snapshot_rejects_old_difficulty_version_or_changed_prompt(self):
        source = self.root / "seed.sqlite3"
        source.write_bytes(b"old seed difficulty must not be accepted")
        write_json(self.config_path, {**self.config, "initial_state": "snapshot",
                                     "source_store": str(source)})
        for field, value in (("prompt_version", "difficulty_v1"),
                             ("prompt_sha256", "old-prompt-content")):
            with self.subTest(field=field):
                seeds = self.snapshot_seeds()
                setattr(seeds[0]["difficulty"].assessment, field, value)
                pool = Mock()
                pool.load_active_members.return_value = seeds
                pool.load_inputs.return_value = seeds
                index = Mock()
                index.inspect.return_value = {"complete": True, "member_count": 2,
                                              "ready_binding_count": 4}
                run = self.root / ("rejected-" + field)
                with (
                    patch("src.psu_memory.integration.common.copy_database",
                          side_effect=shutil.copyfile) as copier,
                    patch("src.psu_memory.pool.pool_store.SQLitePoolStore", return_value=pool),
                    patch("src.psu_memory.retrieval.vector_cache.SQLiteVectorCache"),
                    patch("src.psu_memory.retrieval.index_store.SQLiteRetrievalIndex", return_value=index),
                    patch.object(runner, "_set_completed") as completed,
                    patch.object(runner, "_make_service", side_effect=AssertionError("prepare must not query")),
                ):
                    with self.assertRaisesRegex(ValueError, "种子难度标注须使用当前提示词"):
                        runner.prepare(self.config_path, run)
                self.assertEqual(copier.call_count, 1)
                pool.configure.assert_not_called()
                completed.assert_not_called()
                self.assertFalse((run / ".initial.sqlite3").exists())
                self.assertFalse((run / "config.json").exists())
                self.assertTrue(all(not (run / mode).exists() for mode in self.config["conditions"]))
        self.assertEqual(source.read_bytes(), b"old seed difficulty must not be accepted")

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
        self.assertEqual(service.session_history, self.cases[1]["history"])
        self.assertNotIn(self.public[0], service.session_history)
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

    def test_submission_context_scores_reordered_helpers_and_updates_only_correct_identity(self):
        self.standards = {
            "q1": {"columns": ["stat_month", "operating_revenue_yuan"], "rows": [["2026-08-01", 12]]},
            "q2": {"columns": ["stat_month", "operating_revenue_yuan"], "rows": [["2026-08-01", 13]]},
        }
        self.comparison_contexts = {
            "q1": {"enterprises": [{"enterprise_id": "E1", "enterprise_name": "甲公司"}], "amount_unit": "元"},
            "q2": {"enterprises": [{"enterprise_id": "E2", "enterprise_name": "乙公司"}], "amount_unit": "元"},
        }
        records = [self.record("actual-q1", 12), self.record("actual-q2", 13)]
        for record, amount in zip(records, (12, 13)):
            record["operations"][0].update({
                "columns": ["enterprise_name", "revenue", "month", "amount_unit", "enterprise_id"],
                "rows": [["甲公司", amount, "2026-08", "元", "E1"]],
            })
        self.prepared_fixture()
        with self.runtime(records) as runtime:
            result = runner.run_condition(self.run, "adaptive")
            updater, histories = runtime[2], runtime[6]
        self.assertEqual((result["pass"], result["fail"]), (1, 1))
        updater.apply.assert_called_once()
        self.assertEqual(updater.apply.call_args.kwargs["query_id"], "actual-q1")
        self.assertEqual(updater.apply.call_args.kwargs["case"], self.cases[0])
        self.assertEqual(updater.apply.call_args.kwargs["record"], records[0])
        self.assertEqual(histories, [self.public, [*self.public, *self.cases[1]["history"]]])
        for case_id, verdict in (("q1", "pass"), ("q2", "fail")):
            saved = read_json(self.run / "adaptive" / "results" / (case_id + ".json"))
            self.assertEqual(saved["verdict"], verdict)
            self.assertEqual(set(saved), {
                "case_id", "condition", "question", "query_id", "sql", "actual", "expected", "verdict"})

    def test_fixed_retrieves_but_never_updates(self):
        self.prepared_fixture()
        with self.runtime([self.record("actual-q1", 1), self.record("actual-q2", 2)]) as runtime:
            _, _, _, _, make_retriever, make_updater, _ = runtime
            result = runner.run_condition(self.run, "fixed")
        self.assertTrue(result["completed"])
        make_retriever.assert_called_once()
        make_updater.assert_not_called()

    def test_fifo_learns_only_from_passed_sql(self):
        self.prepared_fixture()
        records = [self.record("actual-q1", 999), self.record("actual-q2", 2)]
        with self.runtime(records) as runtime:
            _, retriever, updater, _, make_retriever, _, _ = runtime
            result = runner.run_condition(self.run, "fifo")
        self.assertEqual((result["pass"], result["fail"]), (1, 1))
        make_retriever.assert_called_once()
        updater.apply.assert_called_once()
        self.assertEqual(updater.apply.call_args.kwargs["query_id"], "actual-q2")
        self.assertEqual(updater.apply.call_args.kwargs["record"], records[1])
        retriever.close.assert_called_once()

    def test_previous_code_version_cannot_start_new_condition_in_old_run(self):
        self.prepared_fixture()
        config = read_json(self.run / "config.json")
        config.pop("implementation_version")
        write_json(self.run / "config.json", config)
        with self.runtime([]) as runtime:
            with self.assertRaisesRegex(ValueError, "其他源码版本"):
                runner.run_condition(self.run, "baseline")
            runtime[3].assert_not_called()
        self.assertFalse((self.run / "baseline" / "results").exists())

    def test_previous_comparison_version_rejected_before_any_store_access_or_output(self):
        self.prepared_fixture()
        original = read_json(self.run / "config.json")
        folder = self.run / "baseline" / "results"
        with patch.object(runner, "condition_completed") as completed, \
                patch.object(runner, "_set_completed") as mark_completed, \
                patch.object(runner, "_make_service") as make_service, \
                patch.object(reporting, "summarize") as summarize:
            for version in (None, "explicit_columns_aux_v1", "obsolete-comparison"):
                with self.subTest(version=version):
                    config = deepcopy(original)
                    if version is None:
                        config.pop("comparison_version")
                    else:
                        config["comparison_version"] = version
                    write_json(self.run / "config.json", config)
                    self.assertEqual(runner.load_run(self.run)[1], config)
                    with self.assertRaisesRegex(ValueError, "其他评分口径"):
                        runner.run_condition(self.run, "baseline")
                    self.assertFalse(folder.exists())
                    folder.mkdir()
                    with self.assertRaisesRegex(ValueError, "其他评分口径"):
                        runner.run_condition(self.run, "baseline")
                    folder.rmdir()
            completed.assert_not_called()
            mark_completed.assert_not_called()
            make_service.assert_not_called()
            summarize.assert_not_called()

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
