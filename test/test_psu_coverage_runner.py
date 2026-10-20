"""用户独立运行：v3版本调度、私有证据隔离及经验准入。

复用直接验证的文件夹fixture；模型、检索、经验更新和数据库接口均模拟。
本文件仅编写测试，不在部署阶段执行。
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from src.psu_memory.validation import coverage_comparison, runner
from src.psu_memory.validation.common import read_json, write_json
from src.psu_memory.validation.comparison_versions import COVERAGE_COMPARISON_VERSION
from test import test_psu_direct_runner as runner_fixtures


class CoverageRunnerTests(unittest.TestCase):
    def setUp(self):
        # Composition avoids inheriting and executing every legacy test again.
        self.fixture = runner_fixtures.DirectRunnerTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.config["comparison_version"] = COVERAGE_COMPARISON_VERSION
        self.fixture.standards = {
            "q1": {"columns": ["stat_month", "operating_revenue_yuan"],
                   "rows": [["2026-06-01", "12.00"]]},
            "q2": {"columns": ["stat_month", "operating_revenue_yuan"],
                   "rows": [["2026-07-01", "15.00"]]},
        }
        self.private_marker = "PRIVATE-EVIDENCE-NOT-FOR-PROMPT"
        self.contexts = {}
        for case_id, supplement in (("q1", "9.00"), ("q2", "11.00")):
            expected = self.fixture.standards[case_id]
            fact = dict(zip(expected["columns"], expected["rows"][0]))
            fact.update(main_business_revenue=supplement,
                        private_audit_marker=self.private_marker)
            self.contexts[case_id] = {
                "enterprises": [], "amount_unit": "元",
                "coverage_evidence": {
                    "version": COVERAGE_COMPARISON_VERSION, "rows": [fact],
                },
            }
        self.fixture.comparison_contexts = deepcopy(self.contexts)
        truth = read_json(self.fixture.truth_path)
        for fact in truth["cases"]:
            if fact["id"] in self.fixture.standards:
                fact["expected_result"] = self.fixture.standards[fact["id"]]
        write_json(self.fixture.truth_path, truth)
        write_json(self.fixture.config_path, self.fixture.config)

    def prepared(self, version=COVERAGE_COMPARISON_VERSION):
        self.fixture.prepared_fixture()
        config = read_json(self.fixture.run / "config.json")
        config["comparison_version"] = version
        write_json(self.fixture.run / "config.json", config)
        return config

    def record(self, case_id, supplement):
        expected = self.fixture.standards[case_id]
        query_id = "actual-" + case_id
        return {
            "status": "completed", "result_query_id": query_id,
            "operations": [{
                "kind": "sql", "query_id": query_id,
                "sql": ("SELECT stat_month, operating_revenue AS operating_revenue_yuan, "
                        "main_business_revenue FROM enterprise_operation_monthly"),
                "success": True, "is_complete": True,
                "columns": [*expected["columns"], "main_business_revenue"],
                "rows": [[*expected["rows"][0], supplement]],
            }],
        }

    def test_prepare_freezes_v3_evidence_without_public_prompt_leakage(self):
        def initialize(path, **_):
            Path(path).write_bytes(b"mocked empty store")

        with patch(
            "src.psu_memory.validation.coverage_evidence.build_coverage_contexts",
            return_value=deepcopy(self.contexts),
        ) as build, patch(
            "src.psu_memory.integration.bootstrap.initialize_empty_store",
            side_effect=initialize,
        ), patch.object(runner, "_set_completed"), patch.object(
            runner, "_make_service", side_effect=AssertionError("prepare must not query"),
        ):
            runner.prepare(self.fixture.config_path, self.fixture.run)

        build.assert_called_once()
        self.assertEqual(build.call_args.args[:2],
                         (self.fixture.cases, self.fixture.standards))
        saved = read_json(self.fixture.run / "config.json")
        self.assertEqual(saved["comparison_version"], COVERAGE_COMPARISON_VERSION)
        self.assertEqual(saved["comparison_contexts"], self.contexts)
        self.assertEqual(saved["standards"], self.fixture.standards)
        self.assertEqual(saved["public_context"], self.fixture.public)
        self.assertEqual(saved["cases"], self.fixture.cases)
        business = yaml.safe_load((self.fixture.run / "business.yaml").read_text())
        self.assertNotIn("comparison_contexts", business)
        self.assertNotIn("coverage_evidence", business)
        self.assertFalse((self.fixture.run / "inputs" / "tables.json").exists())
        public_inputs = json.dumps(
            [business, saved["public_context"], saved["cases"]], ensure_ascii=False)
        self.assertNotIn(self.private_marker, public_inputs)

        write_json(self.fixture.truth_path, {"cases": []})
        self.contexts["q1"]["coverage_evidence"]["rows"][0]["main_business_revenue"] = "999"
        self.assertEqual(runner.load_run(self.fixture.run)[1]["comparison_contexts"],
                         saved["comparison_contexts"])

    def test_v3_dispatch_admits_only_verified_extras_and_keeps_evidence_private(self):
        self.prepared()
        records = [self.record("q1", "9.00"), self.record("q2", "999.00")]
        with self.fixture.runtime(records) as runtime, patch.object(
            runner, "evaluate_result", side_effect=AssertionError("v3 must not use v2"),
        ), patch.object(
            coverage_comparison, "evaluate_coverage_result",
            wraps=coverage_comparison.evaluate_coverage_result,
        ) as evaluate:
            summary = runner.run_condition(self.fixture.run, "adaptive")

        self.assertEqual((summary["pass"], summary["fail"]), (1, 1))
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(evaluate.call_args_list[0].kwargs["context"], self.contexts["q1"])
        runtime[2].apply.assert_called_once()
        self.assertEqual(runtime[2].apply.call_args.kwargs["case"]["id"], "q1")
        self.assertEqual(runtime[6], [
            self.fixture.public,
            [*self.fixture.public, *self.fixture.cases[1]["history"]],
        ])
        self.assertNotIn(self.private_marker, json.dumps(runtime[6], ensure_ascii=False))
        self.assertNotIn("coverage_evidence", json.dumps(runtime[6], ensure_ascii=False))
        first = read_json(self.fixture.run / "adaptive" / "results" / "q1.json")
        second = read_json(self.fixture.run / "adaptive" / "results" / "q2.json")
        self.assertEqual(first["comparison_version"], COVERAGE_COMPARISON_VERSION)
        self.assertEqual(first["comparison_reason"], "complete_and_verified")
        self.assertEqual(second["comparison_reason"], "supplementary_values_differ")
        self.assertNotIn("coverage_evidence", first)

    def test_frozen_v2_still_rejects_the_same_extra_business_column(self):
        self.prepared(version=runner.COMPARISON_VERSION)
        records = [self.record("q1", "9.00"), self.record("q2", "11.00")]
        with self.fixture.runtime(records) as runtime, patch.object(
            coverage_comparison, "evaluate_coverage_result",
            side_effect=AssertionError("v2 must not use v3"),
        ):
            summary = runner.run_condition(self.fixture.run, "fifo")

        self.assertEqual((summary["pass"], summary["fail"]), (0, 2))
        runtime[5].assert_not_called()
        runtime[2].apply.assert_not_called()
        row = read_json(self.fixture.run / "fifo" / "results" / "q1.json")
        self.assertNotIn("comparison_reason", row)
        self.assertNotIn("comparison_version", row)

    def test_invalid_v3_evidence_fails_before_any_service_or_store_operation(self):
        original = self.prepared()
        for problem in ("missing", "wrong_version", "missing_rows", "wrong_row_count",
                        "empty_row", "core_conflict"):
            with self.subTest(problem=problem):
                config = deepcopy(original)
                context = config["comparison_contexts"]["q1"]
                if problem == "missing":
                    context.pop("coverage_evidence")
                elif problem == "wrong_version":
                    context["coverage_evidence"]["version"] = runner.COMPARISON_VERSION
                elif problem == "missing_rows":
                    context["coverage_evidence"].pop("rows")
                elif problem == "empty_row":
                    context["coverage_evidence"]["rows"] = [{}]
                elif problem == "core_conflict":
                    context["coverage_evidence"]["rows"][0]["operating_revenue_yuan"] = "999"
                else:
                    context["coverage_evidence"]["rows"] = []
                write_json(self.fixture.run / "config.json", config)
                with patch.object(runner, "_make_service") as service, patch.object(
                    runner, "_make_retriever",
                ) as retriever, patch.object(runner, "_make_updater") as updater, patch.object(
                    runner, "_set_completed",
                ) as completion:
                    with self.assertRaisesRegex(ValueError, "核验依据"):
                        runner.run_condition(self.fixture.run, "fifo")
                service.assert_not_called()
                retriever.assert_not_called()
                updater.assert_not_called()
                completion.assert_not_called()
                self.assertFalse((self.fixture.run / "fifo" / "results").exists())

    def test_prepare_evidence_failure_happens_before_creating_any_store(self):
        with patch(
            "src.psu_memory.validation.coverage_evidence.build_coverage_contexts",
            side_effect=ValueError("补充核验依据缺失"),
        ), patch(
            "src.psu_memory.integration.bootstrap.initialize_empty_store",
        ) as initialize, patch(
            "src.psu_memory.integration.common.copy_database",
        ) as copy_store, patch.object(runner, "_set_completed") as completion:
            with self.assertRaisesRegex(ValueError, "核验依据"):
                runner.prepare(self.fixture.config_path, self.fixture.run)
        initialize.assert_not_called()
        copy_store.assert_not_called()
        completion.assert_not_called()
        self.assertFalse(self.fixture.run.exists())
