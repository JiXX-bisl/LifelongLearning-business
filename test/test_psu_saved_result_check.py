"""用户独立运行：固定结果诊断入口；仅临时 JSON，不访问数据库或模型。"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.enterprise_query_bench.check_saved_sql_results import check_saved, main


class SavedResultCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.expected = {"columns": ["stat_month", "net_profit_yuan"],
                         "rows": [["2026-04-01", "-10.00"]]}
        self.context = {"enterprises": [{"enterprise_id": "E1", "enterprise_name": "甲公司"}],
                        "amount_unit": "元"}
        self.config = {
            "policy": "sql_result_direct_v1", "comparison_version": "explicit_columns_aux_v1",
            "conditions": ["baseline", "fifo", "adaptive"],
            "cases": [{"id": "q1", "question": "甲公司四月净利润"}],
            "standards": {"q1": self.expected}, "comparison_contexts": {"q1": self.context},
        }
        self.record = {
            "case_id": "q1", "condition": "baseline", "question": "甲公司四月净利润",
            "query_id": "query1", "sql": "SELECT saved_result",
            "actual": {"columns": ["企业名称", "月份", "净利润_元"],
                       "rows": [["甲公司", "2026-04", "-10.00"]]},
            "expected": self.expected, "verdict": "fail",
        }
        self.write_config()
        for mode in self.config["conditions"]:
            self.write_record(mode, {**self.record, "condition": mode})

    def write_config(self):
        (self.run / "config.json").write_text(
            json.dumps(self.config, ensure_ascii=False), encoding="utf-8")

    def write_record(self, mode, record):
        folder = self.run / mode / "results"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "q1.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    def test_saved_outputs_diagnostic_changes_without_file_writes(self):
        before = {path: path.read_bytes() for path in self.run.rglob("*") if path.is_file()}
        report = check_saved(self.run)
        self.assertTrue(report["diagnostic_only"])
        self.assertIn("不是正式重评", report["limitation"])
        self.assertEqual(report["comparison_version"]["original"], "explicit_columns_aux_v1")
        for mode in self.config["conditions"]:
            result = report["conditions"][mode]
            self.assertEqual(result["original"], {"pass": 0, "fail": 1})
            self.assertEqual(result["fixed_output_diagnostic"], {"pass": 1, "fail": 0})
            self.assertEqual(result["changed_cases"],
                             [{"case_id": "q1", "original": "fail", "diagnostic": "pass"}])
        self.assertEqual(before, {path: path.read_bytes()
                                 for path in self.run.rglob("*") if path.is_file()})

    def test_missing_evidence_cannot_be_promoted_even_if_comparator_returns_true(self):
        missing = [("query_id", None), ("query_id", "null"), ("query_id", "bad id"),
                   ("sql", None), ("sql", " "), ("actual", None)]
        for field, value in missing:
            with self.subTest(field=field, value=value):
                self.write_record("baseline", {**self.record, field: value})
                with patch("src.psu_memory.validation.comparison.compare_submission_results",
                           return_value=True) as compare:
                    result = check_saved(self.run)
                baseline = result["conditions"]["baseline"]
                self.assertEqual(baseline["fixed_output_diagnostic"], {"pass": 0, "fail": 1})
                self.assertEqual(baseline["changed_cases"], [])
                self.assertEqual(compare.call_count, 2)
                compare.assert_called_with(self.record["actual"], self.expected,
                                           context=self.context)

    def test_record_identity_and_reference_must_match_frozen_config(self):
        changes = [("case_id", "q2"), ("condition", "fifo"), ("question", "另一题"),
                   ("expected", {"columns": ["wrong"], "rows": [[0]]}),
                   ("verdict", "unverified")]
        for field, value in changes:
            with self.subTest(field=field):
                self.write_record("baseline", {**self.record, field: value})
                with patch("src.psu_memory.validation.comparison.compare_submission_results") as compare:
                    with self.assertRaises(ValueError):
                        check_saved(self.run)
                    compare.assert_not_called()

    def test_missing_extra_and_duplicate_record_files_are_rejected(self):
        folder = self.run / "baseline/results"
        original = (folder / "q1.json").read_bytes()
        (folder / "q1.json").unlink()
        with self.assertRaises(ValueError):
            check_saved(self.run)
        (folder / "q1.json").write_bytes(original)
        (folder / "copy.json").write_bytes(original)
        with self.assertRaises(ValueError):
            check_saved(self.run)

    def test_bad_config_or_missing_context_is_rejected_before_comparison(self):
        original = deepcopy(self.config)
        variations = [
            {"policy": "another_policy"},
            {"comparison_version": None},
            {"conditions": ["baseline", "baseline"]},
            {"conditions": ["../baseline"]},
            {"cases": original["cases"] * 2},
            {"standards": {}},
            {"comparison_contexts": {}},
            {"comparison_contexts": {"q1": None}},
        ]
        for change in variations:
            with self.subTest(change=change):
                self.config = {**original, **change}
                self.write_config()
                with patch("src.psu_memory.validation.comparison.compare_submission_results") as compare:
                    with self.assertRaises(ValueError):
                        check_saved(self.run)
                    compare.assert_not_called()

    def test_json_duplicate_keys_are_rejected(self):
        path = self.run / "baseline/results/q1.json"
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw[:-1] + ', "verdict": "pass"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "字段重复"):
            check_saved(self.run)

    def test_cli_requires_directory_and_only_prints_diagnostic_json(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main([])
        self.assertEqual(raised.exception.code, 2)
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--run-dir", str(self.run)])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())["diagnostic_only"])
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--run-dir", str(self.run / "missing")]), 2)


if __name__ == "__main__":
    unittest.main()
