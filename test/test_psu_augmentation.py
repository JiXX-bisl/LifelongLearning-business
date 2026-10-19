"""经验文本及预算验证；不调用模型、不连接数据库。"""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.psu_memory.pool._common import sha256_text
from src.psu_memory.retrieval import augmentation
from src.psu_memory.retrieval.augmentation import (
    PROMPT_SEPARATOR,
    render_experience,
)
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.config = RetrievalConfig()
        self.content = {
            "question": "统计正常营业的企业数量。",
            "context_text": "",
            "sql": (
                "SELECT COUNT(DISTINCT enterprise_id) AS enterprise_count\n"
                "FROM enterprise_basic\n"
                "WHERE operating_status = '正常';"
            ),
            "notes": "企业数量按 enterprise_id 去重。",
            "applicability": "仅作为企业条件计数方法参考。",
            "answer": "ANSWER_ONLY_SENTINEL",
            "reason": "LLM_REASON_ONLY_SENTINEL",
            "operation": {
                "rows": [["QUERY_RESULT_ONLY_SENTINEL"]],
            },
        }

    def render(self, content=None, budget=None, config=None, **kwargs):
        return render_experience(
            self.content if content is None else content,
            config or self.config,
            max_context_chars=budget,
            **kwargs,
        )

    def minimum_size(self, content=None):
        text, report = self.render(content, budget=1)
        self.assertIsNone(text)
        self.assertEqual(
            report["reason_code"],
            "context_budget_exceeded",
        )
        return report["minimum_required_chars"]

    def test_full_fields_and_excluded_material(self):
        before = copy.deepcopy(self.content)
        text, report = self.render()

        self.assertEqual(report["reason_code"], "rendered")
        self.assertEqual(report["render_level"], "full")
        for value in (
            self.content["question"],
            self.content["sql"],
            self.content["notes"],
            self.content["applicability"],
        ):
            self.assertIn(value, text)

        for forbidden in (
            "ANSWER_ONLY_SENTINEL",
            "LLM_REASON_ONLY_SENTINEL",
            "QUERY_RESULT_ONLY_SENTINEL",
        ):
            self.assertNotIn(forbidden, text)

        self.assertNotIn("必要历史用户上下文：", text)
        self.assertEqual(self.content, before)
        self.assertEqual(report["usage"]["model_call_count"], 0)
        self.assertEqual(report["usage"]["http_request_count"], 0)
        self.assertEqual(report["usage"]["embedding_input_tokens"], 0)

    def test_independent_question_ignores_old_messages(self):
        content = {
            **self.content,
            "context_text": json.dumps([
                {"role": "user", "content": "UNRELATED_USER_SENTINEL"},
                {"role": "assistant", "content": "OLD_ANSWER_SENTINEL"},
            ]),
        }
        text, _ = self.render(content)
        self.assertNotIn("UNRELATED_USER_SENTINEL", text)
        self.assertNotIn("OLD_ANSWER_SENTINEL", text)

    def test_default_and_request_override(self):
        config = replace(self.config, max_context_chars=1)
        text, report = self.render(config=config)
        self.assertIsNone(text)
        self.assertEqual(report["effective_budget_chars"], 1)

        text, report = self.render(config=config, budget=20000)
        self.assertIsNotNone(text)
        self.assertEqual(report["effective_budget_chars"], 20000)
        self.assertEqual(config.max_context_chars, 1)

    def test_exact_budget_and_separator_accounting(self):
        text, report = self.render()
        required = report["added_chars"]

        self.assertEqual(
            required,
            len(PROMPT_SEPARATOR) + len(text),
        )
        exact, exact_report = self.render(budget=required)
        self.assertEqual(exact, text)
        self.assertEqual(exact_report["added_chars"], required)
        self.assertEqual(exact_report["render_level"], "full")
        self.assertEqual(
            exact_report["content_sha256"],
            sha256_text(text),
        )

    def test_independent_question_is_removed_before_sql(self):
        _, full = self.render()
        text, report = self.render(
            budget=full["added_chars"] - 1
        )

        self.assertEqual(report["render_level"], "sql_only")
        self.assertEqual(report["omitted_fields"], ["question"])
        self.assertNotIn("历史问题：", text)
        self.assertIn(self.content["sql"], text)
        self.assertIn(self.content["notes"], text)

    def test_dependent_question_and_context_are_removed_together(self):
        content = {
            **self.content,
            "question": "那北京呢？",
            "context_text": json.dumps([
                {
                    "role": "user",
                    "content": "统计正常营业的企业数量。",
                }
            ], ensure_ascii=False),
        }
        full_text, _ = self.render(content)
        self.assertIn("那北京呢？", full_text)
        self.assertIn("必要历史用户上下文：", full_text)

        minimum = self.minimum_size(content)
        text, report = self.render(content, budget=minimum)

        self.assertEqual(report["render_level"], "sql_only")
        self.assertEqual(
            set(report["omitted_fields"]),
            {"question", "context_questions"},
        )
        self.assertNotIn("那北京呢？", text)
        self.assertNotIn("必要历史用户上下文：", text)
        self.assertIn(content["sql"], text)

    def test_minimum_boundary_and_constraints(self):
        minimum = self.minimum_size()
        text, report = self.render(budget=minimum)

        self.assertEqual(report["render_level"], "sql_only")
        self.assertEqual(report["added_chars"], minimum)
        self.assertIn(self.content["sql"], text)
        self.assertIn(self.content["notes"], text)
        self.assertIn(self.content["applicability"], text)

        text, report = self.render(budget=minimum - 1)
        self.assertIsNone(text)
        self.assertEqual(
            report["reason_code"],
            "context_budget_exceeded",
        )
        self.assertEqual(report["added_chars"], 0)
        self.assertEqual(report["minimum_required_chars"], minimum)

    def test_sql_itself_over_budget_is_not_truncated(self):
        content = {
            **self.content,
            "sql": (
                "SELECT '"
                + "完整长文本常量" * 1000
                + "' AS example_text;"
            ),
        }
        before = copy.deepcopy(content)
        text, report = self.render(content)

        self.assertIsNone(text)
        self.assertEqual(
            report["reason_code"],
            "context_budget_exceeded",
        )
        self.assertGreater(
            report["minimum_required_chars"],
            self.config.max_context_chars,
        )
        self.assertEqual(content, before)

    def test_sql_text_is_preserved(self):
        sql = (
            " \nSELECT '$sections' AS a, 'a  b' AS b,\n"
            "       '100%' AS c /* 保留注释 */;\n  "
        )
        content = {**self.content, "sql": sql}
        text, report = self.render(content)

        self.assertIn("历史 SQL：\n" + sql, text)
        self.assertEqual(report["sql_sha256"], sha256_text(sql))

    def test_missing_or_broken_required_context(self):
        cases = (
            ("", "missing_context"),
            ("not-json", "invalid_context"),
            (
                json.dumps([
                    {"role": "user", "content": "那上海呢？"}
                ], ensure_ascii=False),
                "incomplete_context_chain",
            ),
        )
        for context, expected in cases:
            with self.subTest(expected=expected):
                content = {
                    **self.content,
                    "question": "那北京呢？",
                    "context_text": context,
                }
                text, report = self.render(content)
                self.assertIsNone(text)
                self.assertEqual(report["reason_code"], expected)

    def test_invalid_content_returns_diagnostics(self):
        cases = (
            {**self.content, "sql": ""},
            {**self.content, "question": ""},
            {**self.content, "notes": ["不是字符串"]},
        )
        for content in cases:
            with self.subTest(content=content):
                text, report = self.render(content)
                self.assertIsNone(text)
                self.assertEqual(
                    report["reason_code"],
                    "invalid_content",
                )
                self.assertEqual(report["usage"]["model_call_count"], 0)

    def test_disabled_skips_input_and_template(self):
        with patch.object(
            augmentation.Path,
            "read_text",
            side_effect=AssertionError("停用时不应读取模板"),
        ):
            text, report = render_experience(
                None,
                self.config,
                max_context_chars=0,
            )

        self.assertIsNone(text)
        self.assertEqual(report["reason_code"], "disabled")
        self.assertEqual(report["added_chars"], 0)
        self.assertEqual(report["usage"]["model_call_count"], 0)

    def test_invalid_template_rejected(self):
        for template in (
            "没有内容槽",
            "$unknown",
            "$sections\n$sections",
            "$",
        ):
            with self.subTest(template=template):
                text, report = self.render(template_text=template)
                self.assertIsNone(text)
                self.assertEqual(
                    report["reason_code"],
                    "template_invalid",
                )

    def test_invalid_budget_rejected(self):
        for budget in (-1, True, False, 1.5, "3000"):
            with self.subTest(budget=budget):
                with self.assertRaises(ValueError):
                    self.render(budget=budget)


if __name__ == "__main__":
    unittest.main()