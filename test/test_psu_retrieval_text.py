"""第一步回归测试：不调用模型、不操作数据库、不创建临时文件。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from src.psu_memory.models import RetrievalRequest
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.text import (
    build_matching_text,
    build_request_text,
)


def context(*questions):
    messages = []
    for question in questions:
        messages.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": "历史回答与 SELECT SQL 不参与初筛"},
        ])
    return json.dumps(messages, ensure_ascii=False)


def request(question, **kwargs):
    return RetrievalRequest(
        task_id="retrieval-step1",
        question=question,
        namespace="test",
        data_version="v1",
        **kwargs,
    )


class RetrievalTextTests(unittest.TestCase):
    def test_old_request_constructor(self):
        value = RetrievalRequest("t", "统计企业数量", "n", "v", "", 1, None)
        self.assertEqual(value.data_advice, "")

    def test_config_defaults_and_file(self):
        config = RetrievalConfig.from_json(
            "src/psu_memory/retrieval/retrieval_config.example.json"
        )
        self.assertEqual(config, RetrievalConfig())
        self.assertEqual(config.embedding_dimension, 1024)
        self.assertEqual(config.prefilter_limit, 3)
        self.assertEqual(config.selector_timeout_seconds, 30)
        self.assertEqual(config.max_context_chars, 3000)

    def test_invalid_config(self):
        for values in (
            {"prefilter_limit": 0},
            {"embedding_batch_size": True},
            {"max_context_questions": -1},
            {"max_context_questions": 3},
            {"selector_timeout_seconds": float("nan")},
            {"selector_max_retries": 1},
            {"pooling_method": "mean"},
            {"normalize_embeddings": False},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    RetrievalConfig(**values)

    def test_explicit_history_is_kept_for_independent_question(self):
        first = build_matching_text("  统计未缴社保企业数量。  ")
        second = build_matching_text(
            "统计未缴社保企业数量。",
            context("查询企业营收排名", "比较资产变化"),
        )
        self.assertTrue(first.ready)
        self.assertEqual(first.text, "统计未缴社保企业数量。")
        self.assertEqual(first.context_questions, ())
        self.assertNotEqual(first.input_sha256, second.input_sha256)
        self.assertEqual(second.context_questions, ("查询企业营收排名", "比较资产变化"))
        self.assertEqual(second.rule_version, "question_context_v2")

    def test_recent_two_questions_do_not_depend_on_wording(self):
        value = build_matching_text(
            "先只看4、5月吧",
            context("更早的问题", "查询锐序公司2025年的经营情况", "再列一下营业收入"),
        )
        self.assertTrue(value.ready)
        self.assertEqual(
            value.context_questions,
            ("查询锐序公司2025年的经营情况", "再列一下营业收入"),
        )
        self.assertNotIn("更早的问题", value.text)
        self.assertNotIn("SELECT", value.text)
        self.assertIn("先只看4、5月吧", value.text)

    def test_history_limit_and_repeated_current_question(self):
        history = context("统计上海未缴社保企业数量。", "只看2025年。", "那北京呢？")
        complete = build_matching_text("那北京呢？", history)
        self.assertTrue(complete.ready)
        self.assertEqual(complete.context_questions, ("只看2025年。", "那北京呢？"))
        limited = build_matching_text("那北京呢？", history, max_context_questions=1)
        self.assertTrue(limited.ready)
        self.assertEqual(limited.context_questions, ("那北京呢？",))
        disabled = build_matching_text("那北京呢？", history, max_context_questions=0)
        self.assertEqual(disabled.context_questions, ())

    def test_empty_history_is_allowed_and_invalid_history_rejected(self):
        missing = build_matching_text("那北京呢？")
        self.assertTrue(missing.ready)
        self.assertEqual(missing.reason_code, "question_only")
        invalid = build_matching_text("那北京呢？", "上一条问题：统计企业数量")
        self.assertFalse(invalid.ready)
        self.assertEqual(invalid.reason_code, "invalid_context")
        assistant_only = json.dumps([
            {"role": "assistant", "content": "统计上海企业数量"},
            {"role": "system", "content": "公开业务约定"},
        ], ensure_ascii=False)
        value = build_matching_text("那北京呢？", assistant_only)
        self.assertTrue(value.ready)
        self.assertEqual(value.context_questions, ())

    def test_data_advice_does_not_enter_matching_text(self):
        first = request("统计未缴社保企业数量。")
        second = replace(
            first,
            data_advice="排名前十，并参考 SELECT 查询",
        )
        config = RetrievalConfig()
        a = build_request_text(first, config)
        b = build_request_text(second, config)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_request_validation(self):
        base = request("统计企业数量")
        for invalid in (
            replace(base, limit=True),
            replace(base, limit=-1),
            replace(base, max_context_chars=-1),
            replace(base, namespace=""),
            replace(base, context_text=[]),
        ):
            with self.assertRaises(ValueError):
                build_request_text(invalid, RetrievalConfig())

    def test_normalization_is_stable(self):
        a = build_matching_text("统计  企业数量")
        b = build_matching_text("统计\n企业数量")
        self.assertEqual(a.input_sha256, b.input_sha256)


if __name__ == "__main__":
    unittest.main()
