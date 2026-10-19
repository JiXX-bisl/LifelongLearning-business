"""第一步回归测试：不调用模型、不操作数据库、不创建临时文件。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from src.psu_memory.models import RetrievalRequest
from src.psu_memory.pool.classification import classify
from src.psu_memory.pool.pool_models import GroupingInput
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
            {"selector_timeout_seconds": float("nan")},
            {"selector_max_retries": 1},
            {"pooling_method": "mean"},
            {"normalize_embeddings": False},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    RetrievalConfig(**values)

    def test_independent_question_ignores_context(self):
        first = build_matching_text("  统计未缴社保企业数量。  ")
        second = build_matching_text(
            "统计未缴社保企业数量。",
            context("查询企业营收排名", "比较资产变化"),
        )
        self.assertTrue(first.ready)
        self.assertEqual(first.text, "统计未缴社保企业数量。")
        self.assertEqual(first.context_questions, ())
        self.assertEqual(first.input_sha256, second.input_sha256)

        # 独立问题不读取无关上下文。
        third = build_matching_text(first.question, "不是 JSON")
        self.assertEqual(third.input_sha256, first.input_sha256)

    def test_followup_uses_nearest_anchor(self):
        value = build_matching_text(
            "那北京呢？",
            context("查询无关的营收排名", "统计上海未缴社保企业数量。"),
        )
        self.assertTrue(value.ready)
        self.assertEqual(
            value.context_questions,
            ("统计上海未缴社保企业数量。",),
        )
        self.assertNotIn("营收排名", value.text)
        self.assertNotIn("SELECT", value.text)
        self.assertIn("那北京呢？", value.text)

    def test_followup_chain_and_limit(self):
        history = context("统计上海未缴社保企业数量。", "只看2025年。")
        complete = build_matching_text("那北京呢？", history)
        self.assertTrue(complete.ready)
        self.assertEqual(len(complete.context_questions), 2)

        incomplete = build_matching_text(
            "那北京呢？",
            history,
            max_context_questions=1,
        )
        self.assertFalse(incomplete.ready)
        self.assertEqual(
            incomplete.reason_code,
            "incomplete_context_chain",
        )

    def test_missing_and_invalid_context(self):
        missing = build_matching_text("那北京呢？")
        self.assertFalse(missing.ready)
        self.assertEqual(missing.reason_code, "missing_context")

        invalid = build_matching_text("那北京呢？", "上一条问题：统计企业数量")
        self.assertFalse(invalid.ready)
        self.assertEqual(invalid.reason_code, "invalid_context")

        assistant_only = json.dumps([
            {"role": "assistant", "content": "统计上海企业数量"}
        ], ensure_ascii=False)
        self.assertFalse(
            build_matching_text("那北京呢？", assistant_only).ready
        )

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

    def test_colloquial_regressions(self):
        cases = (
            ("帮我看看锐序精工制造有限公司现在登记的基本情况，"
             "企业ID、那个虚拟统一社会信用代码，都给我一下。", "detail"),
            ("我想看建筑行业每家公司的现任法定代表人，"
             "重复的不要，最后按公司名称排列。", "detail"),
            ("我这边只有一个信用代码，帮我反查下是哪家公司，"
             "顺便看看它是什么企业类型。", "detail"),
            ("把信息技术行业的存续公司筛出来，我想看企业ID和注册资本，"
             "注册资本大的排前面。", "detail"),
            ("把企业财务情况列一下，包括总资产和总负债。", "detail"),
            ("看看企业近一年的营业收入走势，挑出比上个月收入低的月份。", "compare"),
            ("看看有没有企业经营范围同时提到区块链和数字货币，"
             "如果有，把企业信息一起给我。", "detail"),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                decision = classify(GroupingInput(
                    execution_id="test",
                    content_version="v1",
                    question=question,
                    namespace="test",
                    data_version="v1",
                ))
                self.assertEqual(decision.group_id, expected)
                self.assertEqual(decision.rule_version, "action_group_v2")
                self.assertTrue(build_matching_text(question).ready)

    def test_negated_colloquial_action(self):
        value = classify(GroupingInput(
            execution_id="test",
            content_version="v1",
            question="不要看走势，只列一下企业信息。",
            namespace="test",
            data_version="v1",
        ))
        self.assertEqual(value.group_id, "detail")

    def test_normalization_is_stable(self):
        a = build_matching_text("统计  企业数量")
        b = build_matching_text("统计\n企业数量")
        self.assertEqual(a.input_sha256, b.input_sha256)


if __name__ == "__main__":
    unittest.main()