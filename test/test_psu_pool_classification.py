"""粗分类和配置验证；不使用数据库、模型或临时文件。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from src.psu_memory.pool.classification import (
    apply_group_override,
    classify,
)
from src.psu_memory.pool.pool_models import (
    GroupingInput,
    PoolConfig,
)


def make_request(
    question: str,
    *,
    context_text: str = "",
    data_advice: str = "",
) -> GroupingInput:
    return GroupingInput(
        execution_id="lesson-execution-001",
        content_version="lesson-content-001",
        question=question,
        namespace="lesson",
        data_version="v1",
        context_text=context_text,
        data_advice=data_advice,
    )


class PoolClassificationTests(unittest.TestCase):
    def test_four_groups(self):
        cases = (
            ("列出没有缴纳社保的企业名单。", "detail"),
            ("统计没有缴纳社保的企业数量。", "aggregate"),
            ("比较企业今年和去年的营业收入。", "compare"),
            ("查询营业收入最高的三家企业。", "rank"),
        )

        for question, expected in cases:
            with self.subTest(question=question):
                result = classify(make_request(question))
                self.assertEqual(result.group_id, expected)
                self.assertEqual(result.method, "rule")
                self.assertTrue(result.matches)

    def test_compound_priority(self):
        ranked = classify(
            make_request(
                "先汇总各企业营收，再返回排名前五的企业。"
            )
        )
        compared = classify(
            make_request("统计近三个月的营业收入变化。")
        )

        self.assertEqual(ranked.group_id, "rank")
        self.assertEqual(compared.group_id, "compare")

    def test_display_order_is_not_rank(self):
        result = classify(
            make_request(
                "列出企业名单，并按企业ID升序排列。"
            )
        )
        self.assertEqual(result.group_id, "detail")

    def test_negation(self):
        cases = (
            (
                "不需要统计数量，只列出企业名单。",
                "detail",
            ),
            (
                "不要排名，统计各地区企业数量。",
                "aggregate",
            ),
            (
                "不要比较，直接统计企业数量。",
                "aggregate",
            ),
            (
                "查询不缴纳社保的企业数量。",
                "aggregate",
            ),
        )

        for question, expected in cases:
            with self.subTest(question=question):
                result = classify(make_request(question))
                self.assertEqual(result.group_id, expected)

    def test_data_advice_and_examples(self):
        advised = classify(
            make_request(
                "查看这些企业的情况。",
                data_advice=(
                    "先筛选符合条件的企业，"
                    "按企业标识去重，再统计企业数量。"
                ),
            )
        )
        self.assertEqual(advised.group_id, "aggregate")
        self.assertEqual(advised.text_source, "data_advice")

        examples = classify(
            make_request(
                "查看企业名单。",
                data_advice=(
                    "示例：统计企业数量并取排名前十。\n"
                    "```sql\n"
                    "SELECT '排名' AS example;\n"
                    "```\n"
                    "当前建议：列出企业名单，按企业ID升序。"
                ),
            )
        )
        self.assertEqual(examples.group_id, "detail")

    def test_context_boundaries(self):
        context = json.dumps(
            [
                {
                    "role": "user",
                    "content": "比较企业今年和去年的营业收入。",
                },
                {
                    "role": "assistant",
                    "content": "示例中包含排名和统计。",
                },
            ],
            ensure_ascii=False,
        )

        explicit = classify(
            make_request(
                "列出企业名单。",
                context_text=context,
            )
        )
        self.assertEqual(explicit.group_id, "detail")
        self.assertEqual(explicit.text_source, "question")

        unclear = classify(
            make_request(
                "这些企业呢？",
                context_text=context,
            )
        )
        self.assertEqual(unclear.group_id, "compare")
        self.assertEqual(unclear.text_source, "context")

        follow_up = classify(
            make_request(
                "同样的条件，换一家企业。",
                context_text=context,
            )
        )
        self.assertEqual(follow_up.group_id, "compare")
        self.assertEqual(follow_up.text_source, "context")

        plain_context = classify(
            make_request(
                "同样的条件，换一家企业。",
                context_text="上一条问题：比较企业收入。",
            )
        )
        self.assertIsNone(plain_context.group_id)

    def test_unknown_and_no_admission_fields(self):
        result = classify(
            make_request("请帮我处理这件事。")
        )

        self.assertIsNone(result.group_id)
        self.assertEqual(result.text_source, "none")
        self.assertIn("未识别", result.reason)
        self.assertNotIn(
            "admission_state",
            result.to_dict(),
        )

    def test_manual_override_history(self):
        original = classify(
            make_request("请帮我处理这件事。")
        )

        updated = apply_group_override(
            original,
            "aggregate",
            reason="人工确认当前需求是统计企业数量",
            actor="local_user",
        )

        self.assertIsNone(original.group_id)
        self.assertEqual(updated.group_id, "aggregate")
        self.assertEqual(updated.method, "manual")
        self.assertEqual(
            updated.content_version,
            original.content_version,
        )
        self.assertEqual(len(updated.manual_history), 1)

        replayed = apply_group_override(
            updated,
            "aggregate",
            reason="人工确认当前需求是统计企业数量",
            actor="local_user",
        )
        self.assertEqual(replayed, updated)

        corrected = apply_group_override(
            updated,
            "detail",
            reason="再次确认，用户需要企业名单",
        )
        self.assertEqual(len(corrected.manual_history), 2)
        self.assertEqual(
            corrected.manual_history[-1].previous_group_id,
            "aggregate",
        )

    def test_manual_validation(self):
        result = classify(make_request("列出企业名单。"))

        with self.assertRaises(ValueError):
            apply_group_override(
                result,
                "other",
                reason="不支持的组",
            )

        with self.assertRaises(ValueError):
            apply_group_override(
                result,
                "aggregate",
                reason=" ",
            )

        with self.assertRaises(ValueError):
            apply_group_override(
                result,
                "aggregate",
                reason="调整",
                actor=" ",
            )

    def test_capacity_config(self):
        default = PoolConfig()
        self.assertEqual(default.capacity_for("detail"), 8)

        values = {
            "detail": 0,
            "aggregate": 6,
            "compare": 5,
            "rank": 2,
        }
        custom = PoolConfig(capacities=values)

        self.assertEqual(custom.capacity_for("detail"), 0)
        self.assertEqual(
            custom.capacity_for("aggregate"),
            6,
        )

        values["aggregate"] = 99
        self.assertEqual(
            custom.capacity_for("aggregate"),
            6,
        )

    def test_invalid_capacity_config(self):
        valid = {
            "detail": 8,
            "aggregate": 8,
            "compare": 8,
            "rank": 8,
        }

        bad_configs = [
            {"detail": 8},
            {**valid, "other": 8},
            {**valid, "rank": -1},
            {**valid, "rank": True},
            {**valid, "rank": 1.5},
        ]

        for capacities in bad_configs:
            with self.subTest(capacities=capacities):
                with self.assertRaises(ValueError):
                    PoolConfig(capacities=capacities)

    def test_repeatability(self):
        request = make_request("统计企业数量。")
        first = classify(request)
        second = classify(request)

        self.assertEqual(first.to_dict(), second.to_dict())

        another_execution = classify(
            replace(
                request,
                execution_id="lesson-execution-002",
                content_version="lesson-content-002",
            )
        )

        self.assertEqual(
            first.input_sha256,
            another_execution.input_sha256,
        )
        self.assertNotEqual(
            first.execution_id,
            another_execution.execution_id,
        )

    def test_input_validation(self):
        request = make_request("统计企业数量。")

        with self.assertRaises(ValueError):
            replace(request, question=" ")

        with self.assertRaises(ValueError):
            replace(request, execution_id="")

        with self.assertRaises(ValueError):
            replace(request, context_text=[])


if __name__ == "__main__":
    unittest.main()