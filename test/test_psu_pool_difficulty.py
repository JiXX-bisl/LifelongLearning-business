"""难度评估测试：使用模拟模型，不访问网络或数据库。"""

from __future__ import annotations

import json
import unittest

from dataclasses import replace

from src.psu_memory.pool.classification import classify
from src.psu_memory.pool.pool_models import GroupingInput
from src.psu_memory.pool.difficulty import (
    DifficultyAssessor,
    DifficultyInput,
    DifficultyUsage,
    MemoryDifficultyCache,
    ModelReply,
    extract_user_questions,
    parse_difficulty_output,
)
from src.psu_memory.pool.difficulty_llm import usage_from_record


def make_request() -> DifficultyInput:
    return DifficultyInput(
        execution_id="lesson-execution-001",
        content_version="lesson-content-001",
        question="统计目前仍营业的企业数量。",
        namespace="lesson",
        data_version="v1",
    )


def model_output(
    level: str = "L1",
    *,
    missing: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "level": level,
            "reason": "根据任务需要的数据处理步骤判断。",
            "missing_information": (
                [] if missing is None else missing
            ),
        },
        ensure_ascii=False,
    )


class FakeClient:
    def __init__(
        self,
        outputs,
        *,
        signature: str = "fake-model-v1",
    ):
        self.signature = signature
        self.outputs = list(outputs)
        self.calls = 0
        self.payloads = []

    def generate(self, *, system_prompt, user_text):
        self.calls += 1
        self.payloads.append(json.loads(user_text))

        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output

        return ModelReply(
            text=output,
            usage=DifficultyUsage(
                model_call_count=1,
                http_request_count=1,
                retry_count=0,
                prompt_tokens=11,
                completion_tokens=7,
                token_usage_complete=True,
            ),
        )


class PoolDifficultyTests(unittest.TestCase):
    def test_group_rule_version_field(self):
        grouping = GroupingInput(
            execution_id="e1",
            content_version="v1",
            question="统计企业数量。",
            namespace="lesson",
            data_version="v1",
        )
        payload = classify(grouping).to_dict()

        self.assertIn("rule_version", payload)
        self.assertNotIn("rule_versoin", payload)

    def test_parse_four_levels(self):
        for expected in ("L1", "L2", "L3", "L4"):
            with self.subTest(level=expected):
                level, reason, missing = parse_difficulty_output(
                    model_output(expected)
                )
                self.assertEqual(level, expected)
                self.assertTrue(reason)
                self.assertEqual(missing, ())

    def test_invalid_output(self):
        cases = [
            "这是一段普通说明",
            "```json\n" + model_output() + "\n```",
            model_output("L5"),
            model_output("undetermined"),
            model_output("L1", missing=["缺少上下文"]),
            '{"level":"L1"}',
            json.dumps(
                {
                    "level": "L1",
                    "reason": " ",
                    "missing_information": [],
                }
            ),
            json.dumps(
                {
                    "level": "L1",
                    "reason": "直接查询",
                    "missing_information": [],
                    "score": 100,
                }
            ),
        ]

        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_difficulty_output(text)

    def test_context_and_input_projection(self):
        grouping = GroupingInput(
            execution_id="e1",
            content_version="v1",
            question="同样条件，再统计另一地区。",
            namespace="lesson",
            data_version="v1",
            context_text=json.dumps(
                [
                    {
                        "role": "user",
                        "content": "统计正常营业的企业数量。",
                    },
                    {
                        "role": "agent",
                        "content": "SELECT COUNT(*) FROM enterprise;",
                    },
                ],
                ensure_ascii=False,
            ),
        )

        request = DifficultyInput.from_grouping(grouping)
        client = FakeClient([model_output("L2")])
        assessor = DifficultyAssessor(client)

        assessor.assess(request, admission_state="eligible")
        payload = client.payloads[0]

        self.assertEqual(
            set(payload),
            {
                "question",
                "context_questions",
                "schema_context",
                "data_advice",
            },
        )
        self.assertEqual(
            payload["context_questions"],
            ["统计正常营业的企业数量。"],
        )
        self.assertNotIn(
            "SELECT",
            json.dumps(payload, ensure_ascii=False),
        )

    def test_reject_unstructured_context(self):
        with self.assertRaises(ValueError):
            extract_user_questions(
                "用户问了数量，模型返回 SELECT COUNT(*)"
            )

        self.assertEqual(extract_user_questions(""), ())

    def test_only_eligible_can_be_assessed(self):
        client = FakeClient([])
        assessor = DifficultyAssessor(client)

        with self.assertRaises(ValueError):
            assessor.assess(
                make_request(),
                admission_state="suspended",
            )

        self.assertEqual(client.calls, 0)

    def test_cache_rebinds_execution_and_version(self):
        client = FakeClient([model_output()])
        assessor = DifficultyAssessor(client)

        first = assessor.assess(
            make_request(),
            admission_state="eligible",
        )

        another = replace(
            make_request(),
            execution_id="another-execution",
            content_version="another-version",
        )
        second = assessor.assess(
            another,
            admission_state="eligible",
        )

        self.assertEqual(client.calls, 1)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(
            second.execution_id,
            "another-execution",
        )
        self.assertEqual(
            second.content_version,
            "another-version",
        )
        self.assertEqual(second.usage.model_call_count, 0)
        self.assertEqual(second.usage.prompt_tokens, 0)
        self.assertEqual(
            second.assessment.origin_usage.prompt_tokens,
            11,
        )
        self.assertEqual(
            first.assessment.cache_key,
            second.assessment.cache_key,
        )

    def test_cache_key_changes_with_inputs_and_configuration(self):
        client = FakeClient([])
        assessor = DifficultyAssessor(client)
        request = make_request()
        original_key = assessor.cache_key(request)

        changes = [
            {"question": "列出正常营业企业的名称。"},
            {"namespace": "another-namespace"},
            {"data_version": "v2"},
            {"context_questions": ("只考虑上海地区。",)},
            {"schema_context": "企业表包含营业状态字段。"},
            {"data_advice": "先按企业标识去重。"},
        ]

        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(
                    original_key,
                    assessor.cache_key(replace(request, **change)),
                )

        changed_model = DifficultyAssessor(
            FakeClient([], signature="fake-model-v2")
        )
        changed_prompt_version = DifficultyAssessor(
            client,
            prompt_version="difficulty_v2",
        )
        changed_prompt_text = DifficultyAssessor(
            client,
            prompt_text=assessor.prompt_text + "\n补充规则。",
        )

        for other in (
            changed_model,
            changed_prompt_version,
            changed_prompt_text,
        ):
            self.assertNotEqual(
                original_key,
                other.cache_key(request),
            )

        self.assertEqual(client.calls, 0)

    def test_undetermined_is_not_l1_and_requires_explicit_retry(self):
        client = FakeClient(
            [
                model_output(
                    "undetermined",
                    missing=["缺少“这些企业”的具体范围。"],
                ),
                model_output("L2"),
            ]
        )
        assessor = DifficultyAssessor(client)

        first = assessor.assess(
            make_request(),
            admission_state="eligible",
        )
        self.assertEqual(first.assessment.status, "undetermined")
        self.assertIsNone(first.assessment.level)

        cached = assessor.assess(
            make_request(),
            admission_state="eligible",
        )
        self.assertTrue(cached.cache_hit)
        self.assertEqual(client.calls, 1)

        retried = assessor.assess(
            make_request(),
            admission_state="eligible",
            retry_failed=True,
        )
        self.assertEqual(retried.assessment.level, "L2")
        self.assertEqual(client.calls, 2)

    def test_invalid_output_is_cached_until_explicit_retry(self):
        client = FakeClient(["非 JSON", model_output("L3")])
        assessor = DifficultyAssessor(client)

        first = assessor.assess(
            make_request(),
            admission_state="eligible",
        )
        self.assertEqual(first.assessment.status, "failed")
        self.assertIsNone(first.assessment.level)
        self.assertEqual(first.usage.prompt_tokens, 11)

        repeated = assessor.assess(
            make_request(),
            admission_state="eligible",
        )
        self.assertTrue(repeated.cache_hit)
        self.assertEqual(client.calls, 1)

        retried = assessor.assess(
            make_request(),
            admission_state="eligible",
            retry_failed=True,
        )
        self.assertEqual(retried.assessment.level, "L3")
        self.assertEqual(client.calls, 2)

    def test_success_is_not_replaced_by_retry_flag(self):
        client = FakeClient([model_output()])
        assessor = DifficultyAssessor(client)

        assessor.assess(
            make_request(),
            admission_state="eligible",
        )
        second = assessor.assess(
            make_request(),
            admission_state="eligible",
            retry_failed=True,
        )

        self.assertTrue(second.cache_hit)
        self.assertEqual(client.calls, 1)

    def test_unexpected_client_error_has_unknown_usage(self):
        client = FakeClient([RuntimeError("模拟异常正文")])
        assessor = DifficultyAssessor(client)

        result = assessor.assess(
            make_request(),
            admission_state="eligible",
        )

        self.assertEqual(result.assessment.status, "failed")
        self.assertIsNone(result.usage.prompt_tokens)
        self.assertIsNone(result.usage.http_request_count)
        self.assertFalse(result.usage.token_usage_complete)
        self.assertNotIn(
            "模拟异常正文",
            result.assessment.reason,
        )
        self.assertNotIn("admission_state", result.to_dict())

    def test_shared_cache_can_be_injected(self):
        cache = MemoryDifficultyCache()
        first_client = FakeClient([model_output()])
        second_client = FakeClient([])

        first = DifficultyAssessor(first_client, cache)
        second = DifficultyAssessor(second_client, cache)

        first.assess(make_request(), admission_state="eligible")
        result = second.assess(
            make_request(),
            admission_state="eligible",
        )

        self.assertTrue(result.cache_hit)
        self.assertEqual(second_client.calls, 0)

    def test_usage_conversion(self):
        known = usage_from_record(
            {
                "request_count": 1,
                "retry_count": 0,
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                },
            },
            elapsed_seconds=0.5,
        )
        self.assertEqual(known.prompt_tokens, 20)
        self.assertTrue(known.token_usage_complete)

        unknown = usage_from_record(
            {
                "request_count": 1,
                "retry_count": 0,
                "usage": None,
            },
            elapsed_seconds=0.5,
        )
        self.assertIsNone(unknown.prompt_tokens)
        self.assertFalse(unknown.token_usage_complete)

        retried = usage_from_record(
            {
                "request_count": 2,
                "retry_count": 1,
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                },
            },
            elapsed_seconds=1.0,
        )
        self.assertEqual(retried.http_request_count, 2)
        self.assertEqual(retried.prompt_tokens, 20)
        self.assertFalse(retried.token_usage_complete)

        no_request = usage_from_record(
            {
                "request_count": 0,
                "retry_count": 0,
                "usage": None,
            },
            elapsed_seconds=0.1,
        )
        self.assertEqual(no_request.prompt_tokens, 0)
        self.assertTrue(no_request.token_usage_complete)


if __name__ == "__main__":
    unittest.main()