"""选择输入及解析测试；不调用真实模型。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

import test_psu_prefilter as prefilter_fixtures

from src.psu_memory.pool._common import canonical_json, sha256_text
from src.psu_memory.retrieval.embedding import embedding_signature
from src.psu_memory.retrieval.retrieval_models import (
    RetrievalConfig,
    SelectorResult,
)
from src.psu_memory.retrieval.selector_prompt import (
    parse_selection_output,
    prepare_selector_prompt,
)
from src.psu_memory.retrieval.text import retrieval_request_fingerprint


class SelectorPromptTests(unittest.TestCase):
    def setUp(self):
        self.fixture = prefilter_fixtures.PrefilterTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

        self.config = self.fixture.config
        self.request = self.fixture.request()
        self.prefilter = self.fixture.service.prefilter(self.request)
        self.assertEqual(self.prefilter.status, "ok")

    def prepare(self, config=None):
        return prepare_selector_prompt(
            self.request,
            self.prefilter,
            config or self.config,
        )

    def test_config_defaults_and_embedding_compatibility(self):
        config = RetrievalConfig.from_json(
            "src/psu_memory/retrieval/retrieval_config.example.json"
        )
        self.assertEqual(config, RetrievalConfig())
        self.assertEqual(config.selector_max_input_chars, 12000)
        self.assertEqual(config.selector_max_output_tokens, 512)
        self.assertEqual(config.selector_reason_max_chars, 160)

        changed = replace(
            config,
            selector_max_input_chars=8000,
            selector_max_output_tokens=256,
            selector_reason_max_chars=80,
        )
        self.assertEqual(
            embedding_signature(config),
            embedding_signature(changed),
        )

    def test_invalid_selector_config(self):
        for field in (
            "selector_max_input_chars",
            "selector_max_output_tokens",
            "selector_reason_max_chars",
        ):
            for value in (0, -1, True):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        replace(self.config, **{field: value})

    def test_request_fingerprint_covers_input(self):
        original = retrieval_request_fingerprint(self.request)
        self.assertEqual(
            original,
            retrieval_request_fingerprint(replace(self.request)),
        )
        for change in (
            {"question": "统计企业数量"},
            {"context_text": "不同上下文"},
            {"data_advice": "统计"},
            {"namespace": "other"},
            {"data_version": "v2"},
            {"task_id": "other-task"},
            {"limit": 0},
            {"max_context_chars": 100},
        ):
            self.assertNotEqual(
                original,
                retrieval_request_fingerprint(
                    replace(self.request, **change)
                ),
            )

    def test_payload_and_aliases_reuse_candidates(self):
        prepared = self.prepare()
        self.assertTrue(prepared.ready)
        payload = json.loads(prepared.user_text)

        self.assertEqual(
            set(payload["current"]),
            {
                "question", "context_questions", "data_advice",
                "namespace", "data_version",
            },
        )
        self.assertEqual(
            set(payload["candidates"][0]),
            {
                "id", "question", "context_questions",
                "sql", "notes", "applicability",
            },
        )
        self.assertEqual(len(prepared.alias_map), 3)

        for number, item in enumerate(self.prefilter.candidates, 1):
            alias = f"c{number}"
            snapshot = prepared.alias_map[alias]
            self.assertIs(snapshot["candidate"], item)
            self.assertEqual(
                snapshot["identity"], item["member"]["identity"]
            )
            self.assertEqual(
                payload["candidates"][number - 1]["sql"],
                item["member"]["quality"].content["sql"],
            )

    def test_original_context_restored_after_vector_fallback(self):
        history = json.dumps([
            {
                "role": "user",
                "content": prefilter_fixtures.BASE * 200,
            }
        ], ensure_ascii=False)
        request = self.fixture.request(
            "那北京呢？", context_text=history
        )
        result = self.fixture.service.prefilter(request)
        self.assertEqual(result.status, "ok")
        self.assertEqual(
            result.query["reason"],
            "context_overflow_question_only",
        )
        self.assertEqual(
            result.query["matching"]["context_questions"], ()
        )

        prepared = prepare_selector_prompt(
            request, result, self.config
        )
        self.assertTrue(prepared.ready)
        current = json.loads(prepared.user_text)["current"]
        self.assertEqual(
            current["context_questions"],
            [prefilter_fixtures.BASE * 200],
        )

    def test_unavailable_prefilter_and_empty_candidates(self):
        unavailable = replace(
            self.prefilter,
            status="index_not_ready",
            reason="test",
        )
        prepared = prepare_selector_prompt(
            self.request, unavailable, self.config
        )
        self.assertFalse(prepared.ready)
        self.assertEqual(prepared.reason_code, "prefilter_unavailable")

        empty = replace(self.prefilter, candidates=[])
        prepared = prepare_selector_prompt(
            self.request, empty, self.config
        )
        self.assertEqual(prepared.reason_code, "no_candidates")

    def test_request_mismatch(self):
        prepared = prepare_selector_prompt(
            replace(self.request, data_advice="新的建议"),
            self.prefilter,
            self.config,
        )
        self.assertFalse(prepared.ready)
        self.assertEqual(prepared.reason_code, "request_mismatch")

    def test_missing_request_fingerprint(self):
        prepared = prepare_selector_prompt(
            self.request,
            replace(self.prefilter, diagnostics={}),
            self.config,
        )
        self.assertEqual(
            prepared.reason_code, "request_fingerprint_missing"
        )

    def test_exact_budget_and_candidate_prefix(self):
        full = self.prepare()
        exact = self.prepare(replace(
            self.config,
            selector_max_input_chars=full.input_chars,
        ))
        self.assertEqual(len(exact.alias_map), 3)

        payload = json.loads(full.user_text)
        payload["candidates"] = payload["candidates"][:2]
        budget = len(full.system_prompt) + len(canonical_json(payload))

        smaller = self.prepare(replace(
            self.config,
            selector_max_input_chars=budget,
        ))
        self.assertTrue(smaller.ready)
        self.assertEqual(list(smaller.alias_map), ["c1", "c2"])
        self.assertEqual(smaller.input_chars, budget)
        self.assertEqual(len(smaller.excluded_candidates), 1)
        self.assertEqual(
            json.loads(smaller.user_text)["candidates"],
            payload["candidates"],
        )

    def test_budget_too_small_has_no_candidates(self):
        prepared = self.prepare(replace(
            self.config,
            selector_max_input_chars=1,
        ))
        self.assertFalse(prepared.ready)
        self.assertEqual(prepared.alias_map, {})
        self.assertEqual(prepared.reason_code, "input_budget_exceeded")

    def test_valid_choice_and_null(self):
        prepared = self.prepare()
        self.assertEqual(
            parse_selection_output(
                '{"selected_id":"c1","reason":"可借用企业筛选方法。"}',
                prepared,
                self.config,
            ),
            ("c1", "可借用企业筛选方法。"),
        )
        self.assertEqual(
            parse_selection_output(
                '{"selected_id":null,"reason":"当前意图不明确。"}',
                prepared,
                self.config,
            ),
            (None, "当前意图不明确。"),
        )

    def test_invalid_json_and_identifier(self):
        prepared = self.prepare()
        invalid = (
            "不是 JSON",
            '```json\n{"selected_id":"c1","reason":"说明"}\n```',
            '{"selected_id":"c1","selected_id":"c2","reason":"说明"}',
            '{"selected_id":"c9","reason":"说明"}',
            '{"selected_id":true,"reason":"说明"}',
            '{"selected_id":1,"reason":"说明"}',
            '{"selected_id":["c1"],"reason":"说明"}',
            '{"selected_id":null,"reason":NaN}',
            '{"selected_id":null,"reason":"说明","score":1}',
            '{"selected_id":"c1"}',
            '[]',
            'null',
        )
        for text in invalid:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_selection_output(
                        text, prepared, self.config
                    )

    def test_reason_constraints_and_unprepared_input(self):
        prepared = self.prepare()
        for reason in (" ", "字" * 161, 1):
            with self.subTest(reason=reason):
                with self.assertRaises(ValueError):
                    parse_selection_output(
                        json.dumps({
                            "selected_id": "c1",
                            "reason": reason,
                        }),
                        prepared,
                        self.config,
                    )

        unavailable = self.prepare(replace(
            self.config, selector_max_input_chars=1
        ))
        with self.assertRaises(ValueError):
            parse_selection_output(
                '{"selected_id":null,"reason":"说明"}',
                unavailable,
                self.config,
            )

    def test_prompt_hash_and_result_serialization(self):
        first = self.prepare()
        second = self.prepare()
        self.assertEqual(first.prompt_sha256, second.prompt_sha256)
        self.assertEqual(
            first.prompt_sha256, sha256_text(first.system_prompt)
        )
        self.assertEqual(
            first.input_chars,
            len(first.system_prompt) + len(first.user_text),
        )
        self.assertNotIn("candidate", first.to_dict()["aliases"]["c1"])

        result = SelectorResult(
            reason_code="serialization_test",
            reason="仅验证结果结构",
            selected=self.prefilter.candidates[0],
            selected_id="c1",
        )
        selected = result.to_dict()["selected"]
        self.assertEqual(
            selected,
            self.prefilter.to_dict()["candidates"][0],
        )
        self.assertEqual(result.usage.model_call_count, 0)


if __name__ == "__main__":
    unittest.main()