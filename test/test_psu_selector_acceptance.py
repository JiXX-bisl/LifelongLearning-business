"""择优阶段综合补充测试；不调用真实模型。"""

from __future__ import annotations

import json
import sqlite3
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from test import test_psu_model_client as client_fixtures
from test import test_psu_prefilter as prefilter_fixtures
from test import test_psu_selector as selector_fixtures

from src.psu_memory.pool._common import canonical_json
from src.psu_memory.retrieval.index_store import SQLiteRetrievalIndex
from src.psu_memory.retrieval.selector import LLMExperienceSelector
from src.psu_memory.retrieval.selector_prompt import prepare_selector_prompt


class SelectorAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = selector_fixtures.SelectorTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

        self.request = self.fixture.request
        self.prefilter = self.fixture.prefilter
        self.config = self.fixture.index.config

    def index_for(self, config):
        return SQLiteRetrievalIndex(
            self.fixture.fixture.cache,
            config,
        )

    def test_actual_adapter_timeout_reaches_coordinator(self):
        sdk = client_fixtures.ModelClientTests()
        self.addCleanup(sdk.doCleanups)
        sdk.setUp()
        sdk.delay = 1.0

        config = replace(
            self.config,
            selector_timeout_seconds=0.05,
        )
        client = sdk.selector(config)

        with LLMExperienceSelector(
            self.index_for(config),
            client,
        ) as selector:
            result = selector.select(self.request, self.prefilter)

        self.assertEqual(result.reason_code, "model_timeout")
        self.assertIsNone(result.selected)
        self.assertEqual(result.usage.model_call_count, 1)
        self.assertEqual(result.usage.http_request_count, 1)
        self.assertEqual(result.usage.retry_count, 0)
        self.assertIsNone(result.usage.prompt_tokens)
        self.assertFalse(result.usage.token_usage_complete)
        self.assertEqual(sdk.cancelled, 1)
        self.assertEqual(sdk.completed, 0)

    def test_budget_prefix_controls_actual_selectable_ids(self):
        full = prepare_selector_prompt(
            self.request,
            self.prefilter,
            self.config,
        )
        payload = json.loads(full.user_text)
        payload["candidates"] = payload["candidates"][:2]
        budget = len(full.system_prompt) + len(canonical_json(payload))
        config = replace(
            self.config,
            selector_max_input_chars=budget,
        )

        for selected_id, expected_code in (
            ("c2", "selected"),
            ("c3", "invalid_model_output"),
        ):
            with self.subTest(selected_id=selected_id):
                client = selector_fixtures.FakeClient(
                    selected_id=selected_id
                )
                with LLMExperienceSelector(
                    self.index_for(config),
                    client,
                ) as selector:
                    result = selector.select(
                        self.request,
                        self.prefilter,
                    )

                sent = json.loads(client.last_user_text)
                self.assertEqual(
                    [item["id"] for item in sent["candidates"]],
                    ["c1", "c2"],
                )
                self.assertEqual(result.reason_code, expected_code)
                self.assertEqual(client.calls, 1)

                for actual, original in zip(
                    sent["candidates"],
                    self.prefilter.candidates,
                ):
                    self.assertEqual(
                        actual["sql"],
                        original["member"]["quality"].content["sql"],
                    )

                if selected_id == "c3":
                    self.assertIsNone(result.selected)

    def test_insufficient_budget_does_not_create_client(self):
        config = replace(
            self.config,
            selector_max_input_chars=1,
        )
        factory = Mock()

        with LLMExperienceSelector(
            self.index_for(config),
            client_factory=factory,
        ) as selector:
            result = selector.select(self.request, self.prefilter)

        self.assertEqual(result.reason_code, "input_budget_exceeded")
        self.assertEqual(result.usage.model_call_count, 0)
        factory.assert_not_called()

    def test_original_context_is_sent_after_embedding_fallback(self):
        historical_question = prefilter_fixtures.BASE * 200
        request = self.fixture.fixture.request(
            "那北京呢？",
            context_text=json.dumps([
                {"role": "user", "content": historical_question}
            ], ensure_ascii=False),
        )
        prefilter = self.fixture.fixture.service.prefilter(request)
        self.assertEqual(prefilter.status, "ok")
        self.assertEqual(
            prefilter.query["reason"],
            "context_overflow_question_only",
        )

        client = selector_fixtures.FakeClient(selected_id=None)
        with LLMExperienceSelector(
            self.fixture.index,
            client,
        ) as selector:
            result = selector.select(request, prefilter)

        sent = json.loads(client.last_user_text)
        self.assertEqual(
            sent["current"]["context_questions"],
            [historical_question],
        )
        self.assertEqual(result.reason_code, "no_selection")

    def test_before_read_failure_is_zero_call(self):
        factory = Mock()

        with (
            LLMExperienceSelector(
                self.fixture.index,
                client_factory=factory,
            ) as selector,
            patch.object(
                self.fixture.index,
                "inspect",
                side_effect=sqlite3.OperationalError("private-body"),
            ),
        ):
            result = selector.select(self.request, self.prefilter)

        self.assertEqual(
            result.reason_code,
            "recheck_failed_before_call",
        )
        self.assertEqual(result.usage.model_call_count, 0)
        factory.assert_not_called()

    def test_unexpected_client_exception_has_unknown_cost(self):
        client = selector_fixtures.FakeClient()
        client.generate = Mock(
            side_effect=RuntimeError("private-body")
        )

        with LLMExperienceSelector(
            self.fixture.index,
            client,
        ) as selector:
            result = selector.select(self.request, self.prefilter)

        self.assertEqual(result.reason_code, "model_failed")
        self.assertIsNone(result.selected)
        self.assertEqual(result.usage.model_call_count, 1)
        self.assertIsNone(result.usage.http_request_count)
        self.assertIsNone(result.usage.prompt_tokens)
        self.assertFalse(result.usage.token_usage_complete)
        self.assertNotIn(
            "private-body",
            json.dumps(result.to_dict(), ensure_ascii=False),
        )


if __name__ == "__main__":
    unittest.main()
