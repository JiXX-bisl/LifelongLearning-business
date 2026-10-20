"""选择协调器与本地命令测试，不调用真实模型。"""

from __future__ import annotations

import io
import json
import sqlite3
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from unittest.mock import Mock, patch

from test import test_psu_active_members as active_fixtures
from test import test_psu_prefilter as prefilter_fixtures

from src.psu_memory.cli import select_experience
from src.psu_memory.evaluation.quality_models import Feedback
from src.psu_memory.pool.pool_models import DifficultyUsage, ModelReply
from src.psu_memory.retrieval.selector import LLMExperienceSelector


class FakeClient:
    signature = "mock_selector_v1"

    def __init__(
        self,
        *,
        selected_id="c1",
        text=None,
        callback=None,
        error_type=None,
        usage=None,
    ):
        self.text = (
            json.dumps({
                "selected_id": selected_id,
                "reason": "可借用候选查询方法。",
            }, ensure_ascii=False)
            if text is None else text
        )
        self.callback = callback
        self.error_type = error_type
        self.usage = usage or DifficultyUsage(
            elapsed_seconds=0.01,
            model_call_count=1,
            http_request_count=1,
            retry_count=0,
            prompt_tokens=20,
            completion_tokens=8,
            token_usage_complete=True,
        )
        self.calls = 0
        self.closed = 0
        self.last_user_text = ""

    def generate(self, *, system_prompt, user_text):
        self.calls += 1
        self.last_user_text = user_text
        if self.callback is not None:
            self.callback()

        return ModelReply(
            text=self.text,
            usage=self.usage,
            error_type=self.error_type,
            diagnostics={"model_signature": self.signature},
        )

    def close(self):
        self.closed += 1


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = prefilter_fixtures.PrefilterTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

        self.index = self.fixture.index
        self.pool = self.fixture.pool
        self.members = self.fixture.members
        self.request = self.fixture.request()
        self.prefilter = self.fixture.service.prefilter(self.request)
        self.assertEqual(self.prefilter.status, "ok")

    def selector(self, client=None, factory=None):
        selector = (
            LLMExperienceSelector(self.index, client_factory=factory)
            if factory is not None
            else LLMExperienceSelector(
                self.index,
                client or FakeClient(),
            )
        )
        self.addCleanup(selector.close)
        return selector

    def suspend(self, position=0):
        member = self.prefilter.candidates[position]["member"]
        return self.members.quality.submit_feedback(Feedback(
            member["candidate"].execution_id,
            member["candidate"].content_version,
            "problem",
            comment="选择期间收到负面反馈",
        ))

    def test_selected_candidate_and_two_read_checks(self):
        client = FakeClient(selected_id="c2")
        selector = self.selector(client)
        before = active_fixtures.database_snapshot(self.members.db)
        encoding_calls = self.fixture.encoder.calls

        with patch.object(
            self.index,
            "inspect",
            wraps=self.index.inspect,
        ) as inspect:
            result = selector.select(self.request, self.prefilter)

        self.assertEqual(result.reason_code, "selected")
        self.assertEqual(result.selected_id, "c2")
        self.assertIs(result.selected, self.prefilter.candidates[1])
        self.assertEqual(client.calls, 1)
        self.assertEqual(inspect.call_count, 2)
        self.assertEqual(
            result.diagnostics["before_call"]["checked_aliases"],
            ["c1", "c2", "c3"],
        )
        self.assertEqual(
            result.diagnostics["after_call"]["checked_aliases"],
            ["c2"],
        )
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(self.fixture.encoder.calls, encoding_calls)
        self.assertEqual(
            active_fixtures.database_snapshot(self.members.db),
            before,
        )

        serialized = result.to_dict()
        self.assertTrue(serialized["selected"]["content"]["sql"])

    def test_null_does_not_change_admission(self):
        before = active_fixtures.database_snapshot(self.members.db)
        client = FakeClient(selected_id=None)
        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )

        self.assertEqual(result.reason_code, "no_selection")
        self.assertIsNone(result.selected)
        self.assertEqual(client.calls, 1)
        self.assertEqual(
            active_fixtures.database_snapshot(self.members.db),
            before,
        )

    def test_factory_is_lazy_and_reused(self):
        client = FakeClient()
        factory = Mock(return_value=client)
        selector = self.selector(factory=factory)

        selector.select(self.request, self.prefilter)
        selector.select(self.request, self.prefilter)
        factory.assert_called_once_with()
        self.assertEqual(client.calls, 2)

        selector.close()
        selector.close()
        self.assertEqual(client.closed, 1)

    def test_disabled_and_mismatched_request_do_not_create_client(self):
        factory = Mock()
        selector = self.selector(factory=factory)

        for request in (
            replace(self.request, limit=0),
            replace(self.request, question="不同问题"),
        ):
            with self.subTest(question=request.question):
                result = selector.select(request, self.prefilter)
                self.assertIsNone(result.selected)
                self.assertEqual(result.usage.model_call_count, 0)

        factory.assert_not_called()

    def test_changed_candidate_before_call_skips_model(self):
        client = FakeClient()
        self.suspend(1)

        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )
        self.assertEqual(
            result.reason_code,
            "candidates_changed_before_call",
        )
        self.assertEqual(client.calls, 0)
        self.assertEqual(result.usage.model_call_count, 0)

    def test_selected_suspended_during_call_is_discarded(self):
        client = FakeClient(callback=lambda: self.suspend(0))
        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )

        self.assertEqual(
            result.reason_code,
            "selected_changed_after_call",
        )
        self.assertIsNone(result.selected)
        self.assertIsNone(result.selected_id)
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.usage.prompt_tokens, 20)

    def test_other_candidate_suspended_does_not_discard_valid_choice(self):
        client = FakeClient(callback=lambda: self.suspend(1))
        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )

        self.assertEqual(result.reason_code, "selected")
        self.assertEqual(result.selected_id, "c1")
        self.assertEqual(client.calls, 1)

    def test_correction_during_call_does_not_replace_selected_version(self):
        member = self.prefilter.candidates[0]["member"]

        def correct():
            self.members.quality.submit_feedback(Feedback(
                member["candidate"].execution_id,
                member["candidate"].content_version,
                "correction",
                corrected_sql=(
                    "SELECT DISTINCT enterprise_id FROM enterprise_basic;"
                ),
                confirmed=True,
                notes="必须按企业去重。",
            ))

        client = FakeClient(callback=correct)
        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )

        self.assertEqual(
            result.reason_code,
            "selected_changed_after_call",
        )
        self.assertIsNone(result.selected)
        self.assertEqual(client.calls, 1)

    def test_invalid_output_has_no_top1_fallback(self):
        outputs = (
            "not-json",
            '{"selected_id":"c99","reason":"未知标识"}',
            '{"selected_id":"c1","reason":"说明","extra":1}',
            '```json\n{"selected_id":"c1","reason":"说明"}\n```',
        )
        for text in outputs:
            with self.subTest(text=text):
                client = FakeClient(text=text)
                result = self.selector(client).select(
                    self.request,
                    self.prefilter,
                )
                self.assertEqual(
                    result.reason_code,
                    "invalid_model_output",
                )
                self.assertIsNone(result.selected)
                self.assertEqual(client.calls, 1)
                self.assertEqual(result.usage.completion_tokens, 8)

    def test_timeout_keeps_unknown_cost(self):
        usage = DifficultyUsage(
            elapsed_seconds=0.03,
            model_call_count=1,
            http_request_count=1,
            retry_count=0,
            prompt_tokens=None,
            completion_tokens=None,
            token_usage_complete=False,
        )
        client = FakeClient(
            error_type="SelectorDeadlineExceeded",
            usage=usage,
        )
        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )

        self.assertEqual(result.reason_code, "model_timeout")
        self.assertIsNone(result.selected)
        self.assertEqual(result.usage, usage)
        self.assertEqual(client.calls, 1)

    def test_factory_failure_is_zero_call(self):
        factory = Mock(
            side_effect=ValueError("private-error-body")
        )
        result = self.selector(factory=factory).select(
            self.request,
            self.prefilter,
        )

        self.assertEqual(result.reason_code, "model_unavailable")
        self.assertEqual(result.usage.model_call_count, 0)
        self.assertNotIn(
            "private-error-body",
            json.dumps(result.to_dict(), ensure_ascii=False),
        )

    def test_post_read_failure_discards_choice_and_keeps_cost(self):
        original = self.index.inspect
        calls = 0

        def inspect(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise sqlite3.OperationalError("private-error-body")
            return original(**kwargs)

        client = FakeClient()
        with patch.object(self.index, "inspect", side_effect=inspect):
            result = self.selector(client).select(
                self.request,
                self.prefilter,
            )

        self.assertEqual(
            result.reason_code,
            "recheck_failed_after_call",
        )
        self.assertIsNone(result.selected)
        self.assertEqual(result.usage.prompt_tokens, 20)

    def test_changed_in_memory_sql_is_not_sent(self):
        client = FakeClient()
        self.prefilter.candidates[0]["member"]["quality"].content["sql"] = (
            "SELECT 'unexpected';"
        )

        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )
        self.assertEqual(
            result.reason_code,
            "candidates_changed_before_call",
        )
        self.assertEqual(client.calls, 0)

    def test_missing_binding_is_not_rebuilt(self):
        index_key = self.prefilter.candidates[0]["binding"]["index_key"]
        with self.pool.connection(write=True) as connection:
            connection.execute(
                "DELETE FROM psu_retrieval_index_members WHERE index_key=?",
                (index_key,),
            )

        before = active_fixtures.database_snapshot(self.members.db)
        client = FakeClient()
        result = self.selector(client).select(
            self.request,
            self.prefilter,
        )

        self.assertEqual(
            result.reason_code,
            "candidates_changed_before_call",
        )
        self.assertEqual(client.calls, 0)
        self.assertEqual(
            active_fixtures.database_snapshot(self.members.db),
            before,
        )

    def test_cli_uses_existing_services_and_writes_report(self):
        request_file = self.members.root / "selector-request.json"
        report_file = self.members.root / "selector-report.json"
        request_file.write_text(
            json.dumps(asdict(self.request), ensure_ascii=False),
            encoding="utf-8",
        )

        client = FakeClient()
        before = active_fixtures.database_snapshot(self.members.db)

        with (
            patch.object(
                select_experience,
                "EmbeddingService",
                return_value=self.fixture.embeddings,
            ),
            patch.object(
                select_experience,
                "_create_client",
                return_value=client,
            ),
            redirect_stdout(io.StringIO()),
        ):
            code = select_experience.main([
                "--store", str(self.members.db),
                "--request", str(request_file),
                "--report", str(report_file),
            ])

        self.assertEqual(code, 0)
        report = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["selection"]["reason_code"], "selected")
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.closed, 1)
        self.assertEqual(
            report["prefilter"]["usage"]["encoded_text_count"],
            0,
        )
        self.assertEqual(
            active_fixtures.database_snapshot(self.members.db),
            before,
        )

    def test_cli_report_cannot_overwrite_database(self):
        with (
            patch.object(select_experience, "_create_client") as factory,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                select_experience.main([
                    "--store", str(self.members.db),
                    "--request", str(self.members.root / "request.json"),
                    "--report", str(self.members.db),
                ])

        self.assertEqual(raised.exception.code, 2)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
