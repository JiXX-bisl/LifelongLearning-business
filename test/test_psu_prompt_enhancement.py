"""提示词增强与使用前复核；不调用真实模型。"""

from __future__ import annotations

import copy
import io
import json
import sqlite3
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from unittest.mock import Mock, patch

import test_psu_active_members as active_fixtures
import test_psu_retriever as retriever_fixtures
import test_psu_selector as selector_fixtures
from test_psu_embedding import FakeEncoder

from src.psu_memory.cli import enhance_prompt as cli
from src.psu_memory.evaluation.quality_models import Feedback
from src.psu_memory.pool._common import sha256_text
from src.psu_memory.retrieval.augmentation import PROMPT_SEPARATOR
from src.psu_memory.retrieval.service import PSURetriever


class PromptEnhancementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = retriever_fixtures.RetrieverTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

        self.service, self.client = self.fixture.service()
        self.request = self.fixture.request
        self.db = self.fixture.db
        self.original = "原提示词\r\n当前问题：查询企业。\n末尾空格  \t"

    def test_automatic_retrieval_and_exact_prefix(self):
        before = active_fixtures.database_snapshot(self.db)
        text, result = self.service.enhance_prompt(
            self.original,
            self.request,
        )

        self.assertEqual(result.reason_code, "enhanced")
        self.assertEqual(
            text,
            self.original + PROMPT_SEPARATOR + result.items[0].content,
        )
        self.assertTrue(text.startswith(self.original))
        self.assertEqual(self.client.calls, 1)
        self.assertEqual(
            result.diagnostics["augmentation"]["usage"]["model_call_count"],
            0,
        )
        self.assertEqual(
            active_fixtures.database_snapshot(self.db),
            before,
        )

    def test_reused_result_does_not_query_again_or_mutate_source(self):
        source = self.service.retrieve(self.request)
        before = source.to_dict()

        with patch.object(
            self.service,
            "retrieve",
            side_effect=AssertionError("禁止重新检索"),
        ):
            text, result = self.service.enhance_prompt(
                self.original,
                self.request,
                retrieval_result=source,
            )

        self.assertEqual(result.reason_code, "enhanced")
        self.assertEqual(self.client.calls, 1)
        self.assertEqual(source.to_dict(), before)
        self.assertEqual(result.usage.model_call_count, 1)
        self.assertTrue(text.startswith(self.original))

    def test_repeated_enhancement_does_not_accumulate_old_extra_time(self):
        source = self.service.retrieve(self.request)
        _, first = self.service.enhance_prompt(
            self.original,
            self.request,
            retrieval_result=source,
        )
        _, second = self.service.enhance_prompt(
            self.original,
            self.request,
            retrieval_result=first,
        )

        base_seconds = source.diagnostics[
            "stage_seconds"
        ]["retrieve_total_seconds"]
        extra_seconds = second.diagnostics[
            "augmentation"
        ]["usage"]["elapsed_seconds"]
        self.assertAlmostEqual(
            second.usage.elapsed_seconds,
            base_seconds + extra_seconds,
        )
        self.assertEqual(second.usage.model_call_count, 1)
        self.assertEqual(self.client.calls, 1)

    def test_model_null_preserves_original(self):
        service, client = self.fixture.service(
            selector_fixtures.FakeClient(selected_id=None)
        )
        text, result = service.enhance_prompt(
            self.original,
            self.request,
        )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "no_selection")
        self.assertEqual(result.items, [])
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.usage.prompt_tokens, 20)

    def test_budget_too_small_preserves_original_and_cost(self):
        text, result = self.service.enhance_prompt(
            self.original,
            replace(self.request, max_context_chars=1),
        )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "context_budget_exceeded")
        self.assertEqual(result.usage.model_call_count, 1)
        self.assertEqual(self.client.calls, 1)

    def test_negative_feedback_after_retrieval_blocks_append(self):
        source = self.service.retrieve(self.request)
        self.fixture.fixture.suspend(0)

        text, result = self.service.enhance_prompt(
            self.original,
            self.request,
            retrieval_result=source,
        )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "experience_stale")
        self.assertEqual(result.items, [])
        self.assertEqual(len(source.items), 1)
        self.assertEqual(self.client.calls, 1)
        self.assertEqual(result.usage.prompt_tokens, 20)

    def test_correction_does_not_substitute_new_version(self):
        source = self.service.retrieve(self.request)
        identity = source.items[0].metadata["snapshot"]["identity"]

        self.fixture.fixture.members.quality.submit_feedback(Feedback(
            identity["execution_id"],
            identity["content_version"],
            "correction",
            corrected_sql=(
                "SELECT DISTINCT enterprise_id FROM enterprise_basic;"
            ),
            confirmed=True,
            notes="按企业去重。",
        ))

        text, result = self.service.enhance_prompt(
            self.original,
            self.request,
            retrieval_result=source,
        )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "experience_stale")
        self.assertEqual(result.items, [])
        self.assertEqual(self.client.calls, 1)

    def test_modified_text_is_rejected_even_if_hash_is_updated(self):
        source = self.service.retrieve(self.request)
        changed = copy.deepcopy(source)
        changed.items[0].content += "\n不属于历史记录的新增指令"
        changed.items[0].metadata["content_sha256"] = sha256_text(
            changed.items[0].content
        )

        text, result = self.service.enhance_prompt(
            self.original,
            self.request,
            retrieval_result=changed,
        )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "enhancement_rejected")
        self.assertEqual(
            result.diagnostics["augmentation"]["reason_code"],
            "experience_text_changed",
        )
        self.assertEqual(self.client.calls, 1)

    def test_other_request_is_not_reused_or_charged(self):
        source = self.service.retrieve(self.request)
        text, result = self.service.enhance_prompt(
            self.original,
            replace(self.request, task_id="another-request"),
            retrieval_result=source,
        )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "request_mismatch")
        self.assertEqual(result.usage.model_call_count, 0)
        self.assertEqual(
            result.diagnostics["rejected_result"]["source_usage"][
                "model_call_count"
            ],
            1,
        )
        self.assertEqual(self.client.calls, 1)

    def test_read_failure_preserves_original_and_completed_cost(self):
        source = self.service.retrieve(self.request)
        with patch.object(
            self.service.index,
            "inspect",
            side_effect=sqlite3.OperationalError("PRIVATE_ERROR_BODY"),
        ):
            text, result = self.service.enhance_prompt(
                self.original,
                self.request,
                retrieval_result=source,
            )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "enhancement_failed")
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertNotIn(
            "PRIVATE_ERROR_BODY",
            json.dumps(result.to_dict(), ensure_ascii=False),
        )

    def test_changed_default_budget_is_rechecked_without_model(self):
        source = self.service.retrieve(self.request)
        config = replace(self.service.config, max_context_chars=1)
        factory = Mock()

        with PSURetriever.from_local(
            self.db,
            retrieval_config=config,
            encoder=FakeEncoder(config),
            client_factory=factory,
        ) as other:
            text, result = other.enhance_prompt(
                self.original,
                self.request,
                retrieval_result=source,
            )

        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "context_budget_exceeded")
        factory.assert_not_called()

    def test_disabled_is_zero_call(self):
        text, result = self.service.enhance_prompt(
            self.original,
            replace(self.request, max_context_chars=0),
        )
        self.assertEqual(text, self.original)
        self.assertEqual(result.reason_code, "disabled")
        self.assertEqual(result.usage.model_call_count, 0)
        self.assertEqual(self.client.calls, 0)

    def test_cli_preserves_original_file_bytes(self):
        folder = self.fixture.fixture.members.root
        request_path = folder / "enhance-request.json"
        base_path = folder / "base-prompt.txt"
        output_path = folder / "enhanced-prompt.txt"
        report_path = folder / "enhancement.json"

        request_path.write_text(
            json.dumps(asdict(self.request), ensure_ascii=False),
            encoding="utf-8",
        )
        base_path.write_bytes(self.original.encode("utf-8"))

        with (
            patch.object(
                cli.PSURetriever,
                "from_local",
                return_value=self.service,
            ),
            redirect_stdout(io.StringIO()),
        ):
            code = cli.main([
                "--store", str(self.db),
                "--request", str(request_path),
                "--base-prompt", str(base_path),
                "--output-prompt", str(output_path),
                "--report", str(report_path),
            ])

        self.assertEqual(code, 0)
        self.assertTrue(
            output_path.read_bytes().startswith(base_path.read_bytes())
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["original_prompt"], self.original)
        self.assertEqual(report["status"], "enhanced")
        self.assertEqual(self.client.calls, 1)

    def test_cli_output_cannot_overwrite_database(self):
        folder = self.fixture.fixture.members.root
        with (
            patch.object(cli.PSURetriever, "from_local") as factory,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                cli.main([
                    "--store", str(self.db),
                    "--request", str(folder / "request.json"),
                    "--base-prompt", str(folder / "base.txt"),
                    "--output-prompt", str(self.db),
                    "--report", str(folder / "report.json"),
                ])

        self.assertEqual(raised.exception.code, 2)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()