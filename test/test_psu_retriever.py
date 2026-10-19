"""统一 retrieve 验证；不调用真实模型，不修改正式候选库。"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import test_psu_active_members as active_fixtures
import test_psu_selector as selector_fixtures

from src.psu_memory.interfaces import ExperienceRetriever, MemoryComponent
from src.psu_memory.models import ContextItem, RetrievalResult, UsageMetrics
from src.psu_memory.pool.pool_models import DifficultyUsage
from src.psu_memory.retrieval import service as service_module
from src.psu_memory.retrieval.selector import LLMExperienceSelector
from src.psu_memory.retrieval.service import PSURetriever


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.fixture = selector_fixtures.SelectorTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

        self.prefilter = self.fixture.fixture.service
        self.request = self.fixture.request
        self.db = self.fixture.members.db

    def service(self, client=None):
        client = client or selector_fixtures.FakeClient()
        selector = LLMExperienceSelector(
            self.fixture.index,
            client,
        )
        self.addCleanup(selector.close)

        service = PSURetriever(self.prefilter, selector)
        self.addCleanup(service.close)
        return service, client

    def test_old_public_constructors_remain_valid(self):
        item = ContextItem("id", "text", "source", "v1")
        result = RetrievalResult([item], "说明", UsageMetrics())

        self.assertEqual(item.metadata, {})
        self.assertEqual(result.reason_code, "")
        self.assertEqual(result.diagnostics, {})
        self.assertEqual(result.to_dict()["items"][0]["content"], "text")

        item.metadata["example"] = True
        another = ContextItem("other", "text", "source", "v1")
        self.assertEqual(another.metadata, {})

        self.assertTrue(hasattr(ExperienceRetriever, "retrieve"))
        self.assertTrue(hasattr(MemoryComponent, "retrieve"))
        self.assertTrue(hasattr(MemoryComponent, "observe"))

    def test_retrieve_returns_existing_types_and_snapshot(self):
        service, client = self.service()
        before = active_fixtures.database_snapshot(self.db)

        result = service.retrieve(self.request)
        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.reason_code, "retrieved")
        self.assertEqual(len(result.items), 1)
        self.assertEqual(client.calls, 1)

        item = result.items[0]
        self.assertIsInstance(item, ContextItem)
        snapshot = item.metadata["snapshot"]
        self.assertEqual(item.item_id, snapshot["identity"]["execution_id"])
        self.assertEqual(item.version, snapshot["identity"]["content_version"])
        self.assertIn("历史 SQL：", item.content)
        self.assertTrue(item.source_ref.startswith("psu-memory:"))

        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
        self.assertEqual(
            active_fixtures.database_snapshot(self.db),
            before,
        )

    def test_costs_are_not_counted_twice(self):
        service, _ = self.service()
        result = service.retrieve(self.request)

        self.assertEqual(result.usage.model_call_count, 1)
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(result.usage.completion_tokens, 8)
        self.assertEqual(result.usage.embedding_input_tokens, 0)

        total = result.diagnostics["cost_summary"]
        self.assertEqual(total["http_request_count"], 1)
        self.assertEqual(total["retry_count"], 0)
        self.assertEqual(total["total_llm_tokens"], 28)
        self.assertTrue(total["token_usage_complete"])

        phases = result.diagnostics["usage_by_phase"]
        self.assertEqual(phases["prefilter"]["model_call_count"], 0)
        self.assertEqual(phases["selection"]["model_call_count"], 1)
        self.assertEqual(phases["rendering"]["model_call_count"], 0)

    def test_model_null_preserves_cost_and_data(self):
        service, client = self.service(
            selector_fixtures.FakeClient(selected_id=None)
        )
        before = active_fixtures.database_snapshot(self.db)
        result = service.retrieve(self.request)

        self.assertEqual(result.reason_code, "no_selection")
        self.assertEqual(result.items, [])
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(client.calls, 1)
        self.assertEqual(
            active_fixtures.database_snapshot(self.db),
            before,
        )

    def test_small_budget_preserves_selection_cost(self):
        service, client = self.service()
        result = service.retrieve(
            replace(self.request, max_context_chars=1)
        )

        self.assertEqual(result.reason_code, "context_budget_exceeded")
        self.assertEqual(result.items, [])
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.usage.model_call_count, 1)
        self.assertEqual(result.usage.prompt_tokens, 20)

    def test_explicit_disable_skips_selector(self):
        service, client = self.service()
        calls = self.fixture.fixture.encoder.calls

        with patch.object(
            service.selector,
            "select",
            wraps=service.selector.select,
        ) as select:
            for request in (
                replace(self.request, limit=0),
                replace(self.request, max_context_chars=0),
            ):
                result = service.retrieve(request)
                self.assertEqual(result.reason_code, "disabled")
                self.assertEqual(result.items, [])
                self.assertEqual(result.usage.model_call_count, 0)
                self.assertEqual(result.usage.embedding_input_tokens, 0)

        select.assert_not_called()
        self.assertEqual(client.calls, 0)
        self.assertEqual(self.fixture.fixture.encoder.calls, calls)

    def test_missing_scope_is_zero_model_call(self):
        service, client = self.service()
        result = service.retrieve(
            replace(self.request, namespace="missing")
        )

        self.assertEqual(result.reason_code, "no_members")
        self.assertEqual(result.items, [])
        self.assertEqual(result.usage.model_call_count, 0)
        self.assertEqual(client.calls, 0)

    def test_missing_index_is_not_rebuilt(self):
        with self.fixture.pool.connection(write=True) as connection:
            connection.execute(
                "DELETE FROM psu_retrieval_index_members"
            )

        before = active_fixtures.database_snapshot(self.db)
        service, client = self.service()
        result = service.retrieve(self.request)

        self.assertEqual(result.reason_code, "index_not_ready")
        self.assertEqual(result.items, [])
        self.assertEqual(client.calls, 0)
        self.assertEqual(
            active_fixtures.database_snapshot(self.db),
            before,
        )

    def test_timeout_keeps_unknown_tokens(self):
        usage = DifficultyUsage(
            elapsed_seconds=0.03,
            model_call_count=1,
            http_request_count=1,
            retry_count=0,
            prompt_tokens=None,
            completion_tokens=None,
            token_usage_complete=False,
        )
        service, client = self.service(
            selector_fixtures.FakeClient(
                error_type="SelectorDeadlineExceeded",
                usage=usage,
            )
        )
        result = service.retrieve(self.request)

        self.assertEqual(result.reason_code, "model_timeout")
        self.assertEqual(result.items, [])
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.usage.model_call_count, 1)
        self.assertIsNone(result.usage.prompt_tokens)
        self.assertIsNone(
            result.diagnostics["cost_summary"]["total_llm_tokens"]
        )
        self.assertFalse(
            result.diagnostics["cost_summary"]["token_usage_complete"]
        )

    def test_negative_feedback_during_selection_returns_no_item(self):
        service, client = self.service(
            selector_fixtures.FakeClient(
                callback=lambda: self.fixture.suspend(0)
            )
        )
        result = service.retrieve(self.request)

        self.assertEqual(
            result.reason_code,
            "selected_changed_after_call",
        )
        self.assertEqual(result.items, [])
        self.assertEqual(client.calls, 1)
        self.assertEqual(result.usage.prompt_tokens, 20)

    def test_rendering_exception_keeps_completed_cost(self):
        service, _ = self.service()

        with patch.object(
            service_module,
            "render_experience",
            side_effect=OSError("PRIVATE_ERROR_BODY"),
        ):
            result = service.retrieve(self.request)

        self.assertEqual(result.reason_code, "retrieval_failed")
        self.assertEqual(result.items, [])
        self.assertEqual(result.usage.model_call_count, 1)
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(
            result.diagnostics["failed_phase"],
            "rendering",
        )
        self.assertNotIn(
            "PRIVATE_ERROR_BODY",
            json.dumps(result.to_dict(), ensure_ascii=False),
        )

    def test_unexpected_selector_exception_has_unknown_cost(self):
        service, _ = self.service()
        with patch.object(
            service.selector,
            "select",
            side_effect=RuntimeError("PRIVATE_ERROR_BODY"),
        ):
            result = service.retrieve(self.request)

        self.assertEqual(result.reason_code, "retrieval_failed")
        self.assertEqual(result.items, [])
        self.assertIsNone(result.usage.model_call_count)
        self.assertIsNone(result.usage.prompt_tokens)
        self.assertIsNone(
            result.diagnostics["cost_summary"]["http_request_count"]
        )
        self.assertFalse(
            result.diagnostics["cost_summary"]["token_usage_complete"]
        )

    def test_local_factory_reuses_services_and_closes_owned_client(self):
        client = selector_fixtures.FakeClient()
        factory = Mock(return_value=client)

        with (
            patch.object(
                service_module.SQLitePoolStore,
                "initialize",
                side_effect=AssertionError("禁止初始化"),
            ) as pool_initialize,
            patch.object(
                service_module.SQLiteRetrievalIndex,
                "initialize",
                side_effect=AssertionError("禁止初始化"),
            ) as index_initialize,
        ):
            service = PSURetriever.from_local(
                self.db,
                retrieval_config=self.prefilter.config,
                model_config_path=self.fixture.members.root / "missing.yaml",
                encoder=self.fixture.fixture.encoder,
                client_factory=factory,
            )
            self.addCleanup(service.close)

            first = service.retrieve(self.request)
            embeddings = service.prefilter.embeddings
            second = service.retrieve(self.request)

        self.assertEqual(first.reason_code, "retrieved")
        self.assertEqual(second.reason_code, "retrieved")
        self.assertIs(service.prefilter.embeddings, embeddings)
        factory.assert_called_once_with()
        self.assertEqual(client.calls, 2)
        self.assertEqual(first.usage.model_call_count, 1)
        self.assertEqual(second.usage.model_call_count, 1)
        pool_initialize.assert_not_called()
        index_initialize.assert_not_called()

        service.close()
        service.close()
        self.assertEqual(client.closed, 1)

        closed = service.retrieve(self.request)
        self.assertEqual(closed.reason_code, "service_closed")
        self.assertEqual(closed.usage.model_call_count, 0)
        self.assertEqual(client.calls, 2)

    def test_invalid_request_is_zero_cost(self):
        service, client = self.service()
        result = service.retrieve(
            replace(self.request, limit=-1)
        )

        self.assertEqual(result.reason_code, "invalid_request")
        self.assertEqual(result.items, [])
        self.assertEqual(result.usage.model_call_count, 0)
        self.assertEqual(result.usage.embedding_input_tokens, 0)
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()