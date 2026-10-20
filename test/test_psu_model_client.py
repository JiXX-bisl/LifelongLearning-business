"""共享客户端与择优截止时间测试，不访问模型服务或数据库。"""

from __future__ import annotations

import asyncio
import json
import os
import time
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import src.lifelong_learning.language_models.openai_compatible as lm

from src.chatflow_agent.telemetry import TaskTelemetry
from src.lifelong_learning.core.models import ChatHistory, Role
from src.psu_memory.pool._common import canonical_json, sha256_text
from src.psu_memory.pool.difficulty_llm import LocalDifficultyClient
from src.psu_memory.retrieval.embedding import embedding_signature
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.selector_llm import LocalSelectorClient


class ModelClientTests(unittest.TestCase):
    def setUp(self):
        self.config = RetrievalConfig()
        self.model_config = {
            "type": "openai_compatible",
            "model_name": "kimi-k2.6",
            "temperature": 0.9,
            "max_tokens": 4096,
            "request_interval_seconds": 10,
        }

        self.sdk_options = []
        self.http_options = []
        self.sent = []
        self.clients = []
        self.delay = 0.0
        self.failure = False
        self.with_usage = True
        self.finish_reason = "stop"
        self.cancelled = 0
        self.completed = 0

        environment = patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-secret",
                "OPENAI_BASE_URL": "https://example.invalid/v1",
                "OPENAI_MODEL_NAME": "",
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

        factories = (
            ("DefaultHttpx2Client", self.sync_transport),
            ("DefaultAsyncHttpx2Client", self.async_transport),
            ("OpenAI", self.sync_sdk),
            ("AsyncOpenAI", self.async_sdk),
        )
        for name, factory in factories:
            patcher = patch.object(lm, name, side_effect=factory)
            patcher.start()
            self.addCleanup(patcher.stop)

    def sync_transport(self, **options):
        self.http_options.append(options)
        return SimpleNamespace(
            hooks=options["event_hooks"],
            close=Mock(),
        )

    def async_transport(self, **options):
        self.http_options.append(options)
        return SimpleNamespace(
            hooks=options["event_hooks"],
            aclose=AsyncMock(),
        )

    def completion(self):
        usage = None
        if self.with_usage:
            usage = SimpleNamespace(
                model_dump=lambda **kwargs: {
                    "prompt_tokens": 20,
                    "completion_tokens": 8,
                    "total_tokens": 28,
                }
            )

        return SimpleNamespace(
            usage=usage,
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(
                        content=(
                            '{"selected_id":"c1",'
                            '"reason":"可借用查询方法。"}'
                        )
                    ),
                )
            ],
        )

    def sync_sdk(self, **options):
        self.sdk_options.append(options)
        transport = options["http_client"]

        def create(**parameters):
            self.sent.append(parameters)
            request = SimpleNamespace(
                headers={"x-stainless-retry-count": "0"}
            )
            for hook in transport.hooks["request"]:
                hook(request)

            if self.failure:
                raise RuntimeError("private-error-body")

            self.completed += 1
            return self.completion()

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            ),
            close=Mock(),
        )
        self.clients.append(client)
        return client

    def async_sdk(self, **options):
        self.sdk_options.append(options)
        transport = options["http_client"]

        async def create(**parameters):
            self.sent.append(parameters)
            request = SimpleNamespace(
                headers={"x-stainless-retry-count": "0"}
            )
            for hook in transport.hooks["request"]:
                await hook(request)

            try:
                if self.delay:
                    await asyncio.sleep(self.delay)
                if self.failure:
                    raise RuntimeError("private-error-body")
                self.completed += 1
                return self.completion()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            ),
            close=AsyncMock(),
        )
        self.clients.append(client)
        return client

    def selector(self, config=None):
        client = LocalSelectorClient(
            self.model_config,
            retrieval_config=config or self.config,
        )
        self.addCleanup(client.close)
        return client

    @staticmethod
    def call(client):
        return client.generate(
            system_prompt="仅返回 JSON。",
            user_text='{"question":"统计企业数量"}',
        )

    def test_legacy_signature_and_sdk_defaults(self):
        client = LocalDifficultyClient(self.model_config)
        self.addCleanup(client.close)

        expected = canonical_json({
            "adapter_version": "local_difficulty_client_v1",
            "model_name": "kimi-k2.6",
            "endpoint_sha256": sha256_text(
                "https://example.invalid/v1"
            ),
            "temperature": 0.9,
            "max_tokens": 4096,
            "request_interval_seconds": 10,
        })
        self.assertEqual(client.signature, expected)

        reply = self.call(client)
        self.assertIsNone(reply.error_type)
        self.assertEqual(reply.usage.http_request_count, 1)
        self.assertEqual(reply.usage.prompt_tokens, 20)
        self.assertTrue(reply.usage.token_usage_complete)

        self.assertNotIn("timeout", self.sdk_options[0])
        self.assertNotIn("max_retries", self.sdk_options[0])
        self.assertNotIn("response_format", self.sent[0])
        self.assertNotIn("extra_body", self.sent[0])

    def test_environment_model_name_overrides_all_clients_and_cache_signatures(self):
        clients = []
        with patch.dict(os.environ, {"OPENAI_MODEL_NAME": "  exported-model  "}):
            clients = [
                LocalDifficultyClient({**self.model_config, "request_interval_seconds": 0}),
                LocalSelectorClient(
                    self.model_config,
                    retrieval_config=replace(self.config, selector_model_name="configured-selector"),
                    model_name="explicit-selector",
                ),
            ]
        for client in clients:
            self.addCleanup(client.close)
            self.assertEqual(client._model.model_name, "exported-model")
            self.assertEqual(json.loads(client.signature)["model_name"], "exported-model")
            self.assertIsNone(self.call(client).error_type)
        self.assertEqual([item["model"] for item in self.sent], ["exported-model"] * 2)
        self.assertEqual(self.model_config["model_name"], "kimi-k2.6")

        with patch.dict(os.environ, {"OPENAI_MODEL_NAME": "another-model"}):
            changed = LocalDifficultyClient({**self.model_config, "request_interval_seconds": 0})
        self.addCleanup(changed.close)
        self.assertNotEqual(changed.signature, clients[0].signature)
        self.assertEqual(clients[0]._model.model_name, "exported-model")

    def test_missing_or_blank_environment_model_keeps_configured_names(self):
        for value in (None, "", " \t "):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop("OPENAI_MODEL_NAME", None)
                else:
                    os.environ["OPENAI_MODEL_NAME"] = value
                difficulty = LocalDifficultyClient(self.model_config)
                selector = LocalSelectorClient(
                    self.model_config,
                    retrieval_config=replace(self.config, selector_model_name="configured-selector"),
                )
                self.addCleanup(difficulty.close)
                self.addCleanup(selector.close)
                self.assertEqual(difficulty._model.model_name, "kimi-k2.6")
                self.assertEqual(selector._model.model_name, "configured-selector")

    def test_existing_business_observer_contract(self):
        model = lm.OpenAICompatibleLanguageModel(
            model_name="kimi-k2.6",
            api_key="test-secret",
        )
        self.addCleanup(model.close)

        telemetry = TaskTelemetry()
        model.call_observer = telemetry.record_call
        history = ChatHistory()
        history.append(Role.USER, "查询企业")

        model.generate(history, "业务提示词")
        metrics = telemetry.snapshot()["metrics"]
        self.assertEqual(metrics["model_call_count"], 1)
        self.assertEqual(metrics["request_count"], 1)
        self.assertEqual(metrics["usage"]["prompt_tokens"], 20)

    def test_selector_parameters_are_independent(self):
        client = self.selector()
        reply = self.call(client)
        self.assertIsNone(reply.error_type)

        parameters = self.sent[0]
        self.assertEqual(parameters["model"], "kimi-k2.6")
        self.assertEqual(parameters["temperature"], 0.6)
        self.assertEqual(parameters["max_tokens"], 512)
        self.assertEqual(
            parameters["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            parameters["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(self.sdk_options[0]["max_retries"], 0)
        self.assertEqual(self.sdk_options[0]["timeout"], 30.0)
        self.assertEqual(reply.usage.retry_count, 0)

        # 原配置没有被择优参数改写。
        self.assertEqual(self.model_config["max_tokens"], 4096)
        self.assertEqual(self.model_config["temperature"], 0.9)

    def test_yaml_entry_reads_only_model_config(self):
        client = LocalSelectorClient.from_yaml(
            "configs/business/query.yaml",
            retrieval_config=self.config,
        )
        self.addCleanup(client.close)

        self.assertEqual(len(self.sdk_options), 0)
        self.assertEqual(client._model.model_name, "kimi-k2.6")

    def test_client_and_loop_are_reused(self):
        client = self.selector()
        first = self.call(client)
        runner = client._runner
        loop = runner.get_loop()

        second = self.call(client)
        self.assertIsNone(first.error_type)
        self.assertIsNone(second.error_type)
        self.assertIs(client._runner, runner)
        self.assertIs(client._runner.get_loop(), loop)
        self.assertEqual(len(self.sdk_options), 1)
        self.assertEqual(len(self.sent), 2)

        client.close()
        self.clients[0].close.assert_awaited_once()
        client.close()

    def test_queue_timeout_sends_no_request(self):
        config = replace(
            self.config,
            selector_timeout_seconds=0.03,
        )
        client = self.selector(config)

        client._lock.acquire()
        try:
            reply = self.call(client)
        finally:
            client._lock.release()

        self.assertEqual(
            reply.error_type,
            "SelectorDeadlineExceeded",
        )
        self.assertEqual(reply.usage.model_call_count, 0)
        self.assertEqual(reply.usage.http_request_count, 0)
        self.assertEqual(reply.usage.prompt_tokens, 0)
        self.assertEqual(len(self.sdk_options), 0)
        self.assertGreater(
            reply.diagnostics["stage_seconds"]["queue_wait"],
            0,
        )

    def test_deadline_cancels_request_and_allows_next_call(self):
        config = replace(
            self.config,
            selector_timeout_seconds=0.03,
        )
        client = self.selector(config)
        self.delay = 1.0

        started = time.perf_counter()
        reply = self.call(client)
        elapsed = time.perf_counter() - started

        self.assertEqual(
            reply.error_type,
            "SelectorDeadlineExceeded",
        )
        self.assertLess(elapsed, 0.5)
        self.assertEqual(self.cancelled, 1)
        self.assertEqual(self.completed, 0)
        self.assertEqual(reply.text, "")
        self.assertEqual(reply.usage.model_call_count, 1)
        self.assertEqual(reply.usage.http_request_count, 1)
        self.assertEqual(reply.usage.retry_count, 0)
        self.assertIsNone(reply.usage.prompt_tokens)
        self.assertIsNone(reply.usage.completion_tokens)
        self.assertFalse(reply.usage.token_usage_complete)
        self.assertIsNone(client._model._active_call)
        self.assertIsNone(client._model.call_observer)

        self.delay = 0.0
        recovered = self.call(client)
        self.assertIsNone(recovered.error_type)
        self.assertEqual(self.completed, 1)
        self.assertEqual(len(self.sdk_options), 1)

    def test_network_failure_has_unknown_tokens(self):
        client = self.selector()
        self.failure = True

        reply = self.call(client)
        self.assertEqual(reply.error_type, "RuntimeError")
        self.assertEqual(reply.usage.http_request_count, 1)
        self.assertIsNone(reply.usage.prompt_tokens)
        self.assertFalse(reply.usage.token_usage_complete)

        serialized = json.dumps(
            asdict(reply),
            ensure_ascii=False,
        )
        for private_value in (
            "test-secret",
            "https://example.invalid/v1",
            "private-error-body",
        ):
            self.assertNotIn(private_value, serialized)

    def test_missing_server_usage_is_not_zero(self):
        client = self.selector()
        self.with_usage = False

        reply = self.call(client)
        self.assertIsNone(reply.error_type)
        self.assertEqual(reply.usage.http_request_count, 1)
        self.assertIsNone(reply.usage.prompt_tokens)
        self.assertIsNone(reply.usage.completion_tokens)
        self.assertFalse(reply.usage.token_usage_complete)

    def test_truncated_output_is_rejected_but_cost_is_kept(self):
        client = self.selector()
        self.finish_reason = "length"

        reply = self.call(client)
        self.assertEqual(reply.error_type, "TruncatedModelOutput")
        self.assertEqual(reply.text, "")
        self.assertEqual(reply.usage.prompt_tokens, 20)
        self.assertEqual(reply.usage.completion_tokens, 8)
        self.assertTrue(reply.usage.token_usage_complete)

    def test_closed_client_sends_no_request(self):
        client = self.selector()
        client.close()

        reply = self.call(client)
        self.assertEqual(reply.error_type, "ClientClosed")
        self.assertEqual(reply.usage.model_call_count, 0)
        self.assertEqual(reply.usage.http_request_count, 0)
        self.assertEqual(len(self.sdk_options), 0)

    def test_sync_entry_does_not_nest_event_loop(self):
        client = self.selector()

        async def run():
            return self.call(client)

        reply = asyncio.run(run())
        self.assertEqual(reply.error_type, "ActiveEventLoop")
        self.assertEqual(reply.usage.model_call_count, 0)
        self.assertEqual(len(self.sdk_options), 0)

    def test_selector_settings_do_not_change_embedding_signature(self):
        changed = replace(
            self.config,
            selector_temperature=0.5,
            selector_max_output_tokens=256,
            selector_extra_body={
                "chat_template_kwargs": {"thinking": False},
            },
        )
        self.assertEqual(
            embedding_signature(self.config),
            embedding_signature(changed),
        )

    def test_invalid_selector_settings(self):
        for temperature in (True, -0.1, 2.1, float("nan")):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    replace(
                        self.config,
                        selector_temperature=temperature,
                    )

        for body in ([], {"value": float("nan")}):
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    replace(
                        self.config,
                        selector_extra_body=body,
                    )


if __name__ == "__main__":
    unittest.main()
