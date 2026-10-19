"""一次择优调用：独立参数、可取消请求和完整截止时间。"""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import replace

from ..model_client import LocalModelClient
from ..pool.pool_models import DifficultyUsage, ModelReply
from .retrieval_models import RetrievalConfig


class LocalSelectorClient(LocalModelClient):
    ADAPTER_VERSION = "local_selector_client_v1"

    def __init__(
        self,
        model_config: dict,
        *,
        retrieval_config: RetrievalConfig,
        model_name: str | None = None,
    ) -> None:
        if not isinstance(retrieval_config, RetrievalConfig):
            raise TypeError("retrieval_config 必须是 RetrievalConfig")

        self.config = retrieval_config
        self.timeout_seconds = float(
            retrieval_config.selector_timeout_seconds
        )

        super().__init__(
            model_config,
            model_name=model_name or retrieval_config.selector_model_name,
            generation_options={
                "temperature": retrieval_config.selector_temperature,
                "max_tokens": retrieval_config.selector_max_output_tokens,
                "request_interval_seconds": 0.0,
                "timeout": self.timeout_seconds,
                "max_retries": retrieval_config.selector_max_retries,
                "response_format": {"type": "json_object"},
                "extra_body": deepcopy(
                    retrieval_config.selector_extra_body
                ),
                "reject_truncated": True,
            },
        )

        # 同一实例持续复用同一事件循环和异步 HTTP 客户端。
        self._runner: asyncio.Runner | None = None
        self._closed = False

    @staticmethod
    def _inside_event_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _without_call(
        self,
        error_type: str,
        *,
        started: float,
        queue_wait_seconds: float = 0.0,
    ) -> ModelReply:
        elapsed = time.perf_counter() - started
        return ModelReply(
            text="",
            error_type=error_type,
            usage=DifficultyUsage(elapsed_seconds=elapsed),
            diagnostics={
                "model_signature": self.signature,
                "stage_seconds": {
                    "queue_wait": queue_wait_seconds,
                    "client_call": 0.0,
                    "total": elapsed,
                },
            },
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_text: str,
    ) -> ModelReply:
        started = time.perf_counter()
        deadline = time.monotonic() + self.timeout_seconds

        # 本地版本提供同步入口，不嵌套运行已有事件循环。
        if self._inside_event_loop():
            return self._without_call(
                "ActiveEventLoop",
                started=started,
            )

        acquired = self._lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        )
        queue_wait = time.perf_counter() - started

        if not acquired:
            return self._without_call(
                "SelectorDeadlineExceeded",
                started=started,
                queue_wait_seconds=queue_wait,
            )

        try:
            if self._closed:
                return self._without_call(
                    "ClientClosed",
                    started=started,
                    queue_wait_seconds=queue_wait,
                )

            if time.monotonic() >= deadline:
                return self._without_call(
                    "SelectorDeadlineExceeded",
                    started=started,
                    queue_wait_seconds=queue_wait,
                )

            if self._runner is None:
                self._runner = asyncio.Runner()

            async def request() -> str:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError

                # 截止时间包含前面的排队和启动耗时。
                # 到期取消当前请求，等待取消完成后才返回。
                async with asyncio.timeout(remaining):
                    return await self._model.generate_async(
                        chat_history=self._make_history(user_text),
                        system_prompt=system_prompt,
                    )

            reply = self._invoke_recorded(
                lambda: self._runner.run(request()),
                started=started,
                queue_wait_seconds=queue_wait,
            )

            if reply.error_type == "TimeoutError":
                reply = replace(
                    reply,
                    error_type="SelectorDeadlineExceeded",
                )

            return reply
        finally:
            self._lock.release()

    def close(self) -> None:
        if self._inside_event_loop():
            raise RuntimeError(
                "请在同步调用位置关闭 LocalSelectorClient"
            )

        with self._lock:
            if self._closed:
                return

            try:
                if self._runner is not None:
                    self._runner.run(self._model.aclose())
            finally:
                try:
                    if self._runner is not None:
                        self._runner.close()
                        self._runner = None
                finally:
                    self._model.close()
                    self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()