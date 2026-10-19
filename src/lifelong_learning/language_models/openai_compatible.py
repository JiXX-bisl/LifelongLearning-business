from __future__ import annotations

import asyncio
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable

from openai import (
    AsyncOpenAI,
    DefaultAsyncHttpx2Client,
    DefaultHttpx2Client,
    OpenAI,
)
from openai.types.chat import ChatCompletionMessageParam

from src.lifelong_learning.core.models import ChatHistory, Role
from src.lifelong_learning.language_models.base import LanguageModel


class TruncatedModelOutput(RuntimeError):
    """输出达到 Token 上限，不能作为完整结果使用。"""


class OpenAICompatibleLanguageModel(LanguageModel):
    """共享消息构建、请求参数、响应解析和 HTTP 统计。"""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        request_interval_seconds: float | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        response_format: dict | None = None,
        extra_body: dict | None = None,
        reject_truncated: bool = False,
    ) -> None:
        resolved_name = model_name or os.environ.get("OPENAI_MODEL_NAME")
        if not resolved_name:
            raise ValueError(
                "Model name is missing. Set model_name in the "
                "configuration or export OPENAI_MODEL_NAME."
            )

        self.model_name = resolved_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.request_interval_seconds = request_interval_seconds

        # None 表示不覆盖 SDK 原有默认值。
        self.timeout = timeout
        self.max_retries = max_retries
        self.response_format = deepcopy(response_format)
        self.extra_body = deepcopy(extra_body)
        self.reject_truncated = reject_truncated

        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._last_request_started_at: float | None = None

        self.call_observer: Callable[..., None] | None = None
        self._active_call: dict[str, Any] | None = None

    @staticmethod
    def build_message_list(
        chat_history: ChatHistory,
        system_prompt: str,
    ) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        for message in chat_history.messages:
            messages.append({
                "role": "user" if message.role == Role.USER else "assistant",
                "content": message.content,
            })

        return messages

    def _client_options(self, http_client: Any) -> dict:
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        options = {
            "api_key": api_key,
            "http_client": http_client,
        }
        base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            options["base_url"] = base_url

        if self.timeout is not None:
            options["timeout"] = self.timeout
        if self.max_retries is not None:
            options["max_retries"] = self.max_retries

        return options

    def _http_options(self, request_hook: Callable) -> dict:
        options = {
            "event_hooks": {"request": [request_hook]},
        }
        if self.timeout is not None:
            options["timeout"] = self.timeout
        return options

    def _get_client(self) -> OpenAI:
        if self._client is None:
            http_client = DefaultHttpx2Client(
                **self._http_options(self._observe_request)
            )
            try:
                self._client = OpenAI(
                    **self._client_options(http_client)
                )
            except Exception:
                http_client.close()
                raise

        return self._client

    async def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            http_client = DefaultAsyncHttpx2Client(
                **self._http_options(self._observe_async_request)
            )
            try:
                self._async_client = AsyncOpenAI(
                    **self._client_options(http_client)
                )
            except BaseException:
                await http_client.aclose()
                raise

        return self._async_client

    def _observe_request(self, request: Any) -> None:
        record = self._active_call
        if record is None:
            return

        record["request_count"] += 1
        retry_text = request.headers.get("x-stainless-retry-count", "0")
        if retry_text.isdigit():
            record["retry_count"] = max(
                record["retry_count"],
                int(retry_text),
            )

    async def _observe_async_request(self, request: Any) -> None:
        self._observe_request(request)

    def _slot_wait_seconds(self) -> float:
        interval = self.request_interval_seconds
        if interval is None or interval <= 0:
            return 0.0
        if self._last_request_started_at is None:
            return 0.0

        return max(
            0.0,
            interval - (time.monotonic() - self._last_request_started_at),
        )

    def _wait_for_request_slot(self) -> None:
        interval = self.request_interval_seconds
        if interval is None or interval <= 0:
            return

        wait_seconds = self._slot_wait_seconds()
        if wait_seconds > 0:
            print(
                f"Waiting {wait_seconds:.1f}s "
                "for the API rate limit..."
            )
            time.sleep(wait_seconds)

        self._last_request_started_at = time.monotonic()

    async def _wait_for_async_request_slot(self) -> None:
        interval = self.request_interval_seconds
        if interval is None or interval <= 0:
            return

        wait_seconds = self._slot_wait_seconds()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        self._last_request_started_at = time.monotonic()

    def _request_parameters(
        self,
        chat_history: ChatHistory,
        system_prompt: str,
    ) -> dict:
        parameters = {
            "model": self.model_name,
            "messages": self.build_message_list(
                chat_history,
                system_prompt,
            ),
        }

        for name in ("temperature", "max_tokens"):
            value = getattr(self, name)
            if value is not None:
                parameters[name] = value

        if self.response_format is not None:
            parameters["response_format"] = deepcopy(self.response_format)
        if self.extra_body is not None:
            parameters["extra_body"] = deepcopy(self.extra_body)

        return parameters

    def _completion_text(self, completion: Any) -> str:
        # 先登记用量，之后即使输出被拒绝，也保留真实成本。
        if self._active_call is not None and completion.usage is not None:
            self._active_call["usage"] = completion.usage.model_dump(
                mode="json"
            )

        if not completion.choices:
            raise RuntimeError("The language model returned no choices.")

        choice = completion.choices[0]
        if (
            self.reject_truncated
            and getattr(choice, "finish_reason", None) == "length"
        ):
            raise TruncatedModelOutput(
                "The language model output reached its token limit."
            )

        content = choice.message.content
        if not content:
            raise RuntimeError("The language model returned empty content.")

        return content

    def _generate(
        self,
        chat_history: ChatHistory,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        parameters = self._request_parameters(chat_history, system_prompt)
        self._wait_for_request_slot()
        completion = self._get_client().chat.completions.create(**parameters)
        return self._completion_text(completion)

    async def _generate_async(
        self,
        chat_history: ChatHistory,
        system_prompt: str,
    ) -> str:
        parameters = self._request_parameters(chat_history, system_prompt)
        await self._wait_for_async_request_slot()
        client = await self._get_async_client()
        completion = await client.chat.completions.create(**parameters)
        return self._completion_text(completion)

    @contextmanager
    def _record_call(self):
        observer = self.call_observer
        if observer is None:
            yield
            return

        started = time.perf_counter()
        record = {
            "success": False,
            "request_count": 0,
            "retry_count": 0,
            "elapsed_seconds": 0.0,
            "usage": None,
            "error_type": None,
        }
        self._active_call = record

        try:
            yield
            record["success"] = True
        except BaseException as error:
            # 包含异步取消，但不保存异常正文。
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            self._active_call = None

            # 保持现有业务统计回调接受的七个字段不变。
            observer(**record)

    def generate(
        self,
        chat_history: ChatHistory,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        with self._record_call():
            return self._generate(
                chat_history=chat_history,
                system_prompt=system_prompt,
            )

    async def generate_async(
        self,
        chat_history: ChatHistory,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        with self._record_call():
            return await self._generate_async(
                chat_history=chat_history,
                system_prompt=system_prompt,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None