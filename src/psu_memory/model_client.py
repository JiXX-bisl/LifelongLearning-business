"""PSU 共用模型适配：配置、签名、回复和成本统计。"""

from __future__ import annotations

import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

from .pool._common import (
    canonical_json,
    nonnegative_int_or_none,
    require_text,
    sha256_text,
)
from .pool.pool_models import DifficultyUsage, ModelReply


def read_model_config(path: str | Path) -> dict:
    """只读取 model 配置，不初始化数据库等业务组件。"""
    import yaml

    document = yaml.safe_load(
        Path(path).read_text(encoding="utf-8")
    )
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("model"), dict)
    ):
        raise ValueError("配置文件缺少 model 对象")

    return document["model"]


def usage_from_record(
    record: dict,
    *,
    elapsed_seconds: float,
) -> DifficultyUsage:
    """沿用原难度客户端的成本转换规则。"""
    request_count = nonnegative_int_or_none(
        record.get("request_count")
    )
    retry_count = nonnegative_int_or_none(
        record.get("retry_count")
    )

    raw_usage = record.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}

    if request_count == 0:
        prompt_tokens = 0
        completion_tokens = 0
    else:
        prompt_tokens = nonnegative_int_or_none(
            usage.get("prompt_tokens")
        )
        completion_tokens = nonnegative_int_or_none(
            usage.get("completion_tokens")
        )

    complete = (
        request_count == 0
        or (
            request_count == 1
            and prompt_tokens is not None
            and completion_tokens is not None
        )
    )

    return DifficultyUsage(
        elapsed_seconds=elapsed_seconds,
        model_call_count=1,
        http_request_count=request_count,
        retry_count=retry_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        token_usage_complete=complete,
    )


class LocalModelClient:
    """共用配置和统计；各功能通过独立参数控制调用行为。"""

    ADAPTER_VERSION = "local_model_client_v1"

    def __init__(
        self,
        model_config: dict,
        *,
        model_name: str | None = None,
        generation_options: dict | None = None,
    ) -> None:
        if not isinstance(model_config, dict):
            raise ValueError("model 配置必须是对象")
        if model_config.get("type") != "openai_compatible":
            raise ValueError("当前适配器只支持 openai_compatible")

        resolved_name = model_name or model_config.get("model_name")
        require_text(resolved_name, "model_name")

        key_env = model_config.get("api_key_env", "OPENAI_API_KEY")
        base_env = model_config.get("base_url_env", "OPENAI_BASE_URL")
        require_text(key_env, "api_key_env")
        require_text(base_env, "base_url_env")

        api_key = os.environ.get(key_env)
        if not api_key:
            raise ValueError(f"请先配置模型环境变量：{key_env}")

        base_url = (
            os.environ.get(base_env)
            or os.environ.get("OPENAI_BASE_URL")
        )

        options = deepcopy(generation_options or {})
        settings = {
            "temperature": model_config.get("temperature"),
            "max_tokens": model_config.get("max_tokens"),
            "request_interval_seconds": model_config.get(
                "request_interval_seconds"
            ),
        }
        settings.update(options)

        # 延迟导入，与原难度适配器保持一致。
        from src.lifelong_learning.core.models import ChatHistory, Role
        from src.lifelong_learning.language_models.openai_compatible import (
            OpenAICompatibleLanguageModel,
        )

        self._history_type = ChatHistory
        self._user_role = Role.USER
        self._model = OpenAICompatibleLanguageModel(
            model_name=resolved_name,
            api_key=api_key,
            base_url=base_url,
            **settings,
        )
        self._lock = threading.Lock()

        signature = {
            "adapter_version": self.ADAPTER_VERSION,
            "model_name": resolved_name,
            "endpoint_sha256": sha256_text(base_url or "SDK_DEFAULT"),
            "temperature": settings["temperature"],
            "max_tokens": settings["max_tokens"],
            "request_interval_seconds": settings["request_interval_seconds"],
        }
        if options:
            signature["generation_options"] = options

        self.signature = canonical_json(signature)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        model_name: str | None = None,
        **kwargs,
    ):
        return cls(
            read_model_config(path),
            model_name=model_name,
            **kwargs,
        )

    def _make_history(self, user_text: str):
        history = self._history_type()
        history.append(self._user_role, user_text)
        return history

    def _invoke_recorded(
        self,
        invoke: Callable[[], str],
        *,
        started: float,
        queue_wait_seconds: float = 0.0,
    ) -> ModelReply:
        record = {
            "request_count": 0,
            "retry_count": 0,
            "usage": None,
        }

        def observe(**values) -> None:
            record.update(values)

        previous_observer = self._model.call_observer
        self._model.call_observer = observe

        call_started = time.perf_counter()
        response_text = ""
        error_type = None

        try:
            response_text = invoke()
        except Exception as error:
            error_type = type(error).__name__
        finally:
            self._model.call_observer = previous_observer

        call_seconds = time.perf_counter() - call_started
        elapsed = time.perf_counter() - started

        return ModelReply(
            text=response_text,
            error_type=error_type,
            usage=usage_from_record(
                record,
                elapsed_seconds=elapsed,
            ),
            diagnostics={
                "model_signature": self.signature,
                "transport_error_type": record.get("error_type"),
                "stage_seconds": {
                    "queue_wait": queue_wait_seconds,
                    "client_call": call_seconds,
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
        with self._lock:
            started = time.perf_counter()
            return self._invoke_recorded(
                lambda: self._model.generate(
                    chat_history=self._make_history(user_text),
                    system_prompt=system_prompt,
                ),
                started=started,
            )

    def close(self) -> None:
        with self._lock:
            self._model.close()