"""一次 LLM 择优协调；复用已有提示词、索引检查和成本结构。"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace
from time import perf_counter
from typing import Callable

from ..models import RetrievalRequest
from ..pool._common import require_text
from ..pool.difficulty import DifficultyClient as ModelClient
from ..pool.pool_models import DifficultyUsage, ModelReply
from .index_store import SQLiteRetrievalIndex
from .retrieval_models import (
    PrefilterResult,
    SelectorPrompt,
    SelectorResult,
)
from .selector_prompt import (
    parse_selection_output,
    prepare_selector_prompt,
)


class LLMExperienceSelector:
    """消费同一请求的初筛结果，最多返回一条当前有效经验。"""

    def __init__(
        self,
        index: SQLiteRetrievalIndex,
        client: ModelClient | None = None,
        *,
        client_factory: Callable[[], ModelClient] | None = None,
    ) -> None:
        if (client is None) == (client_factory is None):
            raise ValueError("必须且只能提供 client 或 client_factory")

        self.index = index
        self.config = index.config
        self._client = client
        self._client_factory = client_factory
        self._owns_client = client is None
        self._client_lock = threading.Lock()
        self._closed = False

    def _get_client(self):
        """首次真正需要择优时创建客户端，后续持续复用。"""
        with self._client_lock:
            if self._closed:
                raise RuntimeError("SelectorClosed")

            client = self._client
            created = client is None
            if created:
                client = self._client_factory()

            try:
                require_text(client.signature, "client.signature")
                client_config = getattr(client, "config", self.config)
                if client_config != self.config:
                    raise ValueError("择优客户端与协调器配置不一致")
            except Exception:
                if created:
                    close = getattr(client, "close", None)
                    if callable(close):
                        close()
                raise

            self._client = client
            return client

    def _recheck(
        self,
        request: RetrievalRequest,
        prepared: SelectorPrompt,
        aliases: tuple[str, ...],
    ) -> dict:
        checked, _ = self.index.check_snapshots(
            namespace=request.namespace,
            data_version=request.data_version,
            snapshots={
                alias: prepared.alias_map[alias]
                for alias in aliases
            },
        )
        return checked

    def select(
        self,
        request: RetrievalRequest,
        prefilter_result: PrefilterResult,
    ) -> SelectorResult:
        started = perf_counter()
        result = SelectorResult(
            reason_code="not_started",
            reason="",
            prefilter_usage=prefilter_result.usage,
            diagnostics={"sent_candidate_count": 0},
        )

        def finish(code: str, reason: str) -> SelectorResult:
            result.reason_code = code
            result.reason = reason
            result.stage_seconds["selector_total_seconds"] = (
                perf_counter() - started
            )
            return result

        if self._closed:
            return finish("selector_closed", "选择协调器已经关闭。")

        stage = perf_counter()
        try:
            # 当前请求字段都是标量；固定本次请求，不受外部后续赋值影响。
            request = replace(request)
            prepared = prepare_selector_prompt(
                request,
                prefilter_result,
                self.config,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            result.diagnostics["error_type"] = type(error).__name__
            return finish("invalid_input", "选择输入无法准备。")
        finally:
            result.stage_seconds["prompt_preparation_seconds"] = (
                perf_counter() - stage
            )

        result.request_fingerprint = prepared.request_fingerprint
        result.prompt_version = prepared.prompt_version
        result.prompt_sha256 = prepared.prompt_sha256
        result.diagnostics["prepared_prompt"] = prepared.to_dict()

        if not prepared.ready:
            return finish(
                prepared.reason_code,
                "当前输入不满足择优条件，未调用模型。",
            )

        def recheck(phase: str, aliases: tuple[str, ...]):
            check_started = perf_counter()
            try:
                checked = self._recheck(request, prepared, aliases)
                result.diagnostics[phase] = checked
                return checked
            except (
                sqlite3.Error,
                OSError,
                ValueError,
                TypeError,
                KeyError,
            ) as error:
                result.diagnostics[phase] = {
                    "error_type": type(error).__name__,
                }
                return None
            finally:
                result.stage_seconds[f"{phase}_seconds"] = (
                    perf_counter() - check_started
                )

        before = recheck("before_call", tuple(prepared.alias_map))
        if before is None:
            return finish(
                "recheck_failed_before_call",
                "调用前复核失败，未调用模型。",
            )
        if before["invalid"]:
            return finish(
                "candidates_changed_before_call",
                "待发送候选已经发生变化，未调用模型。",
            )

        stage = perf_counter()
        try:
            client = self._get_client()
            result.diagnostics["model_signature"] = client.signature
        except Exception as error:
            result.diagnostics["error_type"] = type(error).__name__
            return finish(
                "model_unavailable",
                "择优客户端无法创建，未调用模型。",
            )
        finally:
            result.stage_seconds["client_preparation_seconds"] = (
                perf_counter() - stage
            )

        stage = perf_counter()
        try:
            reply = client.generate(
                system_prompt=prepared.system_prompt,
                user_text=prepared.user_text,
            )
            if not isinstance(reply, ModelReply):
                raise TypeError("客户端必须返回 ModelReply")
        except Exception as error:
            # 非标准客户端异常：无法确定 HTTP 次数及服务端用量。
            result.usage = DifficultyUsage(
                elapsed_seconds=perf_counter() - stage,
                model_call_count=1,
                http_request_count=None,
                retry_count=None,
                prompt_tokens=None,
                completion_tokens=None,
                token_usage_complete=False,
            )
            result.diagnostics["error_type"] = type(error).__name__
            result.diagnostics["sent_candidate_count"] = None
            return finish("model_failed", "择优调用失败，返回空结果。")
        finally:
            result.stage_seconds["llm_adapter_seconds"] = (
                perf_counter() - stage
            )

        result.usage = reply.usage
        result.diagnostics["client"] = reply.diagnostics

        request_count = reply.usage.http_request_count
        result.diagnostics["sent_candidate_count"] = (
            None if request_count is None
            else len(prepared.alias_map) if request_count > 0
            else 0
        )

        if reply.error_type:
            result.diagnostics["error_type"] = reply.error_type
            code = (
                "model_timeout"
                if reply.error_type in {
                    "SelectorDeadlineExceeded",
                    "APITimeoutError",
                }
                else "model_failed"
            )
            return finish(code, "择优调用未成功，返回空结果。")

        stage = perf_counter()
        try:
            selected_id, reason = parse_selection_output(
                reply.text,
                prepared,
                self.config,
            )
        except (ValueError, TypeError) as error:
            result.diagnostics["error_type"] = type(error).__name__
            return finish(
                "invalid_model_output",
                "模型输出不符合选择协议，返回空结果。",
            )
        finally:
            result.stage_seconds["output_parsing_seconds"] = (
                perf_counter() - stage
            )

        result.diagnostics["model_selected_id"] = selected_id
        result.diagnostics["model_reason"] = reason

        if selected_id is None:
            result.diagnostics["after_call"] = {
                "skipped": "no_selected_candidate",
            }
            return finish("no_selection", reason)

        after = recheck("after_call", (selected_id,))
        if after is None:
            return finish(
                "recheck_failed_after_call",
                "调用后复核失败，未返回经验。",
            )
        if after["invalid"]:
            return finish(
                "selected_changed_after_call",
                "模型所选版本已经失效，未返回经验。",
            )

        # 只返回实际发送并经模型选中的原候选，不插入其他候选。
        result.selected_id = selected_id
        result.selected = prepared.alias_map[selected_id]["candidate"]
        return finish("selected", reason)

    def close(self) -> None:
        with self._client_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            self._client = None

        # 注入的共享客户端由调用方管理；工厂创建的客户端由本组件关闭。
        if self._owns_client and client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()