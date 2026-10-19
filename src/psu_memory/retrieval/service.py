"""统一查询侧入口：初筛、一次择优、文本组织及成本汇总。"""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

from src.project_paths import resolve_project_path
from ..models import ContextItem, RetrievalRequest, RetrievalResult, UsageMetrics
from ..pool.difficulty import DifficultyClient as ModelClient
from ..pool.pool_store import SQLitePoolStore
from .augmentation import PROMPT_SEPARATOR, RENDER_RULE_VERSION, render_experience
from .embedding import EmbeddingEncoder
from .embedding_service import EmbeddingService
from .index_store import SQLiteRetrievalIndex
from .prefilter import SemanticPrefilter
from .retrieval_models import RetrievalConfig
from .selector import LLMExperienceSelector
from .vector_cache import SQLiteVectorCache
from ..pool._common import sha256_text
from .text import retrieval_request_fingerprint, validate_request


def _zero_usage() -> dict:
    """一个尚未执行的阶段具有确定的零成本。"""
    return {
        **asdict(UsageMetrics(
            elapsed_seconds=0.0,
            model_call_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            embedding_input_tokens=0,
        )),
        "http_request_count": 0,
        "retry_count": 0,
        "token_usage_complete": True,
    }


def _sum_known(values):
    values = list(values)
    return None if any(value is None for value in values) else sum(values)


class PSURetriever:
    """实现 ExperienceRetriever；每次调用使用独立结果与诊断。"""

    def __init__(
        self,
        prefilter: SemanticPrefilter,
        selector: LLMExperienceSelector,
        *,
        owns_selector: bool = False,
    ) -> None:
        if prefilter.index is not selector.index:
            raise ValueError("初筛与择优必须共用同一个索引实例")
        if prefilter.config != selector.config:
            raise ValueError("初筛与择优配置不一致")

        self.prefilter = prefilter
        self.selector = selector
        self.index = prefilter.index
        self.config = prefilter.config
        self._owns_selector = owns_selector
        self._closed = False
        self._state_lock = threading.Lock()

    @classmethod
    def from_local(
        cls,
        store_path: str | Path,
        *,
        retrieval_config: RetrievalConfig,
        model_config_path: str | Path = "configs/business/query.yaml",
        encoder: EmbeddingEncoder | None = None,
        client_factory: Callable[[], ModelClient] | None = None,
    ) -> PSURetriever:
        """组装本地服务，不初始化表或重建索引。"""
        if not isinstance(retrieval_config, RetrievalConfig):
            raise TypeError("retrieval_config 必须是 RetrievalConfig")

        path = resolve_project_path(store_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        pool = SQLitePoolStore(path)
        cache = SQLiteVectorCache(pool)
        index = SQLiteRetrievalIndex(cache, retrieval_config)
        embeddings = EmbeddingService(
            retrieval_config,
            cache,
            encoder=encoder,
        )
        prefilter = SemanticPrefilter(index, embeddings)

        if client_factory is None:
            config_path = resolve_project_path(model_config_path)

            def client_factory():
                # 模型配置和凭据只在确实需要择优时读取。
                from .selector_llm import LocalSelectorClient

                return LocalSelectorClient.from_yaml(
                    config_path,
                    retrieval_config=retrieval_config,
                )

        selector = LLMExperienceSelector(
            index,
            client_factory=client_factory,
        )
        return cls(
            prefilter,
            selector,
            owns_selector=True,
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        started = perf_counter()
        result = RetrievalResult()
        costs = {
            "prefilter": _zero_usage(),
            "selection": _zero_usage(),
            "rendering": _zero_usage(),
        }
        result.diagnostics = {
            "request_fingerprint": None,
            "stage_seconds": {},
        }
        phase = ""

        def finish(code: str, reason: str) -> RetrievalResult:
            result.reason_code = code
            result.reason = reason

            values = {
                name: _sum_known(
                    cost.get(name, 0) for cost in costs.values()
                )
                for name in (
                    "model_call_count",
                    "prompt_tokens",
                    "completion_tokens",
                    "embedding_input_tokens",
                )
            }
            result.usage = UsageMetrics(
                elapsed_seconds=perf_counter() - started,
                **values,
            )

            prompt_tokens = values["prompt_tokens"]
            completion_tokens = values["completion_tokens"]
            summary = {
                **asdict(result.usage),
                "http_request_count": _sum_known(
                    cost.get("http_request_count", 0)
                    for cost in costs.values()
                ),
                "retry_count": _sum_known(
                    cost.get("retry_count", 0)
                    for cost in costs.values()
                ),
                "total_llm_tokens": (
                    None
                    if prompt_tokens is None or completion_tokens is None
                    else prompt_tokens + completion_tokens
                ),
                "token_usage_complete": (
                    all(
                        cost.get("token_usage_complete") is True
                        for cost in costs.values()
                    )
                    and all(value is not None for value in values.values())
                ),
            }

            result.diagnostics["usage_by_phase"] = deepcopy(costs)
            result.diagnostics["cost_summary"] = summary
            result.diagnostics["stage_seconds"]["retrieve_total_seconds"] = (
                result.usage.elapsed_seconds
            )
            return result

        def run_phase(name: str, invoke):
            nonlocal phase
            phase = name
            phase_started = perf_counter()

            try:
                return invoke()
            except Exception as error:
                result.diagnostics["failed_phase"] = name

                if name == "selection":
                    # 异常未按选择器协议返回，无法推断是否已发出请求。
                    costs[name].update({
                        "model_call_count": None,
                        "http_request_count": None,
                        "retry_count": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "token_usage_complete": False,
                    })
                elif (
                    name == "prefilter"
                    and not isinstance(error, (ValueError, TypeError))
                ):
                    # 初筛不会调用 LLM，但未返回时可能已有编码成本。
                    costs[name].update({
                        "embedding_input_tokens": None,
                        "token_usage_complete": False,
                    })

                costs[name]["elapsed_seconds"] = (
                    perf_counter() - phase_started
                )
                raise
            finally:
                result.diagnostics["stage_seconds"][name + "_seconds"] = (
                    perf_counter() - phase_started
                )

        with self._state_lock:
            closed = self._closed
        if closed:
            return finish("service_closed", "检索服务已经关闭。")

        try:
            # 固定本次请求；字段校验沿用初筛入口。
            request = replace(request)

            prefilter = run_phase(
                "prefilter",
                lambda: self.prefilter.prefilter(request),
            )
            costs["prefilter"].update(asdict(prefilter.usage))
            result.diagnostics["request_fingerprint"] = (
                prefilter.diagnostics.get("request_fingerprint")
            )
            result.diagnostics["prefilter"] = prefilter.to_dict()

            if prefilter.status != "ok":
                return finish(
                    prefilter.status,
                    "语义初筛未提供可用候选，未调用择优模型。",
                )

            selection = run_phase(
                "selection",
                lambda: self.selector.select(request, prefilter),
            )
            costs["selection"].update(asdict(selection.usage))
            result.diagnostics["selection"] = selection.to_dict()

            # 不累计 selection.prefilter_usage：
            # 它是前面同一次初筛的引用，不是新的执行成本。
            if selection.selected is None:
                return finish(selection.reason_code, selection.reason)

            item = selection.selected
            member = item["member"]
            identity = member["identity"]

            text, rendering = run_phase(
                "rendering",
                lambda: render_experience(
                    member["quality"].content,
                    self.config,
                    max_context_chars=request.max_context_chars,
                ),
            )
            costs["rendering"].update(rendering["usage"])
            result.diagnostics["rendering"] = rendering

            if text is None:
                return finish(
                    rendering["reason_code"],
                    "已完成经验选择，本次未生成可用增强文本。",
                )

            result.items = [
                ContextItem(
                    item_id=identity["execution_id"],
                    content=text,
                    source_ref=(
                        "psu-memory:"
                        + identity["execution_id"]
                        + "@"
                        + identity["content_version"]
                    ),
                    version=identity["content_version"],
                    kind="experience",
                    group_id=member["candidate"].group_id,
                    scores={"similarity": item["similarity"]},
                    metadata={
                        "request_fingerprint": (
                            result.diagnostics["request_fingerprint"]
                        ),
                        "snapshot": {
                            "identity": dict(identity),
                            "quality_sha256": member["quality_sha256"],
                            "annotation_sha256": member["annotation_sha256"],
                            "index_key": item["binding"]["index_key"],
                        },
                        "render_rule_version": rendering["render_rule_version"],
                        "template_sha256": rendering["template_sha256"],
                        "content_sha256": rendering["content_sha256"],
                        "render_level": rendering["render_level"],
                    },
                )
            ]
            return finish("retrieved", "已生成一条历史 SQL 方法参考。")

        except Exception as error:
            result.items = []
            result.diagnostics["error_type"] = type(error).__name__

            if phase in ("", "prefilter") and isinstance(
                error, (ValueError, TypeError)
            ):
                return finish("invalid_request", "请求未通过现有输入校验。")

            return finish(
                "retrieval_failed",
                "经验检索处理失败，返回空经验。",
            )
        
    def enhance_prompt(
        self,
        original_prompt: str,
        request: RetrievalRequest,
        *,
        retrieval_result: RetrievalResult | None = None,
    ) -> tuple[str, RetrievalResult]:
        """追加当前有效经验，返回最终提示词和新的结果对象。

        提供 retrieval_result 时只复核和拼接，不重新检索。
        原提示词及传入结果均不改写。
        """
        if not isinstance(original_prompt, str):
            raise TypeError("original_prompt 必须是字符串")
        if (
            retrieval_result is not None
            and not isinstance(retrieval_result, RetrievalResult)
        ):
            raise TypeError("retrieval_result 必须是 RetrievalResult")

        started = perf_counter()
        retrieval_wall_seconds = 0.0
        retrieval_started = False
        result = None
        final_prompt = original_prompt
        augmentation = {
            "applied": False,
            "reused_result": retrieval_result is not None,
        }

        def new_empty(code, reason, fingerprint=None):
            empty = RetrievalResult(
                reason=reason,
                reason_code=code,
                usage=UsageMetrics(
                    elapsed_seconds=0.0,
                    model_call_count=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    embedding_input_tokens=0,
                ),
            )
            empty.diagnostics = {
                "request_fingerprint": fingerprint,
                "stage_seconds": {"retrieve_total_seconds": 0.0},
                "cost_summary": {
                    **_zero_usage(),
                    "total_llm_tokens": 0,
                },
            }
            return empty

        def finish(
            augmentation_code,
            *,
            result_code=None,
            reason=None,
        ):
            nonlocal result

            if result is None:
                result = new_empty(
                    result_code or augmentation_code,
                    reason or "本次未追加经验。",
                )

            if not augmentation["applied"]:
                result.items = []

            if result_code is not None:
                result.reason_code = result_code
            if reason is not None:
                result.reason = reason

            # 自动检索耗时已经在 retrieve() 中记录，不再次累计。
            extra_seconds = max(
                0.0,
                perf_counter() - started - retrieval_wall_seconds,
            )
            augmentation_usage = _zero_usage()
            augmentation_usage["elapsed_seconds"] = extra_seconds

            stage_seconds = result.diagnostics.setdefault(
                "stage_seconds", {}
            )
            base_seconds = stage_seconds.get(
                "retrieve_total_seconds",
                result.usage.elapsed_seconds,
            )
            result.usage = replace(
                result.usage,
                elapsed_seconds=(
                    None if base_seconds is None
                    else base_seconds + extra_seconds
                ),
            )

            augmentation.update({
                "reason_code": augmentation_code,
                "original_prompt_chars": len(original_prompt),
                "final_prompt_chars": len(final_prompt),
                "added_chars": len(final_prompt) - len(original_prompt),
                "original_prompt_sha256": sha256_text(original_prompt),
                "final_prompt_sha256": sha256_text(final_prompt),
                "usage": augmentation_usage,
            })
            result.diagnostics["augmentation"] = augmentation
            result.diagnostics.setdefault(
                "usage_by_phase", {}
            )["augmentation"] = augmentation_usage

            stage_seconds["augmentation_seconds"] = extra_seconds
            stage_seconds["enhance_total_seconds"] = (
                result.usage.elapsed_seconds
            )

            summary = result.diagnostics.setdefault("cost_summary", {})
            summary.update(asdict(result.usage))
            summary.setdefault("http_request_count", None)
            summary.setdefault("retry_count", None)
            summary.setdefault("token_usage_complete", False)
            summary["total_llm_tokens"] = _sum_known([
                result.usage.prompt_tokens,
                result.usage.completion_tokens,
            ])
            return final_prompt, result

        try:
            request = replace(request)

            if retrieval_result is not None:
                # 复用分支不经过初筛入口，因此复用原请求校验函数。
                validate_request(request)
                fingerprint = retrieval_request_fingerprint(request)
                source_fingerprint = retrieval_result.diagnostics.get(
                    "request_fingerprint"
                )

                if source_fingerprint != fingerprint:
                    result = new_empty(
                        "request_mismatch",
                        "检索结果不属于当前请求，未追加经验。",
                        fingerprint,
                    )
                    # 不把另一个请求的成本计入本次请求。
                    result.diagnostics["rejected_result"] = {
                        "request_fingerprint": source_fingerprint,
                        "source_usage": asdict(retrieval_result.usage),
                    }
                    return finish("request_mismatch")

                result = deepcopy(retrieval_result)
            else:
                retrieval_started = True
                retrieval_started_at = perf_counter()
                try:
                    retrieved = self.retrieve(request)
                finally:
                    retrieval_wall_seconds = (
                        perf_counter() - retrieval_started_at
                    )
                result = deepcopy(retrieved)
                fingerprint = result.diagnostics.get("request_fingerprint")

            with self._state_lock:
                closed = self._closed
            if closed:
                return finish(
                    "service_closed",
                    result_code="service_closed",
                    reason="服务已经关闭，未追加经验。",
                )

            if request.limit == 0 or request.max_context_chars == 0:
                return finish(
                    "disabled",
                    result_code="disabled",
                    reason="本次明确停用经验增强。",
                )

            if not result.items:
                # 保留检索阶段的具体原因，如模型拒绝、超时或预算不足。
                return finish("no_context")

            if len(result.items) != 1:
                return finish(
                    "unsupported_item_count",
                    result_code="enhancement_rejected",
                    reason="当前增强入口只接受一条经验。",
                )

            item = result.items[0]
            metadata = item.metadata
            if (
                item.kind != "experience"
                or not isinstance(metadata, dict)
                or not isinstance(metadata.get("snapshot"), dict)
                or metadata.get("request_fingerprint") != fingerprint
            ):
                return finish(
                    "invalid_context_metadata",
                    result_code="enhancement_rejected",
                    reason="经验缺少一致的复核信息，未追加。",
                )

            if (
                not isinstance(item.content, str)
                or metadata.get("content_sha256")
                != sha256_text(item.content)
            ):
                return finish(
                    "content_fingerprint_mismatch",
                    result_code="enhancement_rejected",
                    reason="经验展示文本已经发生变化，未追加。",
                )

            checked, current = self.index.check_snapshots(
                namespace=request.namespace,
                data_version=request.data_version,
                snapshots={"experience": metadata["snapshot"]},
            )
            augmentation["snapshot_check"] = checked

            if checked["invalid"]:
                return finish(
                    "experience_stale",
                    result_code="experience_stale",
                    reason="经验已失效或不再是当前有效成员，未追加。",
                )

            member = current["experience"]["member"]
            identity = member["identity"]
            if (
                item.item_id != identity["execution_id"]
                or item.version != identity["content_version"]
                or item.group_id != member["candidate"].group_id
            ):
                return finish(
                    "context_identity_mismatch",
                    result_code="enhancement_rejected",
                    reason="经验标识与当前记录不一致，未追加。",
                )

            # 使用当前内容和当前预算重渲染，只执行本地文本处理。
            current_text, rendering = render_experience(
                member["quality"].content,
                self.config,
                max_context_chars=request.max_context_chars,
            )
            augmentation["rendering_check"] = rendering

            if current_text is None:
                return finish(
                    rendering["reason_code"],
                    result_code=rendering["reason_code"],
                    reason="当前预算或内容条件不允许追加经验。",
                )

            if (
                metadata.get("render_rule_version") != RENDER_RULE_VERSION
                or metadata.get("template_sha256")
                != rendering["template_sha256"]
                or current_text != item.content
            ):
                return finish(
                    "experience_text_changed",
                    result_code="enhancement_rejected",
                    reason="经验文本与当前内容或模板不一致，未追加。",
                )

            # 不清理原字符串，也不加入择优理由或诊断数据。
            final_prompt = original_prompt + PROMPT_SEPARATOR + item.content
            augmentation["applied"] = True

            return finish(
                "appended",
                result_code="enhanced",
                reason="已在原提示词之后追加一条历史 SQL 方法参考。",
            )

        except Exception as error:
            if result is None:
                result = new_empty(
                    "enhancement_failed",
                    "增强处理失败，保留原提示词。",
                )
                if retrieval_started:
                    # 检索没有按协议返回时，不猜测已发生的执行成本。
                    result.usage = UsageMetrics()
                    result.diagnostics["stage_seconds"][
                        "retrieve_total_seconds"
                    ] = None
                    result.diagnostics["cost_summary"].update({
                        "http_request_count": None,
                        "retry_count": None,
                        "token_usage_complete": False,
                    })

            augmentation["error_type"] = type(error).__name__
            return finish(
                "enhancement_failed",
                result_code="enhancement_failed",
                reason="增强处理失败，保留原提示词。",
            )        

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True

        if self._owns_selector:
            self.selector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()