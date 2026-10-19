"""通用查询适配：上游提示词形成之后、业务 LLM 调用之前接入 PSU。"""
from __future__ import annotations

import json
from time import perf_counter

from src.chatflow_agent.query import QueryService
from ..models import RetrievalRequest
from ..pool._common import require_nonnegative_int, require_text
from .common import model_options, query_metadata
from .local_query import _PromptStrategy


class PSUQueryService(QueryService):
    """复用原查询流程，集中管理每个任务的上游输入。"""

    def __init__(
        self,
        config_path,
        *,
        namespace,
        data_version,
        retriever=None,
        experiment_id=None,
        retrieval_limit=1,
        max_context_chars=None,
    ):
        require_text(namespace, "namespace")
        require_text(data_version, "data_version")
        require_nonnegative_int(retrieval_limit, "retrieval_limit")

        if retrieval_limit not in (0, 1):
            raise ValueError("retrieval_limit 只支持 0 或 1")

        if max_context_chars is not None:
            require_nonnegative_int(max_context_chars, "max_context_chars")

        self.namespace = namespace
        self.data_version = data_version
        self.retriever = retriever
        self.retrieval_limit = retrieval_limit
        self.max_context_chars = max_context_chars

        self._upstream_prompt = None
        self._data_advice = ""
        self._query_active = False

        # 新组件独立接入，不叠加旧 PSU/GSC 策略。
        super().__init__(
            config_path,
            mode="standard",
            experiment_id=experiment_id,
        )

        try:
            if self.config["data"]["baseline_id"] != data_version:
                raise ValueError("业务基线与经验池 data_version 不同")

            self.effective_model_options = model_options(
                self.language_model,
                config_path,
            )
        except Exception:
            # 只关闭业务资源，注入的 retriever 由 runtime 管理。
            self.close()
            raise

    def query(
        self,
        question,
        *,
        on_progress=None,
        upstream_prompt=None,
        data_advice="",
    ):
        """一个任务固定一次上游输入，任务结束后清除临时输入。"""
        require_text(question, "question")

        if upstream_prompt is not None:
            require_text(upstream_prompt, "upstream_prompt")

        if not isinstance(data_advice, str):
            raise ValueError("data_advice 必须是字符串")

        if self._query_active:
            raise RuntimeError("同一查询服务仅支持同步顺序调用")

        self._query_active = True
        self._upstream_prompt = upstream_prompt
        self._data_advice = data_advice

        try:
            return super().query(
                question,
                on_progress=on_progress,
            )
        finally:
            self._upstream_prompt = None
            self._data_advice = ""
            self._query_active = False

    def _upstream_inputs(self, strategy, conversation_context):
        """集中映射上游提示词、任务前上下文和建议。"""
        original_prompt = (
            strategy.agent.system_prompt
            if self._upstream_prompt is None
            else self._upstream_prompt
        )

        context_text = json.dumps(
            conversation_context.model_dump(mode="json")["messages"],
            ensure_ascii=False,
        )

        return original_prompt, context_text, self._data_advice

    def _make_strategy(self, question, conversation_context, task_id):
        strategy = super()._make_strategy(
            question,
            conversation_context,
            task_id,
        )

        original_prompt, context_text, data_advice = self._upstream_inputs(
            strategy,
            conversation_context,
        )

        # 上游原提示词保持原文，包括空白和换行。
        strategy.agent.system_prompt = original_prompt

        if self.retriever is None:
            return strategy

        request = RetrievalRequest(
            task_id=task_id,
            question=question,
            namespace=self.namespace,
            data_version=self.data_version,
            context_text=context_text,
            data_advice=data_advice,
            limit=self.retrieval_limit,
            max_context_chars=self.max_context_chars,
        )

        # 关键接入 B：沿用已有的一次检索、后续复核和提示词交付。
        return _PromptStrategy(
            strategy,
            self.retriever,
            request,
        )

    def _execute_operation(self, sql, scope=None):
        started = perf_counter()
        result = super()._execute_operation(sql, scope)
        result["elapsed_seconds"] = perf_counter() - started
        return result

    def _task_metadata(self, strategy, result):
        return query_metadata(strategy, result)