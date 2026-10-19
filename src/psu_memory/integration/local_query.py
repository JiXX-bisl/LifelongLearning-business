"""本地业务接入：原查询循环不变，PSU只增强实际业务提示词。"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone

from src.chatflow_agent.query import QueryService
from src.lifelong_learning.core.models import ChatHistory
from ..models import RetrievalRequest
from .common import add_costs, model_options, zero_cost


class _PromptStrategy:
    def __init__(self, delegate, retriever, request):
        self.delegate = delegate
        self.retriever = retriever
        self.request = request
        self.base_prompt = delegate.agent.system_prompt
        self.result = None
        self.initial_result = None
        self.rechecks = []
        self.prompt_events = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def decide(self, history, *, task_id, round_index):
        # 首轮检索一次；后续推理复用结果，并再次检查停用与版本。
        prompt, result = self.retriever.enhance_prompt(
            self.base_prompt, self.request, retrieval_result=self.result,
        )
        if self.initial_result is None:
            self.initial_result = deepcopy(result)
        else:
            self.rechecks.append(deepcopy(result.diagnostics["augmentation"]["usage"]))
        self.result = result
        self.delegate.agent.system_prompt = prompt
        self.prompt_events.append({
            "round": round_index, "reason_code": result.reason_code,
            "augmentation": deepcopy(result.diagnostics.get("augmentation", {})),
        })
        return self.delegate.decide(history, task_id=task_id, round_index=round_index)

    def metadata(self):
        if self.initial_result is None:
            return {"usage": zero_cost(), "used_experience_ids": [], "prompt_events": []}
        summary = self.initial_result.diagnostics.get("cost_summary", {})
        first_cost = {**asdict(self.initial_result.usage),
                      "http_request_count": summary.get("http_request_count"),
                      "retry_count": summary.get("retry_count"),
                      "token_usage_complete": summary.get("token_usage_complete", False)}
        return {
            "usage": add_costs([first_cost, *self.rechecks]),
            "request": asdict(self.request),
            "used_experience_ids": sorted({
                item.item_id for item in self.initial_result.items
            }),
            "base_prompt": self.base_prompt,
            "final_prompt": self.delegate.agent.system_prompt,
            "initial_retrieval": self.initial_result.to_dict(),
            "last_retrieval": self.result.to_dict(),
            "prompt_events": self.prompt_events,
        }


class LocalValidationQueryService(QueryService):
    """复用QueryService，统一三种条件的SQL查询执行。"""
    def __init__(self, config_path, *, experiment_id, condition, namespace,
                 data_version, retriever=None):
        self.condition = condition
        self.namespace = namespace
        self.data_version = data_version
        self.retriever = retriever
        self.case_id = None
        self.run_id = None
        super().__init__(config_path, mode="standard", experiment_id=experiment_id)
        if self.config["data"]["baseline_id"] != data_version:
            self.close()
            raise ValueError("查询基线与经验池数据范围不同")
        try:
            self.effective_model_options = model_options(self.language_model, config_path)
        except Exception:
            self.close()
            raise

    def _make_strategy(self, question, conversation_context, task_id):
        strategy = super()._make_strategy(question, conversation_context, task_id)
        if self.retriever is None:
            return strategy
        request = RetrievalRequest(
            task_id=task_id, question=question, namespace=self.namespace,
            data_version=self.data_version,
            context_text=json.dumps(conversation_context.model_dump(mode="json")["messages"],
                                    ensure_ascii=False),
        )
        return _PromptStrategy(strategy, self.retriever, request)

    def _task_metadata(self, strategy, result):
        retrieval = getattr(strategy, "initial_result", None)
        return {
            "used_experience_ids": sorted({item.item_id for item in retrieval.items})
            if retrieval is not None else [],
            "validation_case_id": self.case_id,
            "validation_run_id": self.run_id,
            "validation_condition": self.condition,
        }

    def _save_result(self, result):
        """验证入口仅补内存身份；由runner统一保存单题记录。"""
        turn_index = self._turn_index + 1
        result.update({
            "session_id": self.session_id,
            "turn_index": turn_index,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_id": self.config["data"]["baseline_id"],
            "reference_date": self.data_context["reference_date"],
            "schema_version": self.data_context.get("schema_version", "2.1"),
        })
        self._turn_index = turn_index

    def restore_conversation(self, record):
        """恢复上一轮已完成的真实会话，不重新询问或重新执行SQL。"""
        messages = record["messages"]
        if "context_message_count" in record:
            boundary = record["context_message_count"]
            if (type(boundary) is not int or not 0 <= boundary < len(messages)
                    or messages[boundary].get("role") != "user"
                    or messages[boundary].get("content") != record["question"]):
                raise ValueError("无法恢复当前问题的上下文边界")
        else:
            positions = [i for i, message in enumerate(messages)
                         if message["role"] == "user" and message["content"] == record["question"]]
            if len(positions) != 1:
                raise ValueError("无法恢复当前问题的上下文边界")
            boundary = positions[0]
        self._history = ChatHistory.model_validate({"messages": messages[:boundary]})
        self.session_id = record["session_id"]
        self._turn_index = record["turn_index"]
        if record["status"] == "completed":
            self._remember_result(record)
