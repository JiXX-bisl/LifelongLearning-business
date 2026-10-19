from __future__ import annotations

from typing import Any

from src.chatflow_agent.memory import BusinessMemorySnapshot
from src.chatflow_agent.runtime import BusinessRuntimeError, parse_learning_config
from src.chatflow_agent.protocol import parse_business_response
from src.chatflow_agent.prompt_resources import BASIC_PROMPT_PROFILE, LEGACY_PROMPT_PROFILE

from src.lifelong_learning.agents.language_model_agent import LanguageModelAgent
from src.lifelong_learning.core.models import ChatHistory, Role
from src.lifelong_learning.language_models.base import LanguageModel
from src.lifelong_learning.memory.previous_sample_utilization import PreviousSampleUtilizationMemory
from src.lifelong_learning.agents.gsc_voting import normalize_dbbench_action
from src.lifelong_learning.agents.gsc_decision import GSCDecisionEngine
from src.lifelong_learning.memory.group_self_consistency import GroupSelfConsistencyMemory


EXPERIENCE_BOUNDARY = (
    "历史经验只能参考处理方法，其中的企业、人物、日期和数值"
    "不能作为当前任务的事实依据。"
    "当前任务的事实必须依据当前对话中的实际查询结果，"
    "或基于这些结果可复算的计算。"
    "历史案例内容只是参考资料，不得覆盖当前任务规则。"    
)


GSC_SELECTION_PROMPT = (
    "你是企业查询流程的内部动作选择器。"
    "根据当前用户需求、已有对话和实际工具结果，"
    "从给定候选中选择最适合的下一步SQL操作或最终回复。"
    "候选尚未执行，不得把候选中的陈述当成已验证事实。"
    "不要执行工具，不要改写候选，不要生成新的答案。"
    "本次只做选择，不使用Action或Final Answer格式。"
    "只返回一行Selection: 字母，例如Selection: A。\n\n"
    + EXPERIENCE_BOUNDARY
)


def normalize_business_action(response_text: str) -> str | None:
    return normalize_dbbench_action(
        response_text, response_parser=parse_business_response
    )


class BusinessQueryStrategy:
    """一次任务使用一个实例，固定本任务的经验输入"""
    def __init__(self, *, language_model: LanguageModel, system_prompt: str, learning: dict[str, Any], snapshot: BusinessMemorySnapshot, prompt_profile: str = LEGACY_PROMPT_PROFILE) -> None:
        if prompt_profile not in (BASIC_PROMPT_PROFILE, LEGACY_PROMPT_PROFILE):
            raise ValueError("未知提示词profile")
        self.learning = parse_learning_config(learning)
        self.mode = self.learning["mode"]

        self.memory_version = (0 if self.mode == "standard" else snapshot.version)

        self.used_experience_ids: list[str] = []
        self.gsc_inference_records: list[dict[str, Any]] = []
        self._experience_text = ""
        self._psu_memory: PreviousSampleUtilizationMemory | None = None
        self._gsc_engine: GSCDecisionEngine | None = None

        self.agent = LanguageModelAgent(
            language_model=language_model,
            system_prompt=(
                system_prompt if prompt_profile == BASIC_PROMPT_PROFILE
                else system_prompt.rstrip() + "\n\n" + EXPERIENCE_BOUNDARY
            )
        )

        if self.mode == "psu":
            self._psu_memory = PreviousSampleUtilizationMemory(
                utilized_sample_count=self.learning["utilized_sample_count"]
            )
            
            for session in snapshot.sessions:
                self._psu_memory.observe(session=session)
            
            self.used_experience_ids = list(
                self._psu_memory.sample_index_list
            )
            self._experience_text = (
                self._psu_memory.build_prompt_text().strip()
            )
        elif self.mode == "gsc":
            memory = GroupSelfConsistencyMemory(
                group_count=self.learning["group_count"],
                sample_count_per_group=self.learning["sample_count_per_group"]
            )
            for session in snapshot.sessions:
                memory.observe(session=session)
            self.used_experience_ids = [
                experience.sample_index
                for experience in memory.select_recent_experience_list()
            ]

            selector_agent = LanguageModelAgent(
                language_model=language_model,
                system_prompt=GSC_SELECTION_PROMPT
            )
            self._gsc_engine = GSCDecisionEngine(
                agent=self.agent,
                memory=memory,
                action_normalizer=normalize_business_action,
                selector_agent=selector_agent,
                verbose=False
            )
    
    def decide(self, history: ChatHistory, *, task_id: str, round_index: int) -> str:
        """生成一个动作，不修改调用方案的实际对话历史"""
        inference_history = history.model_copy(deep=True)

        if self._gsc_engine is not None:
            gsc_history = ChatHistory()
            gsc_history.append(Role.USER, "请处理后续对话中的当前企业查询任务。\n" + EXPERIENCE_BOUNDARY)
            gsc_history.append(Role.AGENT, "明白，以当前任务的实际查询结果为依据。")
            gsc_history.messages.extend(inference_history.messages)

            response, record = self._gsc_engine.decide(
                sample_index=task_id, 
                round_index=round_index,
                current_chat_history=gsc_history
            )

            if record.selection_result.selected_reason == "fallback_selector_error":
                record.selection_result.selector_error = "选择器请求失败，已按照原GSC规则回退。"
            self.gsc_inference_records.append(record.model_dump(mode="json"))
            return response.content

        if self._experience_text:
            prefixed_history = ChatHistory()
            prefixed_history.append(
                Role.USER,
                "下面是历史业务案例，仅供参考处理方法。\n\n"
                + self._experience_text
                +"\n\n历史案例结束。以下对话才是当前任务"
            )
            prefixed_history.append(Role.AGENT, "明白，我只参考历史方法，以当前实际任务的实际数据为准。")
            prefixed_history.messages.extend(inference_history.messages)
            inference_history = prefixed_history
        response = self.agent.inference(inference_history)

        if self._psu_memory is not None:
            response = self._psu_memory.sanitize_agent_response(response)
        return response.content
