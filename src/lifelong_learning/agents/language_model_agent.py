from __future__ import annotations

from src.lifelong_learning.agents.base import Agent
from src.lifelong_learning.core.models import (
    ChatHistory,
    ChatMessage,
    Role,
)
from src.lifelong_learning.language_models.base import (
    LanguageModel,
)


class LanguageModelAgent(Agent):
    """Unified Agent"""

    def __init__(
        self,
        language_model: LanguageModel,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.language_model = language_model
        self.system_prompt = system_prompt

    def inference(
        self,
        chat_history: ChatHistory,
    ) -> ChatMessage:
        """Package origin texture to Agent message by LLM"""
        response_content = self.language_model.generate(
            chat_history=chat_history,
            system_prompt=self.system_prompt,
        )

        return ChatMessage(
            role=Role.AGENT,
            content=response_content,
        )