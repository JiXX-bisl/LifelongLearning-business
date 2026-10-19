from __future__ import annotations

from abc import ABC, abstractmethod

from src.lifelong_learning.core.models import (
    ChatHistory, 
    ChatMessage
)


class Agent(ABC):
    """All agent must obey unified interface"""
    @abstractmethod
    def inference(self, chat_history: ChatHistory) -> ChatMessage:
        raise NotImplementedError