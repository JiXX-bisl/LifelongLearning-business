from __future__ import annotations

from abc import ABC, abstractmethod

from src.lifelong_learning.core.models import ChatHistory


class LanguageModel(ABC):
    """Unified interface"""

    @abstractmethod
    def generate(
        self,
        chat_history: ChatHistory,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """Generate an origin model texture according to overall chat history"""
        raise NotImplementedError