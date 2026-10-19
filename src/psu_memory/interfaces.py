from __future__ import annotations

from typing import Protocol

from .models import (
    AdmissionResult,
    CompletedTask,
    RetrievalRequest,
    RetrievalResult,
)
from .retrieval.retrieval_models import PrefilterResult, SelectorResult


class ExperienceRetriever(Protocol):
    """查询前的经验获取接口。"""

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        ...


class MemoryComponent(ExperienceRetriever, Protocol):
    """完整经验组件：查询前检索与查询后任务回传。"""

    def observe(
        self,
        task: CompletedTask,
    ) -> AdmissionResult:
        ...


class ExperienceSelector(Protocol):
    """消费同一请求的初筛结果，最多选择一条经验。"""

    def select(
        self,
        request: RetrievalRequest,
        prefilter_result: PrefilterResult,
    ) -> SelectorResult:
        ...