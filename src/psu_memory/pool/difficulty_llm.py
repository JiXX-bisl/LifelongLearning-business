"""保留原难度客户端入口，实际实现复用共享客户端。"""

from __future__ import annotations

from ..model_client import LocalModelClient, usage_from_record


class LocalDifficultyClient(LocalModelClient):
    """旧接口、原参数和原缓存签名继续有效。"""

    ADAPTER_VERSION = "local_difficulty_client_v1"


__all__ = [
    "LocalDifficultyClient",
    "usage_from_record",
]