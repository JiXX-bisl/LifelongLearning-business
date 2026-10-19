"""本地 BGE-M3 dense 编码适配。"""

from __future__ import annotations

from threading import RLock
from time import perf_counter
from typing import Protocol

from ..pool._common import canonical_json
from .retrieval_models import (
    EmbeddingUsage,
    RetrievalConfig,
    validate_dense_vector,
)


EMBEDDING_ADAPTER_VERSION = "bge_m3_dense_v1"
TEXT_BUDGET_POLICY_VERSION = "question_fallback_v1"


class EmbeddingUnavailable(RuntimeError):
    """可由上层记录并降级处理的本地编码失败。"""


class EmbeddingEncoder(Protocol):
    config: RetrievalConfig

    def count_tokens(
        self,
        text: str,
        *,
        usage: EmbeddingUsage,
    ) -> int:
        ...

    def encode(
        self,
        texts: list[str],
        *,
        usage: EmbeddingUsage,
    ) -> list[tuple[float, ...]]:
        ...


def embedding_signature(config: RetrievalConfig) -> str:
    """只包含会影响编码或缓存兼容性的参数。"""

    return canonical_json({
        "adapter_version": EMBEDDING_ADAPTER_VERSION,
        "text_budget_policy_version": TEXT_BUDGET_POLICY_VERSION,
        "embedding_model_name": config.embedding_model_name,
        "embedding_revision": config.embedding_revision,
        "embedding_dimension": config.embedding_dimension,
        "embedding_dtype": config.embedding_dtype,
        "pooling_method": config.pooling_method,
        "normalize_embeddings": config.normalize_embeddings,
        "embedding_max_tokens": config.embedding_max_tokens,
        "device": config.device,
    })


class LocalBgeM3Encoder:
    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._lock = RLock()

    def _load_tokenizer(self, usage: EmbeddingUsage) -> None:
        if self._tokenizer is not None:
            return

        started = perf_counter()
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.model_path),
                local_files_only=True,
                trust_remote_code=False,
            )

            if self.config.embedding_max_tokens > tokenizer.model_max_length:
                raise EmbeddingUnavailable(
                    "configured_token_limit_exceeds_model"
                )

            self._tokenizer = tokenizer

        except EmbeddingUnavailable:
            raise
        except Exception as error:
            raise EmbeddingUnavailable("tokenizer_load_failed") from error
        finally:
            usage.tokenizer_load_seconds += perf_counter() - started

    def _load_model(self, usage: EmbeddingUsage) -> None:
        if self._model is not None:
            return

        try:
            import torch

            if (
                self.config.device == "mps"
                and not torch.backends.mps.is_available()
            ):
                raise EmbeddingUnavailable("mps_unavailable")

        except EmbeddingUnavailable:
            raise
        except Exception as error:
            raise EmbeddingUnavailable("torch_initialization_failed") from error

        started = perf_counter()
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                str(self.config.model_path),
                local_files_only=True,
                trust_remote_code=False,
            )

            if model.config.hidden_size != self.config.embedding_dimension:
                raise EmbeddingUnavailable("model_dimension_mismatch")

            model = model.to(
                device=self.config.device,
                dtype=torch.float32,
            )
            model.eval()

            if self.config.device == "mps":
                torch.mps.synchronize()

            self._torch = torch
            self._model = model
            usage.model_load_count += 1

        except EmbeddingUnavailable:
            raise
        except Exception as error:
            raise EmbeddingUnavailable("model_load_failed") from error
        finally:
            usage.model_load_seconds += perf_counter() - started

    def count_tokens(
        self,
        text: str,
        *,
        usage: EmbeddingUsage,
    ) -> int:
        with self._lock:
            self._load_tokenizer(usage)
            try:
                encoded = self._tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
                return len(encoded["input_ids"])
            except Exception as error:
                raise EmbeddingUnavailable("tokenization_failed") from error

    def encode(
        self,
        texts: list[str],
        *,
        usage: EmbeddingUsage,
    ) -> list[tuple[float, ...]]:
        if not texts:
            return []

        with self._lock:
            self._load_model(usage)
            self._load_tokenizer(usage)
            torch = self._torch
            started = perf_counter()

            try:
                inputs = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )

                lengths = inputs["attention_mask"].sum(dim=1)
                if int(lengths.max().item()) > self.config.embedding_max_tokens:
                    raise EmbeddingUnavailable("unprepared_text_over_limit")

                input_tokens = int(lengths.sum().item())
                inputs = {
                    name: value.to(self.config.device)
                    for name, value in inputs.items()
                }

                # 统计实际发起的编码，不把缓存文本算进去。
                usage.embedding_batch_count += 1
                usage.encoded_text_count += len(texts)
                usage.embedding_input_tokens += input_tokens

                with torch.inference_mode():
                    output = self._model(**inputs)
                    dense = output.last_hidden_state[:, 0]
                    dense = torch.nn.functional.normalize(
                        dense, p=2, dim=1
                    )
                    vectors = dense.detach().to(
                        device="cpu",
                        dtype=torch.float32,
                    ).tolist()

                if self.config.device == "mps":
                    torch.mps.synchronize()

                return [
                    validate_dense_vector(
                        vector,
                        self.config.embedding_dimension,
                    )
                    for vector in vectors
                ]

            except EmbeddingUnavailable:
                raise
            except Exception as error:
                raise EmbeddingUnavailable("encoding_failed") from error
            finally:
                usage.encoding_seconds += perf_counter() - started