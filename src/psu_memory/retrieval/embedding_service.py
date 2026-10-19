"""匹配文本预算、缓存与本地编码的协调入口。
    1. 检查完整匹配文本长度；
    2. 超限时退回问题本身；
    3. 查找缓存；
    4. 合并同批重复文本；
    5. 只编码未命中项；
    6. 原子保存并返回成本。
"""
from __future__ import annotations

from dataclasses import replace
from threading import RLock
from time import perf_counter

from ..pool._common import normalize_question_text, require_text
from .embedding import (
    EmbeddingEncoder,
    EmbeddingUnavailable,
    LocalBgeM3Encoder,
    embedding_signature,
)
from .retrieval_models import (
    EmbeddingBatch,
    EmbeddingItem,
    EmbeddingUsage,
    MatchingText,
    PreparedEmbedding,
    RetrievalConfig,
)
from .text import matching_text_sha256
from .vector_cache import SQLiteVectorCache, vector_cache_key


class EmbeddingService:
    def __init__(
        self,
        config: RetrievalConfig,
        cache: SQLiteVectorCache,
        *,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.encoder = encoder if encoder is not None else LocalBgeM3Encoder(config)

        if self.encoder.config != config:
            raise ValueError("编码器与服务配置不一致")

        self.signature_json = embedding_signature(config)
        self._lock = RLock()

    def _prepare(
        self,
        matching: MatchingText,
        usage: EmbeddingUsage,
        *,
        question_only: bool,
    ) -> PreparedEmbedding:
        if not matching.ready:
            return PreparedEmbedding(matching, None, None, "unavailable")

        original_count = None

        if not question_only:
            original_count = self.encoder.count_tokens(
                matching.text, usage=usage
            )
            if original_count <= self.config.embedding_max_tokens:
                mode = (
                    "with_user_context"
                    if matching.context_questions
                    else "question_only"
                )
                return PreparedEmbedding(
                    matching, original_count, original_count, mode
                )

        question = normalize_question_text(matching.question)
        if matching.text == question and original_count is not None:
            question_count = original_count
        else:
            question_count = self.encoder.count_tokens(
                question, usage=usage
            )

        if question_count > self.config.embedding_max_tokens:
            unavailable = replace(
                matching,
                text=None,
                input_sha256=None,
                reason_code="question_too_long",
            )
            return PreparedEmbedding(
                unavailable, question_count, original_count, "unavailable"
            )

        reason = (
            "question_only_requested"
            if question_only
            else "context_overflow_question_only"
        )
        effective = replace(
            matching,
            question=question,
            context_questions=(),
            text=question,
            input_sha256=matching_text_sha256(
                question,
                rule_version=matching.rule_version,
            ),
            reason_code=reason,
        )
        return PreparedEmbedding(
            effective, question_count, original_count, "question_only"
        )

    def encode_many(
        self,
        texts: list[MatchingText],
        *,
        namespace: str,
        data_version: str,
        question_only: bool = False,
    ) -> EmbeddingBatch:
        require_text(namespace, "namespace")
        require_text(data_version, "data_version")
        if type(question_only) is not bool:
            raise ValueError("question_only 必须是布尔值")
        if any(not isinstance(text, MatchingText) for text in texts):
            raise TypeError("texts 中的元素必须是 MatchingText")

        with self._lock:
            started = perf_counter()
            usage = EmbeddingUsage(
                elapsed_seconds=0.0,
                model_call_count=0,
                prompt_tokens=0,
                completion_tokens=0,
                embedding_input_tokens=0,
            )

            def finish(items: list[EmbeddingItem]) -> EmbeddingBatch:
                usage.unavailable_count = sum(not item.ready for item in items)
                usage.elapsed_seconds = perf_counter() - started
                return EmbeddingBatch(items, usage)

            try:
                prepared = [
                    self._prepare(
                        text, usage, question_only=question_only
                    )
                    for text in texts
                ]
            except EmbeddingUnavailable as error:
                return finish([
                    EmbeddingItem(
                        PreparedEmbedding(text, None, None, "unavailable"),
                        error=str(error),
                    )
                    for text in texts
                ])

            items = [EmbeddingItem(value) for value in prepared]
            unique = {}

            for item in items:
                if not item.prepared.ready:
                    continue

                item.cache_key = vector_cache_key(
                    namespace,
                    data_version,
                    item.prepared.matching.input_sha256,
                    self.signature_json,
                )
                unique.setdefault(item.cache_key, item)

            if not unique:
                return finish(items)

            cache_started = perf_counter()
            cached, corrupted = self.cache.get_many(
                list(unique),
                signature_json=self.signature_json,
            )
            usage.cache_read_seconds += perf_counter() - cache_started
            usage.corrupted_cache_count = len(corrupted)

            for item in items:
                if item.cache_key in cached:
                    item.vector = cached[item.cache_key]
                    item.cache_hit = True

            missing = [
                item for key, item in unique.items()
                if key not in cached
            ]

            usage.cache_hit_count = sum(item.cache_hit for item in items)
            usage.cache_miss_count = sum(
                item.prepared.ready and not item.cache_hit
                for item in items
            )
            usage.deduplicated_count = usage.cache_miss_count - len(missing)

            if not missing:
                return finish(items)

            try:
                batch_size = self.config.embedding_batch_size
                for offset in range(0, len(missing), batch_size):
                    chunk = missing[offset:offset + batch_size]
                    vectors = self.encoder.encode(
                        [item.prepared.matching.text for item in chunk],
                        usage=usage,
                    )
                    if len(vectors) != len(chunk):
                        raise EmbeddingUnavailable(
                            "embedding_output_count_mismatch"
                        )
                    for item, vector in zip(chunk, vectors):
                        item.vector = vector

            except EmbeddingUnavailable as error:
                missing_keys = {item.cache_key for item in missing}
                for item in items:
                    if item.cache_key in missing_keys:
                        item.vector = None
                        item.error = str(error)
                return finish(items)

            cache_started = perf_counter()
            winners, inserted, repaired = self.cache.put_many(
                missing,
                namespace=namespace,
                data_version=data_version,
                signature_json=self.signature_json,
            )
            usage.cache_write_seconds += perf_counter() - cache_started
            usage.inserted_cache_count = inserted
            usage.repaired_cache_count = repaired

            for item in items:
                if item.cache_key in winners:
                    item.vector = winners[item.cache_key]

            return finish(items)