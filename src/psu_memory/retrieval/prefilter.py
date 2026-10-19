"""规则路由与语义初筛；不调用 LLM，不重建历史索引。"""

from __future__ import annotations

import sqlite3
from time import perf_counter

import numpy as np

from ..models import RetrievalRequest
from ..pool.classification import classify
from ..pool.pool_models import GROUP_IDS, ContentKey, GroupingInput
from ..pool.selection import BasicRankingPolicy, RankingPolicy
from .embedding_service import EmbeddingService
from .index_store import (
    SQLiteRetrievalIndex,
    require_index_compatibility,
)
from .retrieval_models import PrefilterResult
from .text import build_matching_text, validate_request, retrieval_request_fingerprint


class SemanticPrefilter:
    def __init__(
        self,
        index: SQLiteRetrievalIndex,
        embeddings: EmbeddingService,
        *,
        ranking_policy: RankingPolicy | None = None,
    ) -> None:
        require_index_compatibility(index, embeddings)
        self.index = index
        self.embeddings = embeddings
        self.config = index.config
        self.ranking_policy = (
            ranking_policy
            if ranking_policy is not None
            else BasicRankingPolicy()
        )

    @staticmethod
    def _ready_pairs(inspection: dict, representation: str) -> list[tuple]:
        """连接已有成员视图与所需表示，不重新判断资格。"""

        bindings = {
            ContentKey(
                item["identity"]["execution_id"],
                item["identity"]["content_version"],
            ): item
            for item in inspection["bindings"]
            if (
                item["representation"] == representation
                and item["status"] == "ready"
            )
        }

        return [
            (member, bindings[member["candidate"].key])
            for member in inspection["members"]
            if member["candidate"].key in bindings
        ]

    @staticmethod
    def _similarities(
        pairs: list[tuple],
        query_vector: tuple[float, ...],
        vectors: dict,
    ) -> dict[str, float]:
        """同一历史向量只计算一次，不合并执行记录。"""

        if not pairs:
            return {}

        keys = sorted({
            binding["cache_key"] for _, binding in pairs
        })
        matrix = np.asarray(
            [vectors[key] for key in keys],
            dtype=np.float64,
        )
        query = np.asarray(query_vector, dtype=np.float64)

        # 向量已由原编码器与缓存完成归一化校验。
        scores = np.clip(matrix @ query, -1.0, 1.0)
        by_cache_key = {
            key: float(score)
            for key, score in zip(keys, scores)
        }

        return {
            binding["index_key"]: by_cache_key[binding["cache_key"]]
            for _, binding in pairs
        }

    def prefilter(self, request: RetrievalRequest) -> PrefilterResult:
        # 复用请求校验；后续直接调用文本整理，避免再调用其请求包装层。
        validate_request(request)
        started = perf_counter()
        result = PrefilterResult(
            status="failed",
            reason="not_started",
        )
        result.diagnostics = {
            "prefilter_limit": self.config.prefilter_limit,
            "ranking_policy_version": self.ranking_policy.version,
            "similarity_metric": "cosine",
            "text_profile": self.index.text_profile,
            "embedding_signature": self.index.signature_json,
            "request_fingerprint": retrieval_request_fingerprint(request),
        }

        def finish(status: str, reason: str) -> PrefilterResult:
            result.status = status
            result.reason = reason
            result.usage.elapsed_seconds = perf_counter() - started
            return result

        if request.limit == 0:
            return finish("disabled", "limit_zero")
        if request.max_context_chars == 0:
            return finish("disabled", "context_budget_zero")

        stage = perf_counter()
        matching = build_matching_text(
            request.question,
            request.context_text,
            max_context_questions=self.config.max_context_questions,
        )
        result.stage_seconds["text_preparation_seconds"] = (
            perf_counter() - stage
        )
        result.query = {
            "matching": matching.to_dict(),
        }

        if not matching.ready:
            return finish("query_unavailable", matching.reason_code)

        stage = perf_counter()
        decision = classify(
            GroupingInput(
                execution_id=request.task_id,
                # 路由使用匹配文本指纹，不创建业务内容版本。
                content_version=matching.input_sha256,
                question=matching.question,
                namespace=request.namespace,
                data_version=request.data_version,
                context_text=request.context_text,
                data_advice=request.data_advice,
            ),
            max_context_questions=self.config.max_context_questions,
        )
        result.stage_seconds["classification_seconds"] = (
            perf_counter() - stage
        )

        scope = {
            "namespace": request.namespace,
            "data_version": request.data_version,
        }
        search_group = decision.group_id
        result.routing = {
            "decision": decision.to_dict(),
            "requested_group": decision.group_id,
            "search_scope": (
                "same_group" if search_group is not None else "all_groups"
            ),
            "fallback_reason": (
                None if search_group is not None else "unclassified"
            ),
        }

        try:
            stage = perf_counter()
            inspection = self.index.inspect(
                **scope, group_id=search_group
            )

            if search_group is not None and inspection["member_count"] == 0:
                search_group = None
                result.routing["search_scope"] = "all_groups"
                result.routing["fallback_reason"] = "group_empty"
                inspection = self.index.inspect(**scope)

            result.stage_seconds["history_inspection_seconds"] = (
                perf_counter() - stage
            )
            result.routing["searched_groups"] = [
                group
                for group in GROUP_IDS
                if any(
                    member["candidate"].group_id == group
                    for member in inspection["members"]
                )
            ]
            result.diagnostics["initial_member_count"] = (
                inspection["member_count"]
            )

            if inspection["member_count"] == 0:
                return finish("no_members", "no_active_members")

            if not inspection["complete"]:
                result.diagnostics["pending"] = inspection["pending"]
                result.diagnostics["pending_count"] = inspection["pending_count"]
                return finish("index_not_ready", "history_index_incomplete")

            if inspection["ready_binding_count"] == 0:
                return finish(
                    "no_encodable_members",
                    "history_text_unavailable",
                )

            stage = perf_counter()
            batch = self.embeddings.encode_many(
                [matching],
                **scope,
            )
            result.usage = batch.usage
            result.stage_seconds["query_embedding_seconds"] = (
                perf_counter() - stage
            )
            query_item = batch.items[0]
            result.query = query_item.to_dict()

            if not query_item.ready:
                return finish(
                    "query_unavailable",
                    query_item.error or query_item.prepared.matching.reason_code,
                )

            # 独立问题自身也是 question_only 模式，不能据此触发降级。
            # 只有明确的上下文超限原因，才统一选择仅问题索引。
            representation = (
                "question_only"
                if query_item.prepared.matching.reason_code
                == "context_overflow_question_only"
                else "default"
            )
            result.diagnostics["comparison_representation"] = representation

            pairs = self._ready_pairs(inspection, representation)
            if not pairs:
                return finish(
                    "no_encodable_members",
                    "selected_representation_unavailable",
                )

            stage = perf_counter()
            scores = self._similarities(
                pairs,
                query_item.vector,
                inspection["vectors"],
            )
            result.stage_seconds["similarity_seconds"] = (
                perf_counter() - stage
            )
            result.diagnostics["scored_member_count"] = len(pairs)
            result.diagnostics["scored_unique_vector_count"] = len({
                binding["cache_key"] for _, binding in pairs
            })

            # 读取新的事务快照，复用原索引和成员校验。
            stage = perf_counter()
            current = self.index.inspect(
                **scope, group_id=search_group
            )
            result.stage_seconds["final_inspection_seconds"] = (
                perf_counter() - stage
            )
            result.diagnostics["current_member_count"] = (
                current["member_count"]
            )

            if not current["complete"]:
                result.diagnostics["pending"] = current["pending"]
                result.diagnostics["pending_count"] = current["pending_count"]
                return finish("index_not_ready", "history_index_changed")

            current_pairs = self._ready_pairs(current, representation)
            current_keys = {
                binding["index_key"] for _, binding in current_pairs
            }
            added_keys = current_keys - set(scores)
            if added_keys:
                result.diagnostics["uncompared_member_count"] = len(added_keys)
                return finish("stale", "comparison_scope_changed")

            result.diagnostics["filtered_member_count"] = (
                len(pairs) - len(current_pairs)
            )

            if not current_pairs:
                return finish("stale", "no_current_candidates")

            # 缓存如在期间被修复，使用同一查询向量重算本地点积。
            # 这里不执行额外编码。
            changed_vectors = any(
                inspection["vectors"][binding["cache_key"]]
                != current["vectors"][binding["cache_key"]]
                for _, binding in current_pairs
            )
            if changed_vectors:
                stage = perf_counter()
                scores = self._similarities(
                    current_pairs,
                    query_item.vector,
                    current["vectors"],
                )
                result.stage_seconds["similarity_refresh_seconds"] = (
                    perf_counter() - stage
                )

            stage = perf_counter()
            candidates = [
                {
                    "member": member,
                    "binding": binding,
                    "similarity": scores[binding["index_key"]],
                    "quality_rank": self.ranking_policy.rank(
                        member["candidate"]
                    ),
                }
                for member, binding in current_pairs
            ]
            candidates.sort(key=lambda item: (
                -item["similarity"],
                item["quality_rank"].key,
                item["member"]["candidate"].execution_id,
                item["member"]["candidate"].content_version,
            ))
            result.candidates = candidates[:self.config.prefilter_limit]
            result.stage_seconds["ranking_seconds"] = perf_counter() - stage

            return finish("ok", "candidates_ready")

        except (sqlite3.Error, ValueError, OSError) as error:
            result.candidates = []
            result.diagnostics["error_type"] = type(error).__name__
            return finish("failed", "retrieval_processing_failed")