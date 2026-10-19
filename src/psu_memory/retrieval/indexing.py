"""离线构建双表示索引；编码服务实例在整批操作中复用。"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict
from time import perf_counter

from ..pool._common import require_text
from ..pool.pool_models import GROUP_IDS
from .embedding_service import EmbeddingService
from .index_store import (
    EXCLUDED_REASONS,
    REPRESENTATIONS,
    SQLiteRetrievalIndex,
    require_index_compatibility
)
from .retrieval_models import EmbeddingUsage
from .text import build_matching_text


class OfflineIndexBuilder:
    def __init__(
        self,
        index: SQLiteRetrievalIndex,
        embeddings: EmbeddingService,
    ) -> None:
        require_index_compatibility(index, embeddings)
        self.index = index
        self.embeddings = embeddings

    def build(
        self,
        *,
        namespace: str,
        data_version: str,
    ) -> dict:
        require_text(namespace, "namespace")
        require_text(data_version, "data_version")
        started = perf_counter()
        stages = {}

        stage = perf_counter()
        inspection = self.index.inspect(
            namespace=namespace,
            data_version=data_version,
        )
        members = inspection["members"]
        stages["initial_inspection_seconds"] = perf_counter() - stage
        if inspection["complete"]:
            # inspect 已校验当前成员、内容版本、文本规则和缓存完整性。
            # 成员减少时，已有剩余关联同样可直接使用，无需再发布索引。
            return self._report(
                namespace=namespace,
                data_version=data_version,
                members=members,
                inspection=inspection,
                status="ok",
                publication={
                    "published": False,
                    "reason": "index_unchanged",
                    "inserted_count": 0,
                    "updated_count": 0,
                    "unchanged_count": inspection["expected_binding_count"],
                },
                batches=[],
                errors=[],
                stages=stages,
                started=started,
            )

        stage = perf_counter()
        texts = [
            build_matching_text(
                member["quality"].content["question"],
                member["quality"].content.get("context_text", ""),
                max_context_questions=self.index.config.max_context_questions,
            )
            for member in members
        ]
        stages["text_preparation_seconds"] = perf_counter() - stage

        batches = []
        bindings = []
        errors = []

        for representation in REPRESENTATIONS:
            stage = perf_counter()
            batch = self.embeddings.encode_many(
                texts,
                namespace=namespace,
                data_version=data_version,
                question_only=(representation == "question_only"),
            )
            batches.append(batch)
            stages[representation + "_embedding_seconds"] = (
                perf_counter() - stage
            )

            for member, item in zip(members, batch.items):
                descriptor = self.index.descriptor(
                    member, representation
                )

                if item.ready:
                    status = "ready"
                    reason = item.prepared.matching.reason_code
                    cache_key = item.cache_key
                elif (
                    item.error is None
                    and item.prepared.matching.reason_code in EXCLUDED_REASONS
                ):
                    status = "excluded"
                    reason = item.prepared.matching.reason_code
                    cache_key = None
                else:
                    errors.append({
                        "identity": member["identity"],
                        "representation": representation,
                        "reason": item.error or "embedding_unavailable",
                    })
                    continue

                bindings.append({
                    **descriptor,
                    "status": status,
                    "reason": reason,
                    "cache_key": cache_key,
                    "matching": item.prepared.matching.to_dict(),
                    "mode": item.prepared.mode,
                    "token_count": item.prepared.token_count,
                    "original_token_count": item.prepared.original_token_count,
                })

            # 编码失败后停止，不通过第二种表示隐式重试。
            if errors:
                break

        publication = {
            "published": False,
            "reason": "embedding_failed",
            "inserted_count": 0,
            "updated_count": 0,
            "unchanged_count": 0,
        }

        stage = perf_counter()
        if not errors:
            try:
                publication = self.index.publish(
                    namespace=namespace,
                    data_version=data_version,
                    members=members,
                    bindings=bindings,
                )
                if not publication["published"]:
                    errors.append({
                        "reason": publication["reason"],
                    })
            except (sqlite3.Error, ValueError) as error:
                publication["reason"] = "index_write_failed"
                errors.append({
                    "reason": "index_write_failed",
                    "error_type": type(error).__name__,
                })
        stages["publication_seconds"] = perf_counter() - stage

        stage = perf_counter()
        inspection = self.index.inspect(
            namespace=namespace,
            data_version=data_version,
        )
        stages["verification_seconds"] = perf_counter() - stage

        if errors:
            status = (
                "stale"
                if publication["reason"] == "members_changed"
                else "failed"
            )
        elif not inspection["complete"]:
            status = "index_not_ready"
            errors.append({
                "reason": "index_incomplete_after_publication",
            })
        else:
            status = "ok"

        return self._report(
            namespace=namespace,
            data_version=data_version,
            members=members,
            inspection=inspection,
            status=status,
            publication=publication,
            batches=batches,
            errors=errors,
            stages=stages,
            started=started,
        )

    def _report(
        self,
        *,
        namespace,
        data_version,
        members,
        inspection,
        status,
        publication,
        batches,
        errors,
        stages,
        started,
    ) -> dict:
        # 聚合现有 EmbeddingUsage，不引入第二套成本结构。
        usage = asdict(EmbeddingUsage(
            elapsed_seconds=0.0,
            model_call_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            embedding_input_tokens=0,
        ))
        for name in usage:
            if name == "elapsed_seconds":
                continue
            values = [getattr(batch.usage, name) for batch in batches]
            usage[name] = (
                None if any(value is None for value in values)
                else sum(values)
            )
        usage["elapsed_seconds"] = perf_counter() - started

        return {
            "operation": "build_retrieval_index",
            "status": status,
            "namespace": namespace,
            "data_version": data_version,
            "text_profile": self.index.text_profile,
            "embedding_signature": self.index.signature_json,
            "member_count": len(members),
            "current_member_count": inspection["member_count"],
            "group_counts": {
                group: sum(
                    member["candidate"].group_id == group
                    for member in members
                )
                for group in GROUP_IDS
            },
            "expected_binding_count": inspection["expected_binding_count"],
            "ready_binding_count": inspection["ready_binding_count"],
            "excluded_binding_count": inspection["excluded_binding_count"],
            "unique_cache_key_count": len({
                item["cache_key"]
                for item in inspection["bindings"]
                if item["status"] == "ready"
            }),
            "index_complete": inspection["complete"],
            "pending_count": inspection["pending_count"],
            "pending": inspection["pending"],
            "publication": publication,
            "error_count": len(errors),
            "errors": errors,
            "bindings": inspection["bindings"],
            "stage_seconds": stages,
            "usage": usage,
        }