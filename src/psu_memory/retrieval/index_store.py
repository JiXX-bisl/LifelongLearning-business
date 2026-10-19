"""当前成员与向量缓存的索引关联；不复制经验或向量。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..pool._common import (
    MATCH_TEXT_RULE_VERSION,
    canonical_json,
    require_text,
    sha256_text,
)
from .embedding import embedding_signature
from .retrieval_models import RetrievalConfig
from .vector_cache import SQLiteVectorCache, vector_cache_key
from ..pool.pool_models import ContentKey


INDEX_SCHEMA_VERSION = 1
INDEX_RULE_VERSION = "member_index_v1"
REPRESENTATIONS = ("default", "question_only")
_COMPONENT = "psu_memory_retrieval_index"

EXCLUDED_REASONS = frozenset({
    "invalid_context",
    "question_too_long",
})

_COLUMNS = (
    "index_key, namespace, data_version, execution_id, content_version, "
    "annotation_sha256, cache_key, binding_json, binding_sha256"
)

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS psu_retrieval_index_members (
        index_key TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        data_version TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        content_version TEXT NOT NULL,
        annotation_sha256 TEXT NOT NULL,
        cache_key TEXT REFERENCES psu_retrieval_vectors(cache_key),
        binding_json TEXT NOT NULL,
        binding_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (
            execution_id, content_version, annotation_sha256
        ) REFERENCES psu_pool_annotations (
            execution_id, content_version, annotation_sha256
        )
    )
"""


def member_stamp(member: dict) -> str:
    """只指纹化影响索引的成员身份、标注和问题内容。"""

    content = member["quality"].content
    return sha256_text(canonical_json({
        "identity": member["identity"],
        "annotation_sha256": member["annotation_sha256"],
        "group_id": member["candidate"].group_id,
        "question": content["question"],
        "context_text": content.get("context_text", ""),
    }))


def _snapshot_stamps(members: list[dict]) -> tuple[str, ...]:
    return tuple(sorted(member_stamp(member) for member in members))


def _binding_rows(connection, keys: list[str]) -> dict:
    exists = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='psu_retrieval_index_members'
        """
    ).fetchone()
    if not exists:
        return {}

    rows = {}
    for start in range(0, len(keys), 500):
        chunk = keys[start:start + 500]
        placeholders = ",".join("?" for _ in chunk)
        fetched = connection.execute(
            f"""
            SELECT {_COLUMNS}
            FROM psu_retrieval_index_members
            WHERE index_key IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        rows.update((row[0], row) for row in fetched)
    return rows


class SQLiteRetrievalIndex:
    def __init__(
        self,
        cache: SQLiteVectorCache,
        config: RetrievalConfig,
    ) -> None:
        self.cache = cache
        self.pool_store = cache.pool_store
        self.config = config
        self.signature_json = embedding_signature(config)

    @property
    def text_profile(self) -> dict:
        return {
            "index_rule_version": INDEX_RULE_VERSION,
            "matching_text_rule_version": MATCH_TEXT_RULE_VERSION,
            "max_context_questions": self.config.max_context_questions,
        }

    def _registered(self, connection, *, required=False) -> bool:
        return self.cache._check_schema(
            connection,
            component=_COMPONENT,
            expected_version=INDEX_SCHEMA_VERSION,
            required=required,
        )

    def initialize(self) -> dict:
        self.cache.initialize()

        with self.pool_store.connection(write=True) as connection:
            self._registered(connection)
            connection.execute(_SCHEMA)
            connection.execute(
                """
                INSERT OR IGNORE INTO psu_retrieval_meta
                    (component, schema_version)
                VALUES (?, ?)
                """,
                (_COMPONENT, INDEX_SCHEMA_VERSION),
            )

        return self.info()

    def descriptor(self, member: dict, representation: str) -> dict:
        if representation not in REPRESENTATIONS:
            raise ValueError("未知索引表示")

        base = {
            "identity": dict(member["identity"]),
            "annotation_sha256": member["annotation_sha256"],
            "group_id": member["candidate"].group_id,
            "member_sha256": member_stamp(member),
            "representation": representation,
            "text_profile": self.text_profile,
            "embedding_signature": self.signature_json,
        }
        return {
            **base,
            "index_key": sha256_text(canonical_json(base)),
        }

    @staticmethod
    def _validate_binding(expected: dict, binding: dict) -> None:
        if not isinstance(binding, dict):
            raise ValueError("索引关联必须是对象")

        if any(binding.get(key) != value for key, value in expected.items()):
            raise ValueError("索引依据与当前成员不一致")

        status = binding.get("status")
        cache_key = binding.get("cache_key")

        if status == "ready":
            matching = binding["matching"]
            if matching.get("ready") is not True:
                raise ValueError("可用关联缺少匹配文本")

            identity = expected["identity"]
            expected_cache_key = vector_cache_key(
                identity["namespace"],
                identity["data_version"],
                matching["input_sha256"],
                expected["embedding_signature"],
            )
            if cache_key != expected_cache_key:
                raise ValueError("索引关联的缓存标识不一致")

        elif status == "excluded":
            if cache_key is not None:
                raise ValueError("排除记录不应引用向量")
            if binding.get("reason") not in EXCLUDED_REASONS:
                raise ValueError("排除原因不是确定性文本问题")
        else:
            raise ValueError("未知索引处理结果")

    def publish(
        self,
        *,
        namespace: str,
        data_version: str,
        members: list[dict],
        bindings: list[dict],
    ) -> dict:
        """发布整批关联；在写事务中复核成员及缓存。"""

        expected = {
            item["index_key"]: item
            for member in members
            for representation in REPRESENTATIONS
            for item in [self.descriptor(member, representation)]
        }

        if (
            len(bindings) != len(expected)
            or {item["index_key"] for item in bindings} != set(expected)
        ):
            raise ValueError("双表示索引处理结果不完整")

        prepared_rows = []
        now = datetime.now(timezone.utc).isoformat()

        for binding in bindings:
            self._validate_binding(
                expected[binding["index_key"]], binding
            )
            identity = binding["identity"]
            text = canonical_json(binding)
            prepared_rows.append((
                binding["index_key"],
                identity["namespace"],
                identity["data_version"],
                identity["execution_id"],
                identity["content_version"],
                binding["annotation_sha256"],
                binding["cache_key"],
                text,
                sha256_text(text),
                now,
            ))

        with self.pool_store.connection(write=True) as connection:
            self._registered(connection, required=True)

            # 复用第一批共享判断，不重新实现成员资格检查。
            current = self.pool_store._read_active_members(
                connection,
                namespace=namespace,
                data_version=data_version,
            )
            if _snapshot_stamps(current) != _snapshot_stamps(members):
                return {
                    "published": False,
                    "reason": "members_changed",
                    "inserted_count": 0,
                    "updated_count": 0,
                    "unchanged_count": 0,
                }

            ready_keys = list({
                item["cache_key"]
                for item in bindings
                if item["status"] == "ready"
            })
            vectors, corrupted = self.cache.get_many(
                ready_keys,
                signature_json=self.signature_json,
                connection=connection,
            )
            if corrupted or set(vectors) != set(ready_keys):
                return {
                    "published": False,
                    "reason": "index_cache_unavailable",
                    "inserted_count": 0,
                    "updated_count": 0,
                    "unchanged_count": 0,
                }

            existing = _binding_rows(connection, list(expected))
            inserted = updated = unchanged = 0

            for row in prepared_rows:
                old = existing.get(row[0])
                if old == row[:-1]:
                    unchanged += 1
                    continue

                if old is None:
                    inserted += 1
                else:
                    updated += 1

                connection.execute(
                    f"""
                    INSERT INTO psu_retrieval_index_members (
                        {_COLUMNS}, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(index_key) DO UPDATE SET
                        namespace=excluded.namespace,
                        data_version=excluded.data_version,
                        execution_id=excluded.execution_id,
                        content_version=excluded.content_version,
                        annotation_sha256=excluded.annotation_sha256,
                        cache_key=excluded.cache_key,
                        binding_json=excluded.binding_json,
                        binding_sha256=excluded.binding_sha256,
                        created_at=excluded.created_at
                    """,
                    row,
                )

        return {
            "published": True,
            "reason": "ok",
            "inserted_count": inserted,
            "updated_count": updated,
            "unchanged_count": unchanged,
        }

    def inspect(
        self,
        *,
        namespace: str,
        data_version: str,
        group_id=None,
    ) -> dict:
        """只读检查当前检索范围；不加载分词器或编码模型。"""

        require_text(namespace, "namespace")
        require_text(data_version, "data_version")

        with self.pool_store.connection() as connection:
            members = self.pool_store._read_active_members(
                connection,
                namespace=namespace,
                data_version=data_version,
                group_id=group_id,
            )
            expected = [
                self.descriptor(member, representation)
                for member in members
                for representation in REPRESENTATIONS
            ]
            rows = (
                _binding_rows(
                    connection,
                    [item["index_key"] for item in expected],
                )
                if self._registered(connection)
                else {}
            )

            bindings = []
            pending = []

            for descriptor in expected:
                row = rows.get(descriptor["index_key"])
                if row is None:
                    pending.append({
                        **descriptor,
                        "reason": "index_missing",
                    })
                    continue

                identity = descriptor["identity"]
                expected_header = (
                    descriptor["index_key"],
                    identity["namespace"],
                    identity["data_version"],
                    identity["execution_id"],
                    identity["content_version"],
                    descriptor["annotation_sha256"],
                )

                try:
                    if row[:6] != expected_header:
                        raise ValueError("索引行身份不一致")
                    if sha256_text(row[7]) != row[8]:
                        raise ValueError("索引关联哈希不一致")

                    binding = json.loads(row[7])
                    self._validate_binding(descriptor, binding)

                    if row[6] != binding["cache_key"]:
                        raise ValueError("缓存指针不一致")

                    bindings.append(binding)

                except (ValueError, TypeError, KeyError):
                    pending.append({
                        **descriptor,
                        "reason": "index_binding_invalid",
                    })

            ready_keys = list({
                item["cache_key"]
                for item in bindings
                if item["status"] == "ready"
            })
            vectors = {}
            corrupted = ()

            if ready_keys:
                vectors, corrupted = self.cache.get_many(
                    ready_keys,
                    signature_json=self.signature_json,
                    connection=connection,
                )

            valid = []
            for binding in bindings:
                key = binding["cache_key"]
                if binding["status"] == "excluded" or key in vectors:
                    valid.append(binding)
                else:
                    pending.append({
                        **binding,
                        "reason": (
                            "cache_corrupted"
                            if key in corrupted else "cache_missing"
                        ),
                    })

            return {
                "complete": not pending,
                "member_count": len(members),
                "expected_binding_count": len(expected),
                "ready_binding_count": sum(
                    item["status"] == "ready" for item in valid
                ),
                "excluded_binding_count": sum(
                    item["status"] == "excluded" for item in valid
                ),
                "pending_count": len(pending),
                "members": members,
                "bindings": valid,
                "pending": pending,
                "vectors": vectors,
            }
        
    def check_snapshots(
        self,
        *,
        namespace: str,
        data_version: str,
        snapshots: dict[str, dict],
    ) -> tuple[dict, dict]:
        """在一个只读检查中复核快照。

        第一个返回值用于报告。
        第二个返回值包含通过检查的当前成员和索引关联，
        只供内部使用，不直接写入 JSON。
        """
        inspection = self.inspect(
            namespace=namespace,
            data_version=data_version,
        )
        members = {
            member["candidate"].key: member
            for member in inspection["members"]
        }
        bindings = {
            binding["index_key"]: binding
            for binding in inspection["bindings"]
            if binding["status"] == "ready"
        }

        invalid = []
        current_candidates = {}

        for alias, snapshot in snapshots.items():
            identity = snapshot["identity"]
            key = ContentKey(
                identity["execution_id"],
                identity["content_version"],
            )
            current = members.get(key)
            binding = bindings.get(snapshot["index_key"])
            reason = None

            if (
                identity["namespace"] != namespace
                or identity["data_version"] != data_version
            ):
                reason = "candidate_scope_mismatch"
            elif current is None:
                reason = "not_current_active_member"
            elif (
                current["identity"] != identity
                or current["annotation_sha256"]
                != snapshot["annotation_sha256"]
                or current["quality_sha256"]
                != snapshot["quality_sha256"]
            ):
                reason = "quality_or_annotation_changed"
            elif "candidate" in snapshot and (
                snapshot["candidate"]["member"]["identity"] != identity
                or snapshot["candidate"]["member"]["quality"]
                != current["quality"]
                or snapshot["candidate"]["member"]["candidate"]
                != current["candidate"]
            ):
                reason = "candidate_payload_changed"
            elif binding is None:
                reason = "index_binding_unavailable"
            elif (
                "candidate" in snapshot
                and snapshot["candidate"]["binding"] != binding
            ):
                reason = "index_binding_changed"

            if reason is not None:
                invalid.append({
                    "alias": alias,
                    "identity": dict(identity),
                    "reason": reason,
                })
            else:
                current_candidates[alias] = {
                    "member": current,
                    "binding": binding,
                }

        return {
            "checked_aliases": list(snapshots),
            "invalid": invalid,
            "active_member_count": inspection["member_count"],
            "pending_index_count": inspection["pending_count"],
        }, current_candidates        

    def info(self) -> dict:
        with self.pool_store.connection() as connection:
            self._registered(connection, required=True)
            count, vector_count = connection.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT cache_key)
                FROM psu_retrieval_index_members
                """
            ).fetchone()

        return {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "stored_binding_count": count,
            "referenced_vector_count": vector_count,
        }
    
def require_index_compatibility(
    index: SQLiteRetrievalIndex,
    embeddings,
) -> None:
    """构建与检索组件共用依赖兼容检查。"""

    if index.pool_store.path != embeddings.cache.pool_store.path:
        raise ValueError("索引和编码服务必须使用同一数据库")
    if index.config != embeddings.config:
        raise ValueError("索引和编码服务配置不一致")