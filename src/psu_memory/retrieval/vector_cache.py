"""SQLite 持久化向量缓存；复用经验池连接与事务。"""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime, timezone

from ..pool._common import canonical_json, require_text, sha256_text
from ..pool.pool_store import POOL_SCHEMA_VERSION, SQLitePoolStore
from .retrieval_models import EmbeddingItem, validate_dense_vector


RETRIEVAL_SCHEMA_VERSION = 1
_COMPONENT = "psu_memory_retrieval"

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS psu_retrieval_meta (
        component TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psu_retrieval_vectors (
        cache_key TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        data_version TEXT NOT NULL,
        input_sha256 TEXT NOT NULL,
        signature_json TEXT NOT NULL,
        token_count INTEGER NOT NULL CHECK(token_count > 0),
        dimension INTEGER NOT NULL CHECK(dimension > 0),
        vector_blob BLOB NOT NULL,
        vector_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
)


def vector_cache_key(
    namespace: str,
    data_version: str,
    input_sha256: str,
    signature_json: str,
) -> str:
    for name, value in (
        ("namespace", namespace),
        ("data_version", data_version),
        ("input_sha256", input_sha256),
        ("signature_json", signature_json),
    ):
        require_text(value, name)

    return sha256_text(canonical_json({
        "namespace": namespace,
        "data_version": data_version,
        "input_sha256": input_sha256,
        "embedding_signature": signature_json,
    }))


def _read_rows(connection, keys: list[str]) -> dict:
    rows = {}
    for start in range(0, len(keys), 500):
        chunk = keys[start:start + 500]
        placeholders = ",".join("?" for _ in chunk)
        fetched = connection.execute(
            f"""
            SELECT cache_key, namespace, data_version, input_sha256,
                   signature_json, token_count, dimension,
                   vector_blob, vector_sha256
            FROM psu_retrieval_vectors
            WHERE cache_key IN ({placeholders})
            """,
            chunk,
        )
        rows.update((row[0], row) for row in fetched)
    return rows


def _decode(row, signature_json: str) -> tuple[float, ...]:
    signature = json.loads(signature_json)
    dimension = signature["embedding_dimension"]

    if row[4] != signature_json:
        raise ValueError("缓存编码签名不匹配")

    if row[0] != vector_cache_key(row[1], row[2], row[3], row[4]):
        raise ValueError("缓存元数据与标识不匹配")

    if row[6] != dimension:
        raise ValueError("缓存维度不匹配")

    if (
        type(row[5]) is not int
        or not 0 < row[5] <= signature["embedding_max_tokens"]
    ):
        raise ValueError("缓存 Token 数不合法")

    blob = row[7]
    if not isinstance(blob, bytes) or len(blob) != dimension * 4:
        raise ValueError("缓存向量字节数不匹配")

    if hashlib.sha256(blob).hexdigest() != row[8]:
        raise ValueError("缓存向量哈希不匹配")

    values = struct.unpack(f"<{dimension}f", blob)
    return validate_dense_vector(values, dimension)


class SQLiteVectorCache:
    def __init__(self, pool_store: SQLitePoolStore) -> None:
        self.pool_store = pool_store

    @staticmethod
    def _check_schema(
        connection,
        *,
        component: str = _COMPONENT,
        expected_version: int = RETRIEVAL_SCHEMA_VERSION,
        required: bool = True,
    ) -> bool:
        """复用同一元数据注册表，检查指定组件版本。"""

        exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='psu_retrieval_meta'
            """
        ).fetchone()

        if not exists:
            if required:
                raise ValueError(f"组件尚未初始化：{component}")
            return False

        row = connection.execute(
            """
            SELECT schema_version FROM psu_retrieval_meta
            WHERE component=?
            """,
            (component,),
        ).fetchone()

        if row is None:
            if required:
                raise ValueError(f"组件尚未初始化：{component}")
            return False

        if row[0] != expected_version:
            raise ValueError(f"组件版本不兼容：{component}")

        return True

    def initialize(self) -> dict:
        with self.pool_store.connection(write=True) as connection:
            connection.execute(_SCHEMA[0])

            old = connection.execute(
                """
                SELECT schema_version FROM psu_retrieval_meta
                WHERE component=?
                """,
                (_COMPONENT,),
            ).fetchone()
            if old is not None and old[0] != RETRIEVAL_SCHEMA_VERSION:
                raise ValueError("向量缓存组件版本不兼容")

            connection.execute(_SCHEMA[1])
            connection.execute(
                """
                INSERT OR IGNORE INTO psu_retrieval_meta
                    (component, schema_version)
                VALUES (?, ?)
                """,
                (_COMPONENT, RETRIEVAL_SCHEMA_VERSION),
            )

        return self.info()

    def get_many(
        self,
        keys: list[str],
        *,
        signature_json: str,
        connection=None,
    ) -> tuple[dict[str, tuple[float, ...]], tuple[str, ...]]:
        """正常调用自行建立连接；同库操作可复用调用方事务。"""

        def read(current_connection):
            found = {}
            corrupted = []

            self._check_schema(current_connection)
            rows = _read_rows(
                current_connection,
                list(dict.fromkeys(keys)),
            )

            for key, row in rows.items():
                try:
                    found[key] = _decode(row, signature_json)
                except (ValueError, TypeError, KeyError, struct.error):
                    corrupted.append(key)

            return found, tuple(sorted(corrupted))

        if connection is not None:
            return read(connection)

        with self.pool_store.connection() as current_connection:
            return read(current_connection)

    def put_many(
        self,
        items: list[EmbeddingItem],
        *,
        namespace: str,
        data_version: str,
        signature_json: str,
    ) -> tuple[dict[str, tuple[float, ...]], int, int]:
        """原子写入；并发情况下保留先前已提交的有效向量。"""

        signature = json.loads(signature_json)
        dimension = signature["embedding_dimension"]
        now = datetime.now(timezone.utc).isoformat()
        payloads = {}

        # 序列化在写事务之外完成。
        for item in items:
            prepared = item.prepared
            if not item.ready or not prepared.ready:
                raise ValueError("不能保存不可用的编码结果")

            input_sha256 = prepared.matching.input_sha256
            key = vector_cache_key(
                namespace, data_version, input_sha256, signature_json
            )
            if item.cache_key != key:
                raise ValueError("编码结果的缓存标识不匹配")

            if not 0 < prepared.token_count <= signature["embedding_max_tokens"]:
                raise ValueError("编码结果的 Token 数不合法")

            vector = validate_dense_vector(item.vector, dimension)
            blob = struct.pack(f"<{dimension}f", *vector)

            payloads[key] = (
                key, namespace, data_version, input_sha256,
                signature_json, prepared.token_count, dimension,
                blob, hashlib.sha256(blob).hexdigest(), now,
            )

        winners = {}
        inserted = 0
        repaired = 0

        with self.pool_store.connection(write=True) as connection:
            self._check_schema(connection)
            existing = _read_rows(connection, list(payloads))

            for key, payload in payloads.items():
                old = existing.get(key)
                if old is not None:
                    try:
                        winners[key] = _decode(old, signature_json)
                        continue
                    except (ValueError, TypeError, KeyError, struct.error):
                        repaired += 1
                else:
                    inserted += 1

                connection.execute(
                    """
                    INSERT INTO psu_retrieval_vectors (
                        cache_key, namespace, data_version, input_sha256,
                        signature_json, token_count, dimension,
                        vector_blob, vector_sha256, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        namespace=excluded.namespace,
                        data_version=excluded.data_version,
                        input_sha256=excluded.input_sha256,
                        signature_json=excluded.signature_json,
                        token_count=excluded.token_count,
                        dimension=excluded.dimension,
                        vector_blob=excluded.vector_blob,
                        vector_sha256=excluded.vector_sha256,
                        created_at=excluded.created_at
                    """,
                    payload,
                )

                # 返回实际持久化的 float32 值。
                winners[key] = _decode(payload[:9], signature_json)

        return winners, inserted, repaired

    def info(self) -> dict:
        with self.pool_store.connection() as connection:
            self._check_schema(connection)
            count, size = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(LENGTH(vector_blob)), 0)
                FROM psu_retrieval_vectors
                """
            ).fetchone()

            return {
                "path": str(self.pool_store.path),
                "history_schema_version": connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0],
                "pool_schema_version": POOL_SCHEMA_VERSION,
                "retrieval_schema_version": RETRIEVAL_SCHEMA_VERSION,
                "vector_count": count,
                "vector_bytes": size,
            }