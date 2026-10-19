"""经验池的 SQLite 存储"""

from __future__ import annotations

import json
import sqlite3

from contextlib import contextmanager
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ..evaluation.quality_models import QualityDecision
from .pool_models import (
    GROUP_IDS,
    ContentKey,
    DifficultyAssessment,
    DifficultyResult,
    DifficultyUsage,
    GroupingDecision,
    GroupOverride,
    PoolConfig,
    PoolScope,
    PoolSnapshot,
    RuleMatch,
)
from .selection import candidate_from_evaluations, plan_group_update

from ._common import canonical_json, require_text, sha256_text


POOL_SCHEMA_VERSION = 2
_COMPONENT = "psu_memory_pool"


# 保存三类信息：标注版本、当前成员和变更记录
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS psu_pool_meta (
        component TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psu_pool_difficulty_versions (
        cache_key TEXT NOT NULL,
        assessment_sha256 TEXT NOT NULL,
        assessment_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (cache_key, assessment_sha256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psu_pool_difficulty_current (
        cache_key TEXT PRIMARY KEY,
        assessment_sha256 TEXT NOT NULL,
        FOREIGN KEY (cache_key, assessment_sha256)
            REFERENCES psu_pool_difficulty_versions (
                cache_key, assessment_sha256
            )
    )
    """,
)

_SCHEMA += (
    """
    CREATE TABLE IF NOT EXISTS psu_pool_annotations (
        execution_id TEXT NOT NULL,
        content_version TEXT NOT NULL,
        annotation_sha256 TEXT NOT NULL,
        grouping_json TEXT NOT NULL,
        cache_key TEXT NOT NULL,
        assessment_sha256 TEXT NOT NULL,
        quality_sha256 TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (
            execution_id, content_version, annotation_sha256
        ),
        FOREIGN KEY (
            execution_id, content_version, quality_sha256
        ) REFERENCES quality_versions (
            execution_id, content_version, decision_sha256
        ),
        FOREIGN KEY (
            cache_key, assessment_sha256
        ) REFERENCES psu_pool_difficulty_versions (
            cache_key, assessment_sha256
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psu_pool_annotation_current (
        execution_id TEXT NOT NULL,
        content_version TEXT NOT NULL,
        annotation_sha256 TEXT NOT NULL,
        PRIMARY KEY (execution_id, content_version),
        FOREIGN KEY (
            execution_id, content_version, annotation_sha256
        ) REFERENCES psu_pool_annotations (
            execution_id, content_version, annotation_sha256
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psu_pool_scopes (
        namespace TEXT NOT NULL,
        data_version TEXT NOT NULL,
        config_json TEXT NOT NULL,
        PRIMARY KEY (namespace, data_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psu_pool_members (
        namespace TEXT NOT NULL,
        data_version TEXT NOT NULL,
        group_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        content_version TEXT NOT NULL,
        annotation_sha256 TEXT NOT NULL,
        position INTEGER NOT NULL,
        joined_at TEXT NOT NULL,
        PRIMARY KEY (namespace, data_version, execution_id),
        FOREIGN KEY (
            execution_id, content_version, annotation_sha256
        ) REFERENCES psu_pool_annotations (
            execution_id, content_version, annotation_sha256
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS psu_pool_events (
        event_id TEXT PRIMARY KEY,
        event_json TEXT NOT NULL
    )
    """,
)

_CACHE_QUERY = """
    SELECT v.assessment_json
    FROM psu_pool_difficulty_current AS current
    JOIN psu_pool_difficulty_versions AS v
      ON v.cache_key = current.cache_key
     AND v.assessment_sha256 = current.assessment_sha256
    WHERE current.cache_key = ?
"""


def _decode_assessment(text: str) -> DifficultyAssessment:
    payload = json.loads(text)
    payload["origin_usage"] = DifficultyUsage(**payload["origin_usage"])
    payload["missing_information"] = tuple(payload["missing_information"])
    return DifficultyAssessment(**payload)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _local_usage(started: float) -> dict:
    return {
        "elapsed_seconds": round(perf_counter() - started, 6),
        "model_call_count": 0,
        "http_request_count": 0,
        "retry_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "embedding_input_tokens": 0,
        "token_usage_complete": True,
    }

def _decode_quality(text: str) -> QualityDecision:
    payload = json.loads(text)
    names = {item.name for item in fields(QualityDecision)}
    return QualityDecision(
        **{key: value for key, value in payload.items() if key in names}
    )

def _decode_grouping(text: str) -> GroupingDecision:
    payload = json.loads(text)
    payload["matches"] = tuple(
        RuleMatch(**item) for item in payload.get("matches", ())
    )
    payload["manual_history"] = tuple(
        GroupOverride(**item)
        for item in payload.get("manual_history", ())
    )
    return GroupingDecision(**payload)

def _pool_event(connection, report: dict, *, record_runs: bool = True) -> dict:
    result = {
        **report,
        "event_id": uuid4().hex,
        "created_at": _now()
    }
    if record_runs:
        connection.execute(
            "INSERT INTO psu_pool_events(event_id, event_json) VALUES (?, ?)",
            (result["event_id"], canonical_json(result)),
        )
    return result

def invalidate_pool_members(
    connection, 
    execution_id: str,
    *,
    keep_version: str | None,
    reason: str,
    record_runs: bool = True,
) -> int:
    """使用调用方事物移除成员"""
    started = perf_counter()
    exists = connection.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='psu_pool_members'
        """
    ).fetchone()

    if not exists:
        return 0
    
    rows = connection.execute(
        """
        SELECT namespace, data_version, group_id, content_version
        FROM psu_pool_members
        WHERE execution_id=?
          AND (? IS NULL OR content_version<>?)
        ORDER BY namespace, data_version, group_id, content_version
        """,
        (execution_id, keep_version, keep_version), 
    ).fetchall()

    if not rows:
        return 0
    
    removed = []
    for namespace, data_version, group_id, version in rows:
        connection.execute(
            """
            DELETE FROM psu_pool_members
            WHERE namespace=? AND data_version=?
              AND execution_id=? AND content_version=?
            """,
            (namespace, data_version, execution_id, version),
        )
        removed.append({
            "namespace": namespace,
            "data_version": data_version,
            "group_id": group_id,
            "execution_id": execution_id,
            "content_version": version,
        })

    _pool_event(connection, {
        "operation": "invalidate",
        "reason": reason,
        "removed": removed,
        "usage": _local_usage(started),
    }, record_runs=record_runs)
    return len(removed)


class SQLitePoolStore:
    def __init__(self, path: str | Path, *, record_runs: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        self.record_runs = record_runs

    @staticmethod
    def _check_pool_schema(connection) -> None:
        exists = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'psu_pool_meta'
            """
        ).fetchone()

        if not exists:
            raise ValueError(
                "经验池存储尚未初始化，请先调用 initialize()"
            )

        row = connection.execute(
            """
            SELECT schema_version
            FROM psu_pool_meta
            WHERE component = ?
            """,
            (_COMPONENT,),
        ).fetchone()

        if row is None:
            raise ValueError("缺少经验池组件版本记录")

        if row[0] != POOL_SCHEMA_VERSION:
            raise ValueError(
                f"不支持的经验池存储版本：{row[0]}"
            )
    
    @contextmanager
    def _connection(
        self,
        *,
        write: bool = False,
        require_pool: bool = True,
    ):
        """每次操作使用独立连接；写操作在一个事务内完成。"""
        mode = "rw" if write else "ro"

        connection = sqlite3.connect(
            self.path.as_uri() + f"?mode={mode}",
            uri=True,
            timeout=30,
        )

        try:
            connection.execute("PRAGMA foreign_keys = ON")

            with connection:
                connection.execute(
                    "BEGIN IMMEDIATE" if write else "BEGIN"
                )

                history_version = connection.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]

                if history_version != 1:
                    raise ValueError(
                        "需要 user_version=1 的现有历史候选库"
                    )

                if require_pool:
                    self._check_pool_schema(connection)

                yield connection
        finally:
            connection.close()

    @contextmanager
    def connection(self, *, write: bool = False):
        with self._connection(write=write) as connection:
            yield connection
    
    def initialize(self) -> dict:
        """追加组件表，不创建新的历史库"""
        with self._connection(
            write=True,
            require_pool=False
        ) as connection:
            existing_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            require_tables = {
                "history_records",
                "quality_contents",
                "quality_heads",
                "quality_versions",
                "quality_current",
            }

            missing = require_tables - existing_tables
            if missing:
                raise ValueError("请先完成候选导入与质量评价")
            
            connection.execute(_SCHEMA[0])

            registered = connection.execute(
                """
                SELECT schema_version
                FROM psu_pool_meta
                WHERE component = ?
                """,
                (_COMPONENT,),
            ).fetchone()

            if (
                registered is not None
                and registered[0] not in (1, POOL_SCHEMA_VERSION)
            ):
                raise ValueError(
                    f"不支持的经验池存储版本: {registered[0]}"
                )
            
            for statement in _SCHEMA[1:]:
                connection.execute(statement)

            connection.execute(
                """
                INSERT INTO psu_pool_meta(component, schema_version)
                VALUES (?, ?)
                ON CONFLICT(component) DO UPDATE SET
                    schema_version=excluded.schema_version
                WHERE schema_version<>excluded.schema_version
                """,
                (_COMPONENT, POOL_SCHEMA_VERSION),
            )
        return self.info()
    
    @staticmethod
    def _read_cache(connection, key: str):
        row = connection.execute(_CACHE_QUERY, (key,)).fetchone()
        return _decode_assessment(row[0]) if row else None
    
    @staticmethod
    def _read_inputs(
        connection,
        *,
        namespace: str,
        data_version: str,
        execution_id: str | None = None,
        members_only: bool = False,
    ) -> list[dict]:
        sql = """
            SELECT
                h.execution_id,
                head.content_version,
                current.decision_sha256,
                v.decision_json,
                a.annotation_sha256,
                av.grouping_json,
                d.assessment_json
            FROM history_records AS h
            JOIN quality_heads AS head
              ON head.execution_id=h.execution_id
            JOIN quality_current AS current
              ON current.execution_id=head.execution_id
             AND current.content_version=head.content_version
            JOIN quality_versions AS v
              ON v.execution_id=current.execution_id
             AND v.content_version=current.content_version
             AND v.decision_sha256=current.decision_sha256
            LEFT JOIN psu_pool_annotation_current AS a
              ON a.execution_id=head.execution_id
             AND a.content_version=head.content_version
            LEFT JOIN psu_pool_annotations AS av
              ON av.execution_id=a.execution_id
             AND av.content_version=a.content_version
             AND av.annotation_sha256=a.annotation_sha256
            LEFT JOIN psu_pool_difficulty_versions AS d
              ON d.cache_key=av.cache_key
             AND d.assessment_sha256=av.assessment_sha256
            WHERE h.namespace=? AND h.data_version=?
        """
        parameters = [namespace, data_version]

        if members_only:
            sql += """
                AND EXISTS (
                    SELECT 1
                    FROM psu_pool_members AS member
                    WHERE member.namespace=h.namespace
                      AND member.data_version=h.data_version
                      AND member.execution_id=h.execution_id
                      AND member.content_version=head.content_version
                )
            """

        if execution_id is not None:
            sql += " AND h.execution_id=?"
            parameters.append(execution_id)

        sql += " ORDER BY h.execution_id"

        result = []
        for row in connection.execute(sql, parameters).fetchall():
            key, version, quality_sha, quality_json, annotation_sha, group_json, difficulty_json = row
            identity = {
                "execution_id": key,
                "content_version": version,
                "namespace": namespace,
                "data_version": data_version,
            }
            difficulty = None
            if difficulty_json is not None:
                difficulty = DifficultyResult(
                    **identity,
                    assessment=_decode_assessment(difficulty_json),
                    cache_hit=True,
                    usage=DifficultyUsage()
                )
            result.append({
                "identity": identity,
                "quality": _decode_quality(quality_json),
                "quality_sha256": quality_sha,
                "annotation_sha256": annotation_sha,
                "grouping": (
                    _decode_grouping(group_json)
                    if group_json is not None else None
                ),
                "difficulty": difficulty,
            })
        return result
    
    def load_inputs(self, *, namespace: str, data_version: str) -> list[dict]:
        require_text(namespace, "namespace")
        require_text(data_version, "data_version")
        with self._connection() as connection:
            return self._read_inputs(
                connection, 
                namespace=namespace,
                data_version=data_version
            )
        
    @staticmethod
    def _candidate(entry: dict):
        identity = entry["identity"]
        return candidate_from_evaluations(
            entry["quality"],
            entry["grouping"],
            entry["difficulty"],
            namespace=identity["namespace"],
            data_version=identity["data_version"],
        )
    
    def save_annotation(
        self,
        source: dict,
        *, 
        grouping: GroupingDecision,
        difficulty: DifficultyResult
    ) -> dict:
        started = perf_counter()
        identity = source["identity"]
        grouping_json = canonical_json(asdict(grouping))
        assessment_sha = sha256_text(canonical_json(asdict(difficulty.assessment)))
        annotation_sha = sha256_text(canonical_json([
            asdict(grouping),
            difficulty.assessment.cache_key,
            assessment_sha
        ]))

        with self._connection(write=True) as connection:
            rows = self._read_inputs(
                connection,
                namespace=identity["namespace"],
                data_version=identity["data_version"],
                execution_id=identity["execution_id"]
            )

            if not rows:
                raise RuntimeError("当前内容或质量结论已变化，请重新读取")
            
            current = rows[0]
            if (
                current["identity"] != identity
                or current["quality_sha256"] != source["quality_sha256"]
                or current["quality"].admission_state != "eligible"
            ):
                raise RuntimeError("内容版本、评价或准入状态已变化，请重新读取")
            
            # 身份、范围、难度等状态检查复用已有适配函数
            candidate_from_evaluations(
                current["quality"],
                grouping,
                difficulty,
                namespace=identity["namespace"],
                data_version=identity["data_version"]
            )

            cached = self._read_cache(connection, difficulty.assessment.cache_key)
            if cached is None or sha256_text(canonical_json(asdict(cached))) != assessment_sha:
                raise RuntimeError("难度结果不是当前持久化结果")
            
            # 相同标注重复提交不新增版本，也不移除成员。
            if current["annotation_sha256"] == annotation_sha:
                return {
                    "changed": False,
                    "annotation_sha256": annotation_sha,
                    "pool_removed_count": 0,
                    "usage": _local_usage(started),
                }
            
            # 防止较早开始的任务覆盖后来保存的人工调整。
            if (
                current["annotation_sha256"]
                != source["annotation_sha256"]
            ):
                raise RuntimeError("标注已被其他操作更新，请重新读取")
            
            previous_grouping = current["grouping"]
            if (
                previous_grouping is not None
                and previous_grouping.method == "manual"
                and grouping.method != "manual"
            ):
                raise RuntimeError("已有人工分类，不能被普通规则结果覆盖")
            
            connection.execute(
                """
                INSERT OR IGNORE INTO psu_pool_annotations (
                    execution_id, content_version, annotation_sha256,
                    grouping_json, cache_key, assessment_sha256,
                    quality_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identity["execution_id"],
                    identity["content_version"],
                    annotation_sha,
                    grouping_json,
                    difficulty.assessment.cache_key,
                    assessment_sha,
                    current["quality_sha256"],
                    _now(),
                ),
            )
            connection.execute(
                """
                INSERT INTO psu_pool_annotation_current (
                    execution_id, content_version, annotation_sha256
                ) VALUES (?, ?, ?)
                ON CONFLICT(execution_id, content_version)
                DO UPDATE SET
                    annotation_sha256=excluded.annotation_sha256
                """,
                (
                    identity["execution_id"],
                    identity["content_version"],
                    annotation_sha,
                ),
            )

            # 标注变化后先移除旧成员，随后由 sync() 重新选择。
            removed = invalidate_pool_members(
                connection,
                identity["execution_id"],
                keep_version=None,
                reason="annotation_changed",
                record_runs=self.record_runs,
            )
            return {
                "changed": True,
                "annotation_sha256": annotation_sha,
                "pool_removed_count": removed,
                "usage": _local_usage(started),
            }
        
    @staticmethod
    def _read_config(connection, namespace, data_version):
        row = connection.execute(
            """
            SELECT config_json FROM psu_pool_scopes
            WHERE namespace=? AND data_version=?
            """,
            (namespace, data_version),
        ).fetchone()
        return PoolConfig(**json.loads(row[0])) if row else PoolConfig()

    @staticmethod
    def _member_rows(
        connection,
        scope: PoolScope,
        *,
        include_position: bool = False,
    ):
        columns = "execution_id, content_version, annotation_sha256"
        if include_position:
            columns += ", position"

        return connection.execute(
            f"""
            SELECT {columns}
            FROM psu_pool_members
            WHERE namespace=? AND data_version=? AND group_id=?
            ORDER BY position, execution_id, content_version
            """,
            (scope.namespace, scope.data_version, scope.group_id),
        ).fetchall()

    def _read_active_members(
        self,
        connection,
        *,
        namespace: str,
        data_version: str,
        group_id=None,
    ) -> list[dict]:
        """在调用方事务中读取有效成员；两个公开入口共用此判断。"""

        config = self._read_config(
            connection, namespace, data_version
        )
        groups = GROUP_IDS if group_id is None else (group_id,)
        enabled_groups = tuple(
            group
            for group in groups
            if config.capacity_for(group) > 0
        )

        if not enabled_groups:
            return []

        entries = self._read_inputs(
            connection,
            namespace=namespace,
            data_version=data_version,
            members_only=True,
        )
        by_key = {
            ContentKey(
                entry["identity"]["execution_id"],
                entry["identity"]["content_version"],
            ): entry
            for entry in entries
        }

        result = []

        for group in enabled_groups:
            scope = PoolScope(namespace, data_version, group)
            capacity = config.capacity_for(group)
            active_count = 0

            for execution_id, version, annotation_sha, position in (
                self._member_rows(
                    connection,
                    scope,
                    include_position=True,
                )
            ):
                key = ContentKey(execution_id, version)
                entry = by_key.get(key)

                # 过期内容不会出现在当前版本读取结果中。
                if entry is None:
                    continue

                # 成员必须仍指向当前标注。
                if annotation_sha != entry["annotation_sha256"]:
                    continue

                # 复用已有评价投影，不重新评价或重新判断难度。
                candidate = self._candidate(entry)

                if (
                    candidate.admission_state != "eligible"
                    or candidate.group_id != group
                    or candidate.difficulty is None
                ):
                    continue

                result.append({
                    **entry,
                    "candidate": candidate,
                    "position": position,
                })
                active_count += 1

                if active_count >= capacity:
                    break

        return result

    def load_active_members(
        self,
        *,
        namespace: str,
        data_version: str,
    ) -> list[dict]:
        """返回同一个只读事务中的有效成员和当前经验内容。"""

        require_text(namespace, "namespace")
        require_text(data_version, "data_version")

        with self.connection() as connection:
            return self._read_active_members(
                connection,
                namespace=namespace,
                data_version=data_version,
            )

    def active_snapshot(self, scope: PoolScope) -> PoolSnapshot:
        with self.connection() as connection:
            members = self._read_active_members(
                connection,
                namespace=scope.namespace,
                data_version=scope.data_version,
                group_id=scope.group_id,
            )

            return PoolSnapshot(
                scope=scope,
                members=tuple(
                    member["candidate"].key
                    for member in members
                ),
            )
        
    def sync(
        self,
        *,
        namespace: str,
        data_version: str,
        config: PoolConfig | None = None,
        ranking_policy=None,
        selection_policy=None,
    ) -> dict:
        """在同一事务中读取最新状态、计算选择计划并保存成员。"""
        started = perf_counter()
        require_text(namespace, "namespace")
        require_text(data_version, "data_version")
        if config is not None and not isinstance(config, PoolConfig):
            raise TypeError("config 必须是 PoolConfig")

        with self._connection(write=True) as connection:
            effective_config = (
                config if config is not None
                else self._read_config(connection, namespace, data_version)
            )
            entries = self._read_inputs(
                connection,
                namespace=namespace,
                data_version=data_version,
            )
            candidates = tuple(self._candidate(item) for item in entries)
            by_key = {
                candidate.key: entry
                for candidate, entry in zip(candidates, entries)
            }
            heads = dict(connection.execute(
                """
                SELECT head.execution_id, head.content_version
                FROM quality_heads AS head
                JOIN history_records AS h
                  ON h.execution_id=head.execution_id
                WHERE h.namespace=? AND h.data_version=?
                """,
                (namespace, data_version),
            ).fetchall())

            plans = []
            for group_id in GROUP_IDS:
                scope = PoolScope(namespace, data_version, group_id)
                snapshot = PoolSnapshot(
                    scope=scope,
                    members=tuple(
                        ContentKey(execution_id, version)
                        for execution_id, version, _ in self._member_rows(
                            connection, scope
                        )
                    ),
                )
                plans.append(plan_group_update(
                    snapshot=snapshot,
                    candidates=candidates,
                    current_versions=heads,
                    capacity=effective_config.capacity_for(group_id),
                    ranking_policy=ranking_policy,
                    selection_policy=selection_policy,
                ))

            # 先处理所有组的移除，再处理新增，支持组间移动。
            for plan in plans:
                for exclusion in plan.removed:
                    connection.execute(
                        """
                        DELETE FROM psu_pool_members
                        WHERE namespace=? AND data_version=? AND group_id=?
                          AND execution_id=? AND content_version=?
                        """,
                        (
                            namespace,
                            data_version,
                            plan.scope.group_id,
                            exclusion.key.execution_id,
                            exclusion.key.content_version,
                        ),
                    )

            for plan in plans:
                for position, key in enumerate(plan.selected):
                    entry = by_key[key]
                    connection.execute(
                        """
                        INSERT INTO psu_pool_members (
                            namespace, data_version, group_id,
                            execution_id, content_version,
                            annotation_sha256, position, joined_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(namespace, data_version, execution_id)
                        DO UPDATE SET
                            group_id=excluded.group_id,
                            content_version=excluded.content_version,
                            annotation_sha256=excluded.annotation_sha256,
                            position=excluded.position
                        WHERE group_id<>excluded.group_id
                           OR content_version<>excluded.content_version
                           OR annotation_sha256<>excluded.annotation_sha256
                           OR position<>excluded.position
                        """,
                        (
                            namespace,
                            data_version,
                            plan.scope.group_id,
                            key.execution_id,
                            key.content_version,
                            entry["annotation_sha256"],
                            position,
                            _now(),
                        ),
                    )

            connection.execute(
                """
                INSERT INTO psu_pool_scopes(
                    namespace, data_version, config_json
                ) VALUES (?, ?, ?)
                ON CONFLICT(namespace, data_version)
                DO UPDATE SET config_json=excluded.config_json
                WHERE config_json<>excluded.config_json
                """,
                (
                    namespace,
                    data_version,
                    canonical_json(effective_config.to_dict()),
                ),
            )

            report = _pool_event(connection, {
                "operation": "sync",
                "namespace": namespace,
                "data_version": data_version,
                "config": effective_config.to_dict(),
                "input_versions": [
                    {
                        **item["identity"],
                        "quality_sha256": item["quality_sha256"],
                        "annotation_sha256": item["annotation_sha256"],
                    }
                    for item in entries
                ],
                "plans": [plan.to_dict() for plan in plans],
                "added_count": sum(len(plan.added) for plan in plans),
                "removed_count": sum(len(plan.removed) for plan in plans),
                "retained_count": sum(len(plan.retained) for plan in plans),
                "usage": _local_usage(started),
            }, record_runs=self.record_runs)

        return report
    
    def get(self, key: str) -> DifficultyAssessment | None:
        require_text(key, "cache_key")
        with self._connection() as connection:
            return self._read_cache(connection=connection, key=key)
        
    def put(self, assessment: DifficultyAssessment) -> None:
        if not isinstance(assessment, DifficultyAssessment):
            raise TypeError("assessment 数据格式错误")
        payload = canonical_json(asdict(assessment))
        fingerprint = sha256_text(payload)

        with self._connection(write=True) as connection:
            current = self._read_cache(
                connection, assessment.cache_key
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO psu_pool_difficulty_versions (
                    cache_key,
                    assessment_sha256,
                    assessment_json,
                    created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    assessment.cache_key,
                    fingerprint,
                    payload,
                    datetime.now(timezone.utc).isoformat()
                )
            )

            # 同一缓存键的成功结果保持稳定。
            # 并发返回的其他结果仍留档，但不覆盖它。
            if current is not None and current.status == "assessed":
                return
            
            connection.execute(
                """
                INSERT INTO psu_pool_difficulty_current (
                    cache_key, assessment_sha256
                ) VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    assessment_sha256 = excluded.assessment_sha256
                WHERE assessment_sha256 != excluded.assessment_sha256
                """,
                (assessment.cache_key, fingerprint),
            )
    
    def info(self) -> dict:
        """读取组件状态"""
        with self._connection() as connection:
            keys = connection.execute(
                "SELECT COUNT(*) FROM psu_pool_difficulty_current"
            ).fetchone()[0]

            versions = connection.execute(
                "SELECT COUNT(*) FROM psu_pool_difficulty_versions"
            ).fetchone()[0]

        return {
            "database": str(self.path),
            "history_schema_version": 1,
            "pool_schema_version": POOL_SCHEMA_VERSION,
            "cache_keys": keys,
            "cache_versions": versions,
        }
    
    def save_run_report(self, report: dict) -> dict:
        with self._connection(write=True) as connection:
            return _pool_event(connection=connection, report=report, record_runs=self.record_runs)
        
