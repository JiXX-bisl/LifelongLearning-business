"""直接比较通过后的候选保存及在线经验更新。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter

from src.project_paths import resolve_project_path
from ..evaluation.evaluation_evidence import SCHEMA as EVALUATION_SCHEMA, attach_sql_result_pass
from ..evaluation.quality_store import SQLiteQualityStore
from ..history.history_adapter import history_from_query_result
from ..history.history_store import SQLiteHistoryStore
from ..pool.difficulty import DifficultyAssessor
from ..pool.difficulty_llm import LocalDifficultyClient
from ..pool.maintenance import run_pool_maintenance
from ..pool.pool_models import PoolConfig
from ..pool.pool_store import SQLitePoolStore
from ..retrieval.indexing import OfflineIndexBuilder


class DirectUpdateError(RuntimeError):
    """本题更新失败，工作库已恢复，可继续下一题。"""


class DirectUpdateIntegrityError(RuntimeError):
    """无法确认工作库完整性，调用方必须停止该组。"""


def _functional_rows(connection, execution_id):
    """仅供本次内存回滚核对，不落盘。"""
    return (
        connection.execute("SELECT * FROM psu_pool_members ORDER BY namespace, data_version, execution_id").fetchall(),
        connection.execute("SELECT * FROM psu_pool_scopes ORDER BY namespace, data_version").fetchall(),
        connection.execute("SELECT * FROM psu_pool_fifo_seen ORDER BY namespace, data_version, execution_id, content_version").fetchall(),
        connection.execute("SELECT * FROM psu_retrieval_index_members ORDER BY index_key").fetchall(),
        connection.execute("SELECT * FROM history_records WHERE execution_id=?", (execution_id,)).fetchall(),
    )


def _save_candidate(history, quality, historical, query_id):
    """只保存实际查询、直接 PASS 证据及准入结论，不维护池或索引。"""
    task = historical.task
    imported = history.import_batch(
        [historical], namespace=task.namespace, data_version=task.data_version,
        source_root=str(Path(task.source_ref).parent),
        scanned_files=1, errors=[], started_at=perf_counter(),
    )
    if imported["errors"] or imported["conflicts"]:
        raise ValueError("本题经验导入冲突")
    attach_sql_result_pass(history.path, historical, query_id)
    evaluated = quality.assess_batch(
        namespace=task.namespace, data_version=task.data_version,
        execution_ids=(task.task_id,),
    )
    decisions = evaluated["decisions"]
    if (len(decisions) != 1 or decisions[0]["admission_state"] != "eligible"
            or decisions[0].get("direct_sql_result_verified") is not True):
        raise ValueError("通过的 SQL 未获得直接比较成功经验准入")
    return {"execution_id": task.task_id, "candidate_saved": True}


class DirectCandidateCollector:
    """独立形成阶段的无限容量候选库；宿主独占串行使用并仅提交当前 PASS。"""

    def __init__(self, path, namespace, data_version, *, initialize=True):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace 不能为空")
        if not isinstance(data_version, str) or not data_version.strip():
            raise ValueError("data_version 不能为空")
        self.path = Path(path).resolve()
        self.namespace, self.data_version = namespace, data_version
        self.history = SQLiteHistoryStore(self.path, record_runs=False)
        self.quality = SQLiteQualityStore(
            self.path, resolve_path=resolve_project_path, record_runs=False,
        )
        if not initialize:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            return
        # 即使整轮没有 PASS，空候选库仍能给出确定的零候选汇总。
        # 空批初始化不调用模型，也不写 import_runs / quality_runs。
        self.history.import_batch(
            [], namespace=namespace, data_version=data_version,
            source_root=str(self.path.parent), scanned_files=0, errors=[],
            started_at=perf_counter(),
        )
        self.quality.initialize()
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            with connection:
                for statement in EVALUATION_SCHEMA:
                    connection.execute(statement)

    def close(self):
        """各次操作自行关闭连接，无常驻模型、检索器或数据库连接。"""

    def apply(self, case, record, record_path, query_id):
        try:
            historical = history_from_query_result(
                case, record, record_path, query_id,
                namespace=self.namespace, data_version=self.data_version,
            )
        except Exception as error:
            raise DirectUpdateError("本题经验来源无效，未开始保存候选") from error
        # 三个既有组件各自提交事务；整体失败时还原内存快照，撤销半候选。
        with closing(sqlite3.connect(":memory:")) as before:
            try:
                with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as source:
                    source.backup(before)
                expected_state = tuple(before.iterdump())
            except Exception as error:
                raise DirectUpdateIntegrityError("无法建立候选保存前的内存快照，停止形成") from error
            try:
                return _save_candidate(self.history, self.quality, historical, query_id)
            except BaseException as error:
                try:
                    with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True)) as target:
                        before.backup(target)
                        if tuple(target.iterdump()) != expected_state:
                            raise ValueError("候选、来源、证据或准入状态未完全恢复")
                except Exception as recovery_error:
                    raise DirectUpdateIntegrityError("候选保存失败且恢复未能确认，停止形成") from recovery_error
                if not isinstance(error, Exception):
                    raise
                raise DirectUpdateError("本题候选保存失败，已撤销本次保存") from error

    def summary(self):
        """只读取候选结论；不重新评价、不加载私有标准、不触发模型。"""
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as connection:
            rows = connection.execute(
                "SELECT h.execution_id, h.record_json, v.decision_json "
                "FROM history_records h JOIN quality_current c ON c.execution_id=h.execution_id "
                "JOIN quality_versions v ON v.execution_id=c.execution_id "
                "AND v.content_version=c.content_version AND v.decision_sha256=c.decision_sha256 "
                "WHERE h.namespace=? AND h.data_version=? ORDER BY h.rowid",
                (self.namespace, self.data_version),
            ).fetchall()
        candidates = [(execution_id, json.loads(record)["task"]["source_ref"])
                      for execution_id, record, raw_decision in rows
                      for decision in [json.loads(raw_decision)]
                      if decision["admission_state"] == "eligible"
                      and decision.get("direct_sql_result_verified") is True]
        return {
            "candidate_count": len(candidates),
            "execution_ids": [candidate[0] for candidate in candidates],
            "source_refs": [candidate[1] for candidate in candidates],
        }


class DirectExperienceUpdater:
    """宿主负责先写不可变 PASS 结果；每组工作库必须由该宿主独占串行使用。"""

    def __init__(self, retriever, namespace, data_version, model_config, pool_config, schema_context=""):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace 不能为空")
        if not isinstance(data_version, str) or not data_version.strip():
            raise ValueError("data_version 不能为空")
        self.retriever = retriever
        self.path = Path(retriever.index.pool_store.path).resolve()
        self.namespace, self.data_version = namespace, data_version
        self.model_config = model_config
        self.config = (pool_config if isinstance(pool_config, PoolConfig) else
                       PoolConfig(**json.loads(resolve_project_path(pool_config).read_text(encoding="utf-8"))))
        self.schema_context = schema_context
        self.pool = SQLitePoolStore(self.path, record_runs=False)
        self.history = SQLiteHistoryStore(self.path, record_runs=False)
        self.quality = SQLiteQualityStore(self.path, resolve_path=resolve_project_path, record_runs=False)
        self._difficulty_client = None

    def close(self):
        if self._difficulty_client is not None:
            self._difficulty_client.close()
            self._difficulty_client = None

    def apply(self, case, record, record_path, query_id):
        try:
            historical = history_from_query_result(
                case, record, record_path, query_id,
                namespace=self.namespace, data_version=self.data_version,
            )
        except Exception as error:
            raise DirectUpdateError("本题经验来源无效，未开始更新") from error
        execution_id = historical.task.task_id
        # SQLite backup 的目标只有内存；失败恢复同时撤销候选、成员淘汰及索引发布。
        # 不保留阶段流水，也不支持从一次中断中的部分更新续跑。
        with closing(sqlite3.connect(":memory:")) as before:
            try:
                with closing(sqlite3.connect(Path(self.path).as_uri() + "?mode=ro", uri=True)) as source:
                    source.backup(before)
                expected_state = _functional_rows(before, execution_id)
            except Exception as error:
                raise DirectUpdateIntegrityError("无法建立经验更新前的内存快照，停止该组") from error
            try:
                result = self._apply(historical, query_id)
            except BaseException as error:
                try:
                    with closing(sqlite3.connect(Path(self.path).as_uri() + "?mode=rw", uri=True)) as target:
                        before.backup(target)
                        if _functional_rows(target, execution_id) != expected_state:
                            raise ValueError("成员、索引或本题候选未恢复")
                    if not self.retriever.index.inspect(
                            namespace=self.namespace, data_version=self.data_version)["complete"]:
                        raise ValueError("恢复后的索引不完整")
                except Exception as recovery_error:
                    raise DirectUpdateIntegrityError("经验更新失败且工作库恢复未能确认，停止该组") from recovery_error
                if not isinstance(error, Exception):
                    raise
                raise DirectUpdateError("本题经验更新失败，已撤销本次更新") from error
            return result

    def _apply(self, historical, query_id):
        execution_id = historical.task.task_id
        _save_candidate(self.history, self.quality, historical, query_id)
        if self._difficulty_client is None:
            self._difficulty_client = LocalDifficultyClient.from_yaml(self.model_config)
        run_pool_maintenance(
            quality_store=self.quality, pool_store=self.pool,
            namespace=self.namespace, data_version=self.data_version,
            assessor=DifficultyAssessor(self._difficulty_client, cache=self.pool),
            config=self.config, schema_context=self.schema_context,
            execution_ids=(execution_id,), strict=True, refresh_quality=False,
        )
        index = OfflineIndexBuilder(self.retriever.index, self.retriever.prefilter.embeddings).build(
            namespace=self.namespace, data_version=self.data_version,
        )
        if index["error_count"] or not index["index_complete"]:
            raise ValueError("更新后的经验索引未就绪")
        return {"admitted": any(item["identity"]["execution_id"] == execution_id
                                for item in self.pool.load_active_members(
                                    namespace=self.namespace, data_version=self.data_version))}
