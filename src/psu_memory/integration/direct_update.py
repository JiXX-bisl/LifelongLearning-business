"""直接比较通过后的成功经验更新；只写既有 SQLite 功能状态。"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from time import perf_counter

from src.project_paths import resolve_project_path
from ..evaluation.evaluation_evidence import attach_sql_result_pass
from ..evaluation.quality_store import SQLiteQualityStore
from ..history.history_adapter import history_from_query_result
from ..history.history_store import SQLiteHistoryStore
from ..pool.classification import classify
from ..pool.difficulty import DifficultyAssessor
from ..pool.difficulty_llm import LocalDifficultyClient
from ..pool.pool_models import DifficultyInput, GroupingInput, PoolConfig
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
        connection.execute("SELECT * FROM psu_retrieval_index_members ORDER BY index_key").fetchall(),
        connection.execute("SELECT * FROM history_records WHERE execution_id=?", (execution_id,)).fetchall(),
    )


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
        started = perf_counter()
        execution_id = historical.task.task_id
        imported = self.history.import_batch(
            [historical], namespace=self.namespace, data_version=self.data_version,
            source_root=str(Path(historical.task.source_ref).parent),
            scanned_files=1, errors=[], started_at=started,
        )
        if imported["errors"] or imported["conflicts"]:
            raise ValueError("本题经验导入冲突")
        attach_sql_result_pass(self.path, historical, query_id)
        evaluated = self.quality.assess_batch(
            namespace=self.namespace, data_version=self.data_version, execution_ids=(execution_id,),
        )
        if evaluated["decisions"][0]["admission_state"] != "eligible":
            raise ValueError("通过的 SQL 未获得成功经验准入")
        entry = next(item for item in self.pool.load_inputs(
            namespace=self.namespace, data_version=self.data_version)
            if item["identity"]["execution_id"] == execution_id)
        content = entry["quality"].content
        request = GroupingInput(**entry["identity"], question=content["question"],
                                context_text=content.get("context_text") or "")
        previous = entry["grouping"]
        grouping = previous if previous is not None and previous.method == "manual" else classify(request)
        if self._difficulty_client is None:
            self._difficulty_client = LocalDifficultyClient.from_yaml(self.model_config)
        difficulty = DifficultyAssessor(self._difficulty_client, cache=self.pool).assess(
            DifficultyInput.from_grouping(request, schema_context=self.schema_context), admission_state="eligible",
        )
        if difficulty.assessment.status == "failed":
            raise ValueError("本题经验的难度标注失败")
        self.pool.save_annotation(entry, grouping=grouping, difficulty=difficulty)
        self.pool.sync(namespace=self.namespace, data_version=self.data_version, config=self.config)
        index = OfflineIndexBuilder(self.retriever.index, self.retriever.prefilter.embeddings).build(
            namespace=self.namespace, data_version=self.data_version,
        )
        if index["error_count"] or not index["index_complete"]:
            raise ValueError("更新后的经验索引未就绪")
        return {"admitted": any(item["identity"]["execution_id"] == execution_id
                                for item in self.pool.load_active_members(
                                    namespace=self.namespace, data_version=self.data_version))}
