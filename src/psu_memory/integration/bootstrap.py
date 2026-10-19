"""从空候选、空成员及空检索缓存创建工作库；导入模块时不执行初始化。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from time import perf_counter

from src.project_paths import resolve_project_path
from ..evaluation.quality_store import SQLiteQualityStore
from ..history.history_store import SQLiteHistoryStore
from ..pool._common import require_text
from ..pool.pool_models import PoolConfig
from ..pool.pool_store import SQLitePoolStore
from ..retrieval.index_store import SQLiteRetrievalIndex
from ..retrieval.retrieval_models import RetrievalConfig
from ..retrieval.vector_cache import SQLiteVectorCache
from .common import zero_cost


_COLD_START_COMPONENT = "psu_memory_cold_start"
_COLD_START_VERSION = 1
_EMPTY_TABLES = (
    "history_records", "psu_pool_members", "psu_retrieval_vectors",
    "psu_retrieval_index_members",
)


def require_cold_start_store(pool_store) -> None:
    """恢复时确认该库由空池建立；不要求后续增长后的记录仍为零。"""
    with pool_store.connection() as connection:
        SQLiteVectorCache._check_schema(
            connection, component=_COLD_START_COMPONENT,
            expected_version=_COLD_START_VERSION, required=True,
        )


def initialize_empty_store(store_path, *, namespace, data_version,
                           retrieval_config: RetrievalConfig,
                           pool_config: PoolConfig, record_runs=True) -> dict:
    """仅在宿主显式调用时创建空工作库，目标已存在时拒绝覆盖。

    所有组件表、容量、缓存和索引注册均由现有存储接口建立。临时库完成
    后在同目录原子发布，不读取旧候选、日志、向量或索引，不初始化模型。
    """
    require_text(namespace, "namespace")
    require_text(data_version, "data_version")
    if not isinstance(retrieval_config, RetrievalConfig):
        raise TypeError("retrieval_config 必须是 RetrievalConfig")
    if not isinstance(pool_config, PoolConfig):
        raise TypeError("pool_config 必须是 PoolConfig")
    requested = Path(store_path).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("冷启动工作库已存在，拒绝重置或覆盖")
    target = resolve_project_path(store_path)
    if target.exists() or target.is_symlink():
        raise FileExistsError("冷启动工作库已存在，拒绝重置或覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + target.name + ".empty-", suffix=".sqlite3", dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    started = perf_counter()
    try:
        SQLiteHistoryStore(temporary, record_runs=record_runs).import_batch(
            [], namespace=namespace, data_version=data_version,
            source_root=str(target.parent), scanned_files=0, errors=[],
            started_at=started,
        )
        SQLiteQualityStore(temporary, resolve_path=resolve_project_path, record_runs=record_runs).initialize()
        pool = SQLitePoolStore(temporary, record_runs=record_runs)
        pool.initialize()
        pool.sync(namespace=namespace, data_version=data_version, config=pool_config)
        index = SQLiteRetrievalIndex(SQLiteVectorCache(pool), retrieval_config)
        # index.initialize 已初始化向量缓存；不调用编码器或离线索引构建。
        index.initialize()
        with pool.connection(write=True) as connection:
            raw_table_counts = {
                name: connection.execute('SELECT COUNT(*) FROM "' + name + '"').fetchone()[0]
                for name in _EMPTY_TABLES
            }
            if any(raw_table_counts.values()):
                raise ValueError("冷启动初始化产生了经验、成员、向量或索引记录")
            connection.execute(
                "INSERT INTO psu_retrieval_meta(component, schema_version) VALUES (?, ?)",
                (_COLD_START_COMPONENT, _COLD_START_VERSION),
            )
        require_cold_start_store(pool)
        inspection = index.inspect(namespace=namespace, data_version=data_version)
        if not inspection["complete"] or inspection["member_count"] != 0:
            raise ValueError("空经验池索引未就绪")
        # 同目录硬链接原子占用目标名称；已存在的目标不会被覆盖。
        # 所有 SQLite 连接已关闭，随后删除临时链接即完成文件移入。
        os.link(temporary, target)
        return {
            "operation": "initialize_empty_store",
            "store_path": str(target), "namespace": namespace,
            "data_version": data_version,
            "cold_start_component": _COLD_START_COMPONENT,
            "cold_start_version": _COLD_START_VERSION,
            "initial_counts": {
                "records": raw_table_counts["history_records"],
                "members": raw_table_counts["psu_pool_members"],
                "vectors": raw_table_counts["psu_retrieval_vectors"],
                "index_bindings": raw_table_counts["psu_retrieval_index_members"],
            },
            "raw_table_counts": raw_table_counts,
            "pool_config": pool_config.to_dict(),
            "index_ready": inspection["complete"],
            "usage": zero_cost(perf_counter() - started),
        }
    finally:
        # 只清理本次创建的临时文件；不删除已发布库或任何已有工作库。
        for path in (temporary, Path(str(temporary) + "-journal"),
                     Path(str(temporary) + "-wal"), Path(str(temporary) + "-shm")):
            path.unlink(missing_ok=True)
