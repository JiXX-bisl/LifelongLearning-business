"""同步查询运行层：配置、独立工作库、资源复用和可选经验更新。

query_adapter.py 与 CLI 的输入、提示词交付和最终输出接线由用户教学部署。
本模块不定义第二套检索、校验、准入或成员选择规则。
"""
from __future__ import annotations

import fcntl
import importlib
import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.project_paths import PROJECT_ROOT, resolve_project_path
from ..pool._common import require_nonnegative_int, require_text
from .common import (
    COST_FIELDS, add_costs, business_cost, copy_database, digest,
    model_options, read_json, zero_cost,
)


DEFAULT_CONFIG = "configs/business/psu_query.json"


def load_query_config(path=DEFAULT_CONFIG):
    """读取接入配置；关闭组件时不要求历史库、索引或模型文件可用。"""
    config = read_json(resolve_project_path(path))
    if not isinstance(config, dict):
        raise ValueError("PSU 接入配置必须是 JSON 对象")
    config = deepcopy(config)
    for name in ("namespace", "data_version"):
        require_text(config.get(name), name)
    initial_state = config.setdefault("initial_state", "empty")
    if initial_state not in ("empty", "snapshot"):
        raise ValueError("initial_state 只支持 empty 或 snapshot")
    for name in ("business_config", "store_path", "retrieval_config", "pool_config"):
        require_text(config.get(name), name)
        config[name] = str(resolve_project_path(config[name]))
    source = config.get("source_store")
    if initial_state == "empty":
        if source is not None:
            raise ValueError("空经验库启动的 source_store 必须为 null，不能配置历史输入库")
        config["source_store"] = None
    else:
        require_text(source, "source_store")
        config["source_store"] = str(resolve_project_path(source))
    for name in ("psu_enabled", "update_enabled"):
        value = config.setdefault(name, False)
        if type(value) is not bool:
            raise ValueError(f"{name} 必须是布尔值")
    root = config.get("evaluation_root")
    if root is not None:
        require_text(root, "evaluation_root")
        config["evaluation_root"] = str(resolve_project_path(root))
    limit = config.setdefault("retrieval_limit", 1)
    require_nonnegative_int(limit, "retrieval_limit")
    if limit not in (0, 1):
        raise ValueError("本轮 retrieval_limit 只支持 0 或 1")
    budget = config.setdefault("max_context_chars", None)
    if budget is not None:
        require_nonnegative_int(budget, "max_context_chars")
    return config


def _query_factory(psu_enabled, provided):
    if provided is not None:
        if not callable(provided):
            raise TypeError("query_service_factory 必须可调用")
        return provided, True
    module_name = "src.psu_memory.integration.query_adapter"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        if psu_enabled:
            raise RuntimeError(
                "关键接入 query_adapter.py 尚未教学部署，不能启用 PSU。"
            ) from None
        # 支撑部署期间保持关闭条件可使用原业务查询服务。
        from src.chatflow_agent.query import QueryService
        return QueryService, False
    return module.PSUQueryService, True


def _protect_working_store(config):
    store = Path(config["store_path"])
    source = Path(config["source_store"]) if config["source_store"] is not None else None
    derived = (PROJECT_ROOT / "data/derived/psu_memory").resolve()
    if not store.is_relative_to(derived):
        raise ValueError("工作库必须位于 data/derived/psu_memory 内")
    if store == source or store == derived / "history.sqlite3":
        raise ValueError("工作库不能是来源库或正式初始候选库")
    for key in ("business_config", "retrieval_config", "pool_config"):
        path = Path(config[key])
        if path == store:
            raise ValueError("工作库与配置输入路径冲突")
        if not path.is_file():
            raise FileNotFoundError(path)
    return store, source


def _pending_update_folders(config):
    expected = {name: config[name] for name in ("store_path", "namespace", "data_version")}
    pending = []
    directory = Path(config["store_path"]).parent / "updates"
    for path in sorted(directory.glob("*/update.json")):
        journal = read_json(path)
        if any(name not in journal for name in expected):
            raise ValueError("更新记录缺少工作库或隔离范围，请核对该 update.json")
        if (all(journal[name] == value for name, value in expected.items())
                and journal.get("status") != "completed"):
            pending.append(path.parent)
    return pending


class QueryRuntime:
    """一个同步进程持续复用资源；每题只触发一次查询后更新。"""

    def __init__(self, service, *, config, psu_enabled, update_enabled,
                 retriever=None, updater=None, store_lock=None, setup_seconds=0.0,
                 initialization=None):
        self.service = service
        self.config = deepcopy(config)
        self.psu_enabled = psu_enabled
        self.update_enabled = update_enabled
        self.retriever = retriever
        self.updater = updater
        self.setup_seconds = setup_seconds
        self.initialization = deepcopy(initialization)
        self._store_lock = store_lock
        self._closed = False
        self._busy = False
        self._query_parts = []
        self._update_parts = []
        self.last_usage = None
        self.last_update = None
        self._pending_records = {}
        self._pending_evaluation_roots = {}
        self._pending_refresh = set()
        self.pending_updates = self._pending_folders() if update_enabled else []

    @property
    def condition(self):
        return "baseline" if not self.psu_enabled else (
            "adaptive" if self.update_enabled else "fixed")

    @property
    def update_directory(self):
        return Path(self.config["store_path"]).parent / "updates"

    @property
    def run_usage(self):
        query = add_costs(self._query_parts)
        update = add_costs(self._update_parts)
        return {"query": query, "update": update,
                "total": add_costs([query, update]), "setup_seconds": self.setup_seconds}

    def _pending_folders(self):
        return _pending_update_folders(self.config)

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("查询运行层已关闭")
        if self._busy:
            raise RuntimeError("本轮运行层仅支持同步顺序查询")

    def _update_folder(self, record_path):
        identity = digest([self.config["store_path"], self.config["namespace"], self.config["data_version"],
                           str(resolve_project_path(record_path))])
        return self.update_directory / identity

    def _apply_update(self, record, *, retry_incomplete=False, folder=None,
                      refresh=False, evaluation_root=None):
        record_path = resolve_project_path(record["record_path"])
        folder = self._update_folder(record_path) if folder is None else Path(folder)
        source = evaluation_root if evaluation_root is not None else self.config.get("evaluation_root")
        # .../<mode>/sessions/<session_id>/<turn>.json：复用已有反馈目录。
        if source is None:
            source = record_path.parents[2]
        started = perf_counter()
        self._pending_records[folder] = str(record_path)
        self._pending_evaluation_roots[folder] = source
        if refresh:
            self._pending_refresh.add(folder)
        try:
            operation = self.updater.refresh if refresh else self.updater.apply
            report = operation(
                record=record, record_path=record_path, folder=folder,
                evaluation_root=source, retry_incomplete=retry_incomplete,
            )
        except Exception as error:
            # 本题业务记录已保存；更新失败不能让调用方丢失已有回答。
            report = {"status": "update_failed", "error_type": type(error).__name__,
                      "record_path": str(record_path), "folder": str(folder),
                      "retry_required": True}
            if folder not in self.pending_updates:
                self.pending_updates.append(folder)
        except BaseException:
            if folder not in self.pending_updates:
                self.pending_updates.append(folder)
            usage = deepcopy(self.updater.last_usage)
            usage["elapsed_seconds"] = perf_counter() - started
            usage["reported_elapsed_seconds"] = usage["elapsed_seconds"]
            self._update_parts.append(usage)
            self.last_update = {
                "status": "interrupted", "record_path": str(record_path),
                "folder": str(folder), "retry_required": True,
                "current_call_usage": usage,
            }
            self.last_usage = {"condition": self.condition, "query": zero_cost(),
                               "update": usage, "total": add_costs([usage]),
                               "status": "update_interrupted"}
            raise
        usage = deepcopy(self.updater.last_usage)
        usage["elapsed_seconds"] = perf_counter() - started
        usage["reported_elapsed_seconds"] = usage["elapsed_seconds"]
        self._update_parts.append(usage)
        report["current_call_usage"] = usage
        if report.get("status") == "completed":
            self.pending_updates = [path for path in self.pending_updates if path != folder]
            self._pending_records.pop(folder, None)
            self._pending_evaluation_roots.pop(folder, None)
            self._pending_refresh.discard(folder)
        self.last_update = report
        return report, usage

    def query(self, question, *, on_progress=None, **query_options):
        """调用已部署的查询适配；成本保存在运行层，不重写业务日志。"""
        self._ensure_open()
        if self.pending_updates:
            raise RuntimeError("有未完成的经验更新，请先显式 retry_update() 后继续提问")
        self._busy = True
        started = perf_counter()
        try:
            try:
                result = self.service.query(question, on_progress=on_progress, **query_options)
            except BaseException:
                # 原业务入口异常时未公开完整回包统计，不能漏记本题或沿用上一题。
                unknown = {**zero_cost(), **{name: None for name in COST_FIELDS},
                           "elapsed_seconds": perf_counter() - started,
                           "token_usage_complete": False}
                if not self.psu_enabled:
                    unknown["embedding_input_tokens"] = 0
                cost = add_costs([unknown])
                self._query_parts.append(cost)
                self.last_update = {"status": "not_started", "usage": zero_cost()}
                self.last_usage = {"condition": self.condition, "query": cost,
                                   "update": zero_cost(), "total": add_costs([cost]),
                                   "status": "query_failed_or_interrupted"}
                raise
            query_cost = add_costs([
                business_cost(result["metrics"]),
                result.get("psu", {}).get("usage", zero_cost()),
            ])
            query_cost["elapsed_seconds"] = perf_counter() - started
            query_cost["reported_elapsed_seconds"] = query_cost["elapsed_seconds"]
            self._query_parts.append(query_cost)
            self.last_update = {"status": "disabled", "usage": zero_cost()}
            update_cost = zero_cost()
            if self.update_enabled:
                try:
                    _, update_cost = self._apply_update(result)
                except BaseException:
                    update_cost = self.last_update.get("current_call_usage", zero_cost())
                    total = add_costs([query_cost, update_cost])
                    total["elapsed_seconds"] = perf_counter() - started
                    total["reported_elapsed_seconds"] = total["elapsed_seconds"]
                    self.last_usage = {"condition": self.condition, "query": query_cost,
                                       "update": update_cost, "total": total,
                                       "status": "update_interrupted"}
                    raise
            total = add_costs([query_cost, update_cost])
            # 已含检索和更新，用整段外层时间替换，避免相加阶段时间重复计费。
            total["elapsed_seconds"] = perf_counter() - started
            total["reported_elapsed_seconds"] = total["elapsed_seconds"]
            self.last_usage = {"condition": self.condition, "query": query_cost,
                               "update": update_cost, "total": total}
            return result
        finally:
            self._busy = False

    def retry_update(self, folder=None):
        """仅显式重试更新；不重新提问、调用查询模型或执行业务 SQL。"""
        self._ensure_open()
        if not self.update_enabled:
            raise RuntimeError("当前未启用经验更新")
        if folder is None:
            if len(self.pending_updates) != 1:
                raise ValueError("请明确指定一个待处理的更新目录")
            folder = self.pending_updates[0]
        folder = resolve_project_path(folder)
        if folder not in self.pending_updates:
            raise ValueError("指定目录不是本运行层的待处理更新")
        journal_path = folder / "update.json"
        refresh = folder in self._pending_refresh
        if journal_path.exists():
            journal = read_json(journal_path)
            record_path = journal["record_path"]
            refresh = refresh or journal.get("refresh_active", False)
        else:
            # 读取来源记录失败时尚未形成阶段 journal，仍保留本进程的来源位置。
            record_path = self._pending_records[folder]
        record_path = resolve_project_path(record_path)
        record = {**read_json(record_path), "record_path": str(record_path)}
        self._busy = True
        try:
            report, usage = self._apply_update(record, retry_incomplete=True, folder=folder,
                                               refresh=refresh,
                                               evaluation_root=self._pending_evaluation_roots.get(folder))
            self.last_usage = {"condition": self.condition, "query": zero_cost(),
                               "update": usage, "total": add_costs([usage])}
            return report
        finally:
            self._busy = False

    def refresh_experience(self, record_path, *, evaluation_root=None):
        """评价或反馈稍后到达时显式维护原候选，不重复业务查询。"""
        self._ensure_open()
        if not self.update_enabled:
            raise RuntimeError("当前未启用经验更新")
        if self.pending_updates:
            raise RuntimeError("请先显式 retry_update() 完成待处理更新")
        record_path = resolve_project_path(record_path)
        # 调用方指定的现存文件为权威；旧日志中保存的位置可能已迁移。
        record = {**read_json(record_path), "record_path": str(record_path)}
        self._busy = True
        try:
            report, usage = self._apply_update(
                record, refresh=True, evaluation_root=evaluation_root,
            )
            self.last_usage = {"condition": self.condition, "query": zero_cost(),
                               "update": usage, "total": add_costs([usage])}
            return report
        finally:
            self._busy = False

    def reset(self):
        self._ensure_open()
        self.service.reset()

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for resource in (self.updater, self.service, self.retriever):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    errors.append(error)
        if self._store_lock is not None:
            self._store_lock.close()
            self._store_lock = None
        if self.service is not None and getattr(self.service, "retriever", None) is self.retriever:
            self.service.retriever = None
        self.updater = None
        self.retriever = None
        if errors:
            raise errors[0]

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def create_query_runtime(config_path=DEFAULT_CONFIG, *, psu_enabled=None,
                         update_enabled=None, experiment_id=None,
                         query_service_factory=None):
    """仅在用户调用启动入口时组装；导入本模块不会准备经验库或查询环境。"""
    started = perf_counter()
    config = load_query_config(config_path)
    for name, override in (("psu_enabled", psu_enabled), ("update_enabled", update_enabled)):
        if override is not None:
            if type(override) is not bool:
                raise ValueError(f"{name} 必须是布尔值")
            config[name] = override
    enabled, updating = config["psu_enabled"], config["update_enabled"]
    if updating and not enabled:
        raise ValueError("经验更新需要启用 PSU；本轮支持 baseline、fixed、adaptive 三种组合")
    factory, is_adapter = _query_factory(enabled, query_service_factory)
    service = retriever = updater = lease = None
    initialization = None
    try:
        if enabled:
            store, source = _protect_working_store(config)
            store.parent.mkdir(parents=True, exist_ok=True)
            lease = store.with_name(store.name + ".lock").open("a", encoding="utf-8")
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("该工作库已有运行进程，请使用独立工作库或等待其关闭") from None
            from ..retrieval.retrieval_models import RetrievalConfig
            from ..retrieval.service import PSURetriever
            from ..pool.pool_models import PoolConfig
            from .bootstrap import initialize_empty_store, require_cold_start_store
            retrieval_config = RetrievalConfig.from_json(config["retrieval_config"])
            pool_config = PoolConfig(**read_json(config["pool_config"]))
            if not store.exists():
                if config["initial_state"] == "empty":
                    initialization = initialize_empty_store(
                        store, namespace=config["namespace"], data_version=config["data_version"],
                        retrieval_config=retrieval_config, pool_config=pool_config,
                    )
                else:
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    temporary = store.with_name(store.name + ".preparing-" + uuid4().hex)
                    try:
                        copy_database(source, temporary)
                        os.link(temporary, store)
                    finally:
                        temporary.unlink(missing_ok=True)
            retriever = PSURetriever.from_local(
                store, retrieval_config=retrieval_config,
                model_config_path=config["business_config"],
            )
            if config["initial_state"] == "empty":
                require_cold_start_store(retriever.index.pool_store)
            inspection = retriever.index.inspect(namespace=config["namespace"],
                                                 data_version=config["data_version"])
            pending = _pending_update_folders(config) if updating else []
            if not inspection["complete"] and not (updating and pending):
                raise ValueError("工作库索引未就绪，请由已有离线构建入口处理；在线启动不修复索引")
            pool = retriever.index.pool_store
            with pool.connection() as connection:
                current_capacity = pool._read_config(connection, config["namespace"], config["data_version"])
            if current_capacity != pool_config:
                raise ValueError("工作库容量与 pool_config 不同，请先通过既有离线维护入口统一配置")
        if is_adapter:
            service = factory(
                config["business_config"], namespace=config["namespace"],
                data_version=config["data_version"], retriever=retriever,
                experiment_id=experiment_id, retrieval_limit=config["retrieval_limit"],
                max_context_chars=config["max_context_chars"],
            )
        else:
            service = factory(config["business_config"], mode="standard", experiment_id=experiment_id)
        if service.config["data"]["baseline_id"] != config["data_version"]:
            raise ValueError("业务基线与经验池 data_version 不同")
        service.effective_model_options = model_options(service.language_model, config["business_config"])
        if updating:
            from .update import ExperienceUpdater
            updater = ExperienceUpdater(
                retriever, namespace=config["namespace"], data_version=config["data_version"],
                model_config=config["business_config"], pool_config=config["pool_config"],
                schema_context="\n\n".join(service.schemas.values()),
                evaluation_root=config.get("evaluation_root"),
            )
        return QueryRuntime(
            service, config=config, psu_enabled=enabled, update_enabled=updating,
            retriever=retriever, updater=updater, store_lock=lease,
            setup_seconds=perf_counter() - started, initialization=initialization,
        )
    except BaseException:
        for resource in (updater, service, retriever):
            if resource is not None:
                try:
                    resource.close()
                except Exception:
                    pass
        if lease is not None:
            lease.close()
        raise
