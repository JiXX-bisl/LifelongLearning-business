"""通用查询后更新协调；沿用业务记录、评价证据与既有准入规则。"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from src.project_paths import resolve_project_path
from ..evaluation.evaluation_evidence import _read_signals, attach_evaluations
from ..evaluation.quality_store import SQLiteQualityStore
from ..history.history_adapter import read_history_file
from ..history.history_store import SQLiteHistoryStore
from ..pool.difficulty import DifficultyAssessor
from ..pool.difficulty_llm import LocalDifficultyClient
from ..pool.maintenance import run_pool_maintenance
from ..pool.pool_models import PoolConfig
from ..retrieval.indexing import OfflineIndexBuilder
from .common import add_costs, digest, now, read_json, write_json, zero_cost


def members(retriever, namespace, data_version):
    """读取既有成员身份，供更新前后比较。"""
    return [dict(member["identity"], group_id=member["candidate"].group_id)
            for member in retriever.index.pool_store.load_active_members(
                namespace=namespace, data_version=data_version)]


class ExperienceUpdater:
    """只协调已有更新组件，不创建业务日志或生成新的审核证据。"""

    def __init__(self, retriever, *, namespace, data_version, model_config,
                 pool_config, schema_context, evaluation_root=None):
        self.retriever = retriever
        self.path = retriever.index.pool_store.path
        self.namespace = namespace
        self.data_version = data_version
        self.model_config = model_config
        self.pool_config = PoolConfig(**read_json(pool_config))
        self.schema_context = schema_context
        self.evaluation_root = (
            resolve_project_path(evaluation_root) if evaluation_root is not None else None
        )
        self.quality = SQLiteQualityStore(self.path, resolve_path=resolve_project_path)
        self._difficulty_client = None
        self.last_usage = zero_cost()

    def close(self):
        """只关闭本更新器创建的客户端；检索器由宿主统一关闭。"""
        if self._difficulty_client is not None:
            self._difficulty_client.close()
            self._difficulty_client = None

    def _journal(self, historical, record_path, folder, *, metadata=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        journal_path = folder / "update.json"
        identity = {
            "execution_id": historical.task.task_id,
            "source_sha256": historical.source_sha256,
            "namespace": self.namespace,
            "data_version": self.data_version,
            "store_path": str(Path(self.path).resolve()),
        }
        if journal_path.exists():
            journal = read_json(journal_path)
            for key, value in identity.items():
                if key in journal and journal[key] != value:
                    raise ValueError("更新记录对应的任务、来源或经验库已变化，拒绝覆盖")
                journal.setdefault(key, value)
            journal.setdefault("record_path", str(record_path))
        else:
            journal = {
                **identity, **(metadata or {}), "record_path": str(record_path),
                "created_at": now(), "events": [], "status": "running",
                "members_before": members(self.retriever, self.namespace, self.data_version),
            }
            write_json(journal_path, journal)
        return journal_path, journal

    def _refresh_state(self, historical, record_path, evidence_root):
        """摘要相关现有证据和当前质量版本；准入与绑定仍由既有接口处理。"""
        evidence = {"source_root": str(evidence_root) if evidence_root is not None else None,
                    "signals": []}
        if evidence_root is not None:
            root = Path(evidence_root)
            evidence["root_exists"] = root.is_dir()
            if root.is_dir():
                paths = sorted(set(root.glob("**/acceptance.json")) |
                               set(root.glob("**/feedback/*/*.json")))
                for path in paths:
                    try:
                        signals = _read_signals(path, root)
                    except (OSError, ValueError):
                        # 无效文件不能形成相关证据；需要关联时仍由既有接口
                        # 报告错误。已有关联文件失效时，它会从摘要中消失。
                        continue
                    for signal in signals:
                        if resolve_project_path(signal["original_record_ref"]) == record_path:
                            evidence["signals"].append(signal)
        entries = self.retriever.index.pool_store.load_inputs(
            namespace=self.namespace, data_version=self.data_version,
        )
        current = next((entry for entry in entries
                        if entry["identity"]["execution_id"] == historical.task.task_id), None)
        return {
            "evidence": evidence,
            "quality": ({"content_version": current["identity"]["content_version"],
                         "quality_sha256": current["quality_sha256"]}
                        if current is not None else None),
        }

    def _save_refresh_signature(self, journal_path, journal, input_evidence):
        # 使用实际证据阶段开始时的摘要与完成后的质量版本。若资料在执行中变化，
        # 下一次显式刷新仍能发现变化，不把尚未处理的新资料标成已处理。
        state = {
            "evidence": input_evidence if input_evidence is not None else {"untracked": True},
            "quality": {"content_version": journal["content_version"],
                        "quality_sha256": journal["quality_sha256"]},
        }
        journal["refresh_signature"] = digest(state)
        write_json(journal_path, journal)

    def apply(self, *, record, record_path, folder=None, evaluation_root=None,
              retry_incomplete=False):
        """处理一份已保存的业务记录；没有评价证据时仍按既有规则评价。

        ``usage`` 仅包含本次调用新执行的阶段；``update_usage`` 保留该任务
        包含中断、重试在内的累计成本。重复调用完成任务不会再次更新。
        后续评价或反馈仍沿用现有接口；调用 refresh() 使已保存候选重新评价。
        """
        self.last_usage = zero_cost()
        record_path = resolve_project_path(record_path)
        historical = read_history_file(record_path, namespace=self.namespace,
                                       data_version=self.data_version)
        if folder is None:
            folder = record_path.parent / "psu_updates" / historical.task.task_id
        else:
            folder = resolve_project_path(folder)
        journal_path, journal = self._journal(historical, record_path, folder)
        if journal.get("refresh_active"):
            if not retry_incomplete:
                raise ValueError("迟到评价刷新尚未完成；请显式重试更新")
            return self.refresh(record=record, record_path=record_path, folder=folder,
                                evaluation_root=evaluation_root, retry_incomplete=True)
        if journal["status"] == "completed":
            return {**journal, "replayed": True, "update_usage": journal["usage"],
                    "usage": zero_cost()}
        evidence_root = (
            resolve_project_path(evaluation_root)
            if evaluation_root is not None else self.evaluation_root
        )
        first_event = len(journal["events"])
        try:
            result = self._apply_historical(
                historical, record_path, journal_path, journal,
                evidence_root=evidence_root, retry_incomplete=retry_incomplete,
                final_status="apply_finalizing",
            )
            self._save_refresh_signature(
                journal_path, journal, journal.get("apply_input_evidence"),
            )
            journal.update(status="completed", completed_at=now())
            write_json(journal_path, journal)
        finally:
            self.last_usage = add_costs(event["usage"]
                                       for event in journal["events"][first_event:])
        return {**result, "replayed": False, "update_usage": result["usage"],
                "usage": self.last_usage}

    def refresh(self, *, record_path, record=None, folder=None, evaluation_root=None,
                retry_incomplete=False):
        """显式处理迟到证据或外部反馈，不重问、不执行 SQL、不改业务记录。"""
        self.last_usage = zero_cost()
        record_path = resolve_project_path(record_path)
        historical = read_history_file(record_path, namespace=self.namespace,
                                       data_version=self.data_version)
        if folder is None:
            folder = record_path.parent / "psu_updates" / historical.task.task_id
        else:
            folder = resolve_project_path(folder)
        if not (Path(folder) / "update.json").exists():
            return self.apply(record=record, record_path=record_path, folder=folder,
                              evaluation_root=evaluation_root, retry_incomplete=retry_incomplete)
        journal_path, journal = self._journal(historical, record_path, folder)
        evidence_root = (
            resolve_project_path(evaluation_root)
            if evaluation_root is not None else self.evaluation_root
        )
        if journal["status"] != "completed":
            if not retry_incomplete:
                raise ValueError("更新阶段未完成；显式启用 retry_incomplete 后才能刷新")
            if not journal.get("refresh_active"):
                return self.apply(record=record, record_path=record_path, folder=folder,
                                  evaluation_root=evidence_root, retry_incomplete=True)
            # 重试同一代时沿用当时的评价来源与证据摘要；新证据在下一次
            # 显式刷新中处理，已经完成的阶段不会被覆盖或重复执行。
            source = journal.get("refresh_evaluation_root")
            evidence_root = Path(source) if source is not None else None
        else:
            state = self._refresh_state(historical, record_path, evidence_root)
            if journal.get("refresh_signature") == digest(state):
                return {**journal, "replayed": True, "update_usage": journal["usage"],
                        "usage": zero_cost()}
            journal.update(
                status="running", refresh_generation=journal.get("refresh_generation", 0) + 1,
                refresh_active=True, refresh_started_at=now(),
                refresh_evaluation_root=str(evidence_root) if evidence_root is not None else None,
                refresh_input_evidence=state["evidence"],
                refresh_members_before=members(self.retriever, self.namespace, self.data_version),
            )
            write_json(journal_path, journal)
        prefix = "refresh:" + str(journal["refresh_generation"]) + ":"
        first_event = len(journal["events"])
        try:
            result = self._apply_historical(
                historical, record_path, journal_path, journal,
                evidence_root=evidence_root, retry_incomplete=retry_incomplete,
                stage_prefix=prefix,
            )
            self._save_refresh_signature(
                journal_path, journal, journal["refresh_input_evidence"],
            )
            journal.update(status="completed", refresh_active=False, completed_at=now())
            write_json(journal_path, journal)
        finally:
            self.last_usage = add_costs(event["usage"]
                                       for event in journal["events"][first_event:])
        return {**result, "replayed": False, "update_usage": result["usage"],
                "usage": self.last_usage}

    def _apply_historical(self, historical, record_path, journal_path, journal, *,
                          evidence_root, retry_incomplete=False, after_quality=None,
                          stage_prefix="", final_status=None):
        """通用与实验入口共用的阶段执行器，保留原 update.json 恢复格式。"""
        execution_id = historical.task.task_id
        journal["execution_id"] = execution_id

        def stage(name, action, *, may_call_model=False):
            name = name if name == "import" else stage_prefix + name
            previous = [event for event in journal["events"] if event["stage"] == name]
            if previous and previous[-1]["status"] == "done":
                return previous[-1]["report"]
            if previous and not retry_incomplete:
                raise ValueError("更新阶段未完成；显式启用 retry_incomplete 后才能继续")
            usage = zero_cost()
            if may_call_model:
                usage.update({key: None for key in (
                    "model_call_count", "http_request_count", "retry_count",
                    "prompt_tokens", "completion_tokens", "embedding_input_tokens",
                )})
                usage["token_usage_complete"] = False
            event = {"stage": name, "status": "started", "started_at": now(),
                     "usage": usage}
            journal["events"].append(event)
            write_json(journal_path, journal)
            started = perf_counter()
            try:
                report = action()
                returned_usage = {**zero_cost(), **report.get("usage", {})}
                # 未返回用量的模型阶段保留未知，不能由零成本默认值补齐。
                if may_call_model:
                    returned_usage = {**usage, **report.get("usage", {})}
                returned_usage["elapsed_seconds"] = perf_counter() - started
                event.update(status="done", report=report, usage=returned_usage,
                             finished_at=now())
                journal["usage"] = add_costs(item["usage"] for item in journal["events"])
                write_json(journal_path, journal)
                return report
            except BaseException as error:
                event.update(status="interrupted", error_type=type(error).__name__,
                             finished_at=now())
                event["usage"]["elapsed_seconds"] = perf_counter() - started
                journal["status"] = "interrupted"
                journal["usage"] = add_costs(item["usage"] for item in journal["events"])
                write_json(journal_path, journal)
                raise

        def needs_retry(name, message):
            name = name if name == "import" else stage_prefix + name
            event = next(item for item in reversed(journal["events"])
                         if item["stage"] == name)
            event["status"] = "needs_retry"
            journal["status"] = "update_failed"
            write_json(journal_path, journal)
            raise ValueError(message)

        def attach_existing_evidence():
            if stage_prefix or final_status == "apply_finalizing":
                field = "refresh_input_evidence" if stage_prefix else "apply_input_evidence"
                # 此动作只在阶段真正执行时运行。显式重试已修好的证据会
                # 捕获新摘要；已完成阶段的缓存命中保留此前实际处理的摘要。
                journal[field] = self._refresh_state(
                    historical, record_path, evidence_root,
                )["evidence"]
            if evidence_root is None:
                return {"operation": "no_evaluation_source", "errors": [],
                        "usage": zero_cost()}
            root = Path(evidence_root)
            if not root.is_dir():
                raise ValueError("配置的评价来源目录不存在")
            if not (any(root.glob("**/acceptance.json"))
                    or any(root.glob("**/feedback/*/*.json"))):
                return {"operation": "no_evaluation_files", "errors": [],
                        "source_root": str(root), "usage": zero_cost()}
            return attach_evaluations(
                self.path, root, namespace=self.namespace, data_version=self.data_version,
                resolve_path=resolve_project_path, execution_ids=(execution_id,),
            )

        imported = stage("import", lambda: SQLiteHistoryStore(self.path).import_batch(
            [historical], namespace=self.namespace, data_version=self.data_version,
            source_root=str(record_path.parent), scanned_files=1, errors=[],
            started_at=perf_counter()))
        if imported.get("errors") or imported.get("conflicts"):
            needs_retry("import", "历史记录导入存在错误或内容冲突，请核对原记录后显式重试")
        attached = stage("evidence", attach_existing_evidence)
        if attached["errors"]:
            needs_retry("evidence", "评价证据关联失败，请审阅 update.json 后显式重试")
        evaluated = stage("quality", lambda: self.quality.assess_batch(
            namespace=self.namespace, data_version=self.data_version))
        decision = next(item for item in evaluated["decisions"]
                        if item["execution_id"] == execution_id)
        if after_quality is not None:
            decision = after_quality(decision, stage)

        if decision["admission_state"] == "eligible":
            def maintain():
                if self._difficulty_client is None:
                    self._difficulty_client = LocalDifficultyClient.from_yaml(self.model_config)
                pool = self.retriever.index.pool_store
                assessor = DifficultyAssessor(self._difficulty_client, cache=pool)
                return run_pool_maintenance(
                    quality_store=self.quality, pool_store=pool,
                    namespace=self.namespace, data_version=self.data_version,
                    assessor=assessor, config=self.pool_config,
                    schema_context=self.schema_context,
                    execution_ids=(execution_id,), retry_failed=retry_incomplete,
                )

            maintained = stage("maintenance", maintain, may_call_model=True)
            if maintained["error_count"]:
                needs_retry("maintenance", "标注或成员更新失败，请审阅 update.json 后显式重试")
            def rebuild_index():
                report = OfflineIndexBuilder(
                    self.retriever.index, self.retriever.prefilter.embeddings
                ).build(namespace=self.namespace, data_version=self.data_version)
                # 当前索引使用本地 dense 编码器，不产生 HTTP 请求或 LLM Token。
                # 保留编码器显式返回的未知值，只补齐本地编码没有的 HTTP 字段。
                report["usage"] = {**zero_cost(), **report["usage"]}
                return report

            indexed = stage("index", rebuild_index, may_call_model=True)
            if not indexed["index_complete"]:
                needs_retry("index", "更新后的索引未就绪，请审阅 update.json 后显式重试")

        after = members(self.retriever, self.namespace, self.data_version)
        journal.update(
            status=final_status or ("refresh_finalizing" if stage_prefix else "completed"),
            completed_at=now(), stored=True,
            admission_state=decision["admission_state"], quality_reasons=decision["reasons"],
            content_version=decision["content_version"],
            quality_sha256=digest(decision),
            admitted=any(member["execution_id"] == execution_id for member in after),
            members_after=after, membership_changed=after != (
                journal["refresh_members_before"] if stage_prefix else journal["members_before"]),
            usage=add_costs(event["usage"] for event in journal["events"]),
        )
        write_json(journal_path, journal)
        return journal
