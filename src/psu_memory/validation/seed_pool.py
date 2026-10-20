"""用户执行的共同种子初始化：已核验经验 → 难度标注 → 检索索引。

不重新请求查询 SQL，不连接业务 MySQL；SQLite、难度模型及嵌入仅在调用时运行。
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from src.project_paths import resolve_project_path
from .authored_formation import DATA_VERSION, NAMESPACE, _actual_record
from .common import write_json


DEFAULT_SEEDS = "data/experiences/psu_memory/enterprise_v30_120_v1_0_2_initial_seed_v1/seeds.json"
DEFAULT_BUSINESS_CONFIG = "configs/business/query_v30_120.yaml"
DEFAULT_RETRIEVAL_CONFIG = "src/psu_memory/retrieval/retrieval_config.example.json"
DEFAULT_POOL_CONFIG = "src/psu_memory/pool/pool_config.example.json"
DEFAULT_DIFFICULTY_REVIEW = "configs/validation/enterprise_v30_120_v1_0_2_seed_difficulty_review_v1.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _schema_context(config_path, data_version):
    """与 QueryService.schemas 使用同一映射及同一组合顺序，不创建 QueryService。"""
    import yaml
    from src.chatflow_agent.prompt_resources import (
        BASIC_PROMPT_PROFILE, load_prompt_resources, parse_prompt_config,
    )
    from src.lifelong_learning.tasks.enterprise_query_bench.schema_catalog import (
        GUIDED_INTERFACE_PROFILE, basic_table_schemas, build_basic_schema_catalog,
        build_compact_schemas,
    )

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data = config["data"]
    baseline = resolve_project_path(data["baseline_dir"])
    manifest_path, schema_path = baseline / "manifest.json", baseline / "database_schema.sql"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = str(manifest["schema_version"])
    if (data.get("expected_baseline_id") != data_version or manifest.get("baseline_id") != data_version
            or str(data.get("expected_schema_version")) != version):
        raise ValueError("业务配置与种子的数据版本或 Schema 不一致")
    schema_sql = schema_path.read_text(encoding="utf-8")
    prompt = parse_prompt_config(config.get("prompt"))
    paths = [Path(config_path), manifest_path, schema_path]
    if prompt["profile"] == BASIC_PROMPT_PROFILE:
        resources = load_prompt_resources(prompt)
        catalog = build_basic_schema_catalog(
            schema_sql, resources["field_mapping"], schema_version=version,
            table_descriptions=resources["table_descriptions"],
        )
        schemas = basic_table_schemas(catalog)
        paths.extend(Path(prompt[name]) for name in ("rules_file", "field_mapping_file"))
    else:
        schemas = build_compact_schemas(schema_sql=schema_sql,
                                        interface_profile=GUIDED_INTERFACE_PROFILE,
                                        schema_version=version)
    return "\n\n".join(schemas.values()), {str(path): _sha(path) for path in paths}


def _make_difficulty_client(config_path):
    from ..pool.difficulty_llm import LocalDifficultyClient
    return LocalDifficultyClient.from_yaml(config_path)


def _make_embeddings(config, cache):
    from ..retrieval.embedding_service import EmbeddingService
    return EmbeddingService(config, cache)


def _tracked_assessor(client, pool, cases_by_execution, outcomes):
    from ..pool.difficulty import DifficultyAssessor

    class TrackedAssessor(DifficultyAssessor):
        def assess(self, request, *, admission_state, retry_failed=False):
            result = super().assess(request, admission_state=admission_state, retry_failed=retry_failed)
            assessment = result.assessment
            outcomes[request.execution_id] = {
                "case_id": cases_by_execution[request.execution_id], "status": assessment.status,
                "level": assessment.level, "reason": assessment.reason,
                "missing_information": list(assessment.missing_information),
                "cache_hit": result.cache_hit,
            }
            return result

    return TrackedAssessor(client, cache=pool)


def _verify_members(pool, bundle, execution_ids):
    """同时确认同一候选集合、SQL分组、实际难度及预定 FIFO 顺序。"""
    from ..pool.pool_models import GROUP_IDS, GROUP_RULE_VERSION
    from ..pool.selection import has_sql_result_pass

    scope = {"namespace": bundle["namespace"], "data_version": bundle["data_version"]}
    active = pool.load_active_members(**scope)
    entries = pool.load_inputs(**scope)
    if ([item["identity"]["execution_id"] for item in active] != execution_ids
            or [item["position"] for item in active] != list(range(len(execution_ids)))):
        raise ValueError("共同种子成员或 FIFO 顺序与选种清单不一致")
    if (len(entries) != len(execution_ids)
            or {item["identity"]["execution_id"] for item in entries} != set(execution_ids)
            or any(not has_sql_result_pass(item["quality"]) for item in entries)):
        raise ValueError("共同种子库夹带未选候选或缺少有效直接 PASS")
    groups = {name: 0 for name in GROUP_IDS}
    levels = {name: 0 for name in ("L1", "L2", "L3", "L4")}
    for seed, member in zip(bundle["entries"], active):
        grouping, difficulty = member["grouping"], member["difficulty"]
        if (grouping.rule_version != GROUP_RULE_VERSION or grouping.group_id != seed["group"]
                or grouping.method != "rule" or grouping.text_source != "sql"
                or difficulty.assessment.status != "assessed"):
            raise ValueError("当前 SQL 分组或模型难度标注无效：" + seed["case_id"])
        groups[grouping.group_id] += 1
        levels[difficulty.assessment.level] += 1
    if any(count > 8 for count in groups.values()) or len(active) > 18:
        raise ValueError("共同种子超过 FIFO18 或 adaptive 单组8容量")
    return groups, levels


def initialize_seed_pool(*, run_dir, seeds_path=DEFAULT_SEEDS,
                         business_config=DEFAULT_BUSINESS_CONFIG,
                         retrieval_config=DEFAULT_RETRIEVAL_CONFIG,
                         pool_config=DEFAULT_POOL_CONFIG):
    """只在全部准备完成后发布 seed.sqlite3；失败保留 partial 与一份汇总。"""
    from .initial_seeds import load_seed_bundle
    from ..evaluation.quality_store import SQLiteQualityStore
    from ..integration.bootstrap import initialize_empty_store
    from ..integration.direct_update import DirectCandidateCollector
    from ..pool.maintenance import run_pool_maintenance
    from ..pool import difficulty, pool_models
    from ..pool.pool_models import DIFFICULTY_PROMPT_VERSION, PoolConfig
    from ..pool.pool_store import SQLitePoolStore
    from ..retrieval.index_store import SQLiteRetrievalIndex
    from ..retrieval.indexing import OfflineIndexBuilder
    from ..retrieval.retrieval_models import RetrievalConfig
    from ..retrieval.vector_cache import SQLiteVectorCache

    seeds_path = resolve_project_path(seeds_path)
    input_hashes = {str(seeds_path): _sha(seeds_path)}
    bundle = load_seed_bundle(seeds_path)
    if (bundle["data_version"] != DATA_VERSION or bundle["namespace"] != NAMESPACE
            or not 1 <= len(bundle["entries"]) <= 18):
        raise ValueError("共同种子须使用当前数据/命名域且为1至18条")
    business_path = resolve_project_path(business_config)
    retrieval_path, pool_path = map(resolve_project_path, (retrieval_config, pool_config))
    input_hashes.update({str(path): _sha(path) for path in (business_path, retrieval_path, pool_path)})
    schema_context, schema_hashes = _schema_context(business_path, DATA_VERSION)
    input_hashes.update(schema_hashes)
    difficulty_sources = (
        Path(pool_models.__file__).resolve(), Path(difficulty.__file__).resolve(),
        Path(pool_models.__file__).resolve().parent / "prompts" / f"{DIFFICULTY_PROMPT_VERSION}.txt",
    )
    input_hashes.update({str(path): _sha(path) for path in difficulty_sources})
    for source in bundle.get("sources", {}).values():
        input_hashes[str(resolve_project_path(source["path"]))] = source["sha256"]
    for entry in bundle["entries"]:
        input_hashes[str(resolve_project_path(entry["source"]["path"]))] = entry["source"]["sha256"]
    retrieval = RetrievalConfig.from_json(retrieval_path)
    adaptive = PoolConfig(**json.loads(pool_path.read_text(encoding="utf-8")))
    if adaptive.strategy != "adaptive" or any(cap != 8 for cap in adaptive.capacities.values()):
        raise ValueError("此共同种子对应四组各8条的 adaptive 配置")
    fifo = PoolConfig(strategy="fifo", fifo_capacity=18, capacities=adaptive.capacities)
    for source, expected in input_hashes.items():
        if _sha(source) != expected:
            raise ValueError("初始化前来源已变化：" + source)
    run = resolve_project_path(run_dir)
    if run.exists():
        raise FileExistsError("运行目录已存在，请指定全新目录：" + str(run))
    run.mkdir(parents=True)
    partial, target = run / "seed.partial.sqlite3", run / "seed.sqlite3"
    scope = {"namespace": bundle["namespace"], "data_version": bundle["data_version"]}
    started = perf_counter()
    stage = "candidate_import"
    client = collector = None
    maintenance = indexing = None
    execution_ids = []
    groups = levels = None
    error = None
    local_seconds = 0.0
    outcomes = {}
    try:
        begin = perf_counter()
        initialize_empty_store(partial, **scope, retrieval_config=retrieval,
                               pool_config=fifo, record_runs=False)
        collector = DirectCandidateCollector(partial, **scope, initialize=False)
        for entry in bundle["entries"]:
            row = entry["record"]
            case = {"id": entry["case_id"], "question": row["question"],
                    "history": deepcopy(row["context"]), "session": row["session_id"], "turn": row["turn"]}
            result = collector.apply(case, _actual_record(case, row, "initial-seed-v1"),
                                     resolve_project_path(entry["source"]["path"]), row["query_id"])
            execution_ids.append(result["execution_id"])
        local_seconds = perf_counter() - begin
        pool = SQLitePoolStore(partial, record_runs=False)
        quality = SQLiteQualityStore(partial, resolve_path=resolve_project_path, record_runs=False)
        stage = "difficulty_and_pool"
        client = _make_difficulty_client(business_path)
        cases_by_execution = dict(zip(execution_ids, (entry["case_id"] for entry in bundle["entries"])))
        assessor = _tracked_assessor(client, pool, cases_by_execution, outcomes)

        def progress(event):
            if event["stage"] == "done" and event["execution_id"] in outcomes:
                outcomes[event["execution_id"]]["preparation"] = event["operation"]
            label = cases_by_execution[event["execution_id"]]
            print(f"[seed {event['index']}/{event['total']}] {label} {event['stage']}"
                  + (" " + event["operation"] + " " + str(event.get("level"))
                     if event["stage"] == "done" else ""), flush=True)

        # 种子和在线 PASS 共用难度输入；显式 ID 保持共同种子的原始顺序。
        maintenance = run_pool_maintenance(
            quality_store=quality, pool_store=pool, **scope, assessor=assessor,
            config=fifo, execution_ids=tuple(execution_ids), schema_context=schema_context,
            strict=False, refresh_quality=False,
            progress=progress,
        )
        if maintenance["error_count"] or maintenance["preparation"]["prepared_count"] != len(execution_ids):
            raise ValueError("部分种子难度标注失败，未发布共同种子库")
        groups, levels = _verify_members(pool, bundle, execution_ids)
        stage = "embedding_index"
        cache = SQLiteVectorCache(pool)
        index = SQLiteRetrievalIndex(cache, retrieval)
        indexing = OfflineIndexBuilder(index, _make_embeddings(retrieval, cache)).build(**scope)
        if (indexing["error_count"] or not indexing["index_complete"]
                or not indexing["ready_binding_count"]):
            raise ValueError("种子检索索引未完整就绪")
        stage = "publication"
        _verify_members(pool, bundle, execution_ids)
        for source, expected in input_hashes.items():
            if _sha(source) != expected:
                raise ValueError("初始化期间来源发生变化，未发布：" + source)
        if client is not None:
            client.close()
            client = None
        # 所有 SQLite 操作均已关闭连接。硬链接发布拒绝覆盖任何既有目标。
        os.link(partial, target)
        try:
            partial.unlink()
        except OSError:
            pass  # 已成功发布；清理同内容临时链接失败不否定已完成的初始化。
    except Exception as exc:
        if stage == "candidate_import":
            local_seconds = perf_counter() - started
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        if collector is not None:
            collector.close()
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                if error is None:
                    error = {"stage": "client_close", "type": type(exc).__name__, "message": str(exc)[:1000]}
    ready = error is None and target.is_file()
    summary = {
        "version": "psu_initial_seed_pool_v1", "ready": ready, "run_dir": str(run),
        **scope, "seed_store": str(target) if ready else None,
        "partial_store": str(partial) if partial.is_file() else None,
        "selected_count": len(bundle["entries"]), "member_count": len(execution_ids) if ready else None,
        "case_ids": [entry["case_id"] for entry in bundle["entries"]],
        "group_counts": groups, "model_difficulty_counts": levels,
        "preparation": maintenance["preparation"] if maintenance is not None else None,
        "preparation_error_count": maintenance["error_count"] if maintenance is not None else None,
        "difficulty_results": [outcomes[key] for key in execution_ids if key in outcomes],
        "difficulty_prompt_version": DIFFICULTY_PROMPT_VERSION,
        "source_sha256": input_hashes, "error": error,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cost": {"elapsed_seconds": perf_counter() - started,
                 "candidate_import_seconds": local_seconds,
                 "difficulty_and_pool": maintenance["usage"] if maintenance is not None else None,
                 "embedding_index": indexing["usage"] if indexing is not None else None,
                 "scope": "共同种子初始化；不包含形成题执行，不计入正式查询对比"},
    }
    write_json(run / "summary.json", summary)
    return summary


def apply_difficulty_review(*, run_dir, review_path=DEFAULT_DIFFICULTY_REVIEW,
                            seeds_path=DEFAULT_SEEDS, retrieval_config=DEFAULT_RETRIEVAL_CONFIG):
    """用户应用已确定的种子难度复核；只复制本地库、改标注并复用已有向量。"""
    started = perf_counter()
    from contextlib import closing
    import sqlite3

    from .initial_seeds import load_seed_bundle
    from ..history.history_adapter import history_from_query_result
    from ..pool import difficulty, pool_models
    from ..pool._common import canonical_json, sha256_text
    from ..pool.pool_models import (
        DIFFICULTY_LEVELS, DIFFICULTY_PROMPT_VERSION,
        DifficultyAssessment, DifficultyResult, DifficultyUsage,
    )
    from ..pool.pool_store import SQLitePoolStore
    from ..retrieval.index_store import REPRESENTATIONS, SQLiteRetrievalIndex
    from ..retrieval.retrieval_models import RetrievalConfig
    from ..retrieval.vector_cache import SQLiteVectorCache

    review_path, seeds_path, retrieval_path = map(
        resolve_project_path, (review_path, seeds_path, retrieval_config))
    review_sha = _sha(review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (review.get("version") != "psu_seed_difficulty_review_v1" or review.get("actor") != "assistant"
            or review.get("namespace") != NAMESPACE or review.get("data_version") != DATA_VERSION
            or not isinstance(review.get("source_run"), dict)
            or not isinstance(review.get("changes"), list) or not review["changes"]):
        raise ValueError("人工复核文件的版本、来源、范围或更正清单无效")
    source_run = resolve_project_path(review["source_run"]["path"])
    source_summary, source_store = source_run / "summary.json", source_run / "seed.sqlite3"
    summary_sha, store_sha = _sha(source_summary), _sha(source_store)
    if (summary_sha != review["source_run"].get("summary_sha256")
            or store_sha != review["source_run"].get("seed_sha256")):
        raise ValueError("人工复核绑定的原种子或摘要已变化")
    original = json.loads(source_summary.read_text(encoding="utf-8"))
    bundle = load_seed_bundle(seeds_path)
    case_ids = [entry["case_id"] for entry in bundle["entries"]]
    if (original.get("version") != "psu_initial_seed_pool_v1" or original.get("ready") is not True
            or original.get("error") is not None or original.get("manual_review")
            or original.get("namespace") != NAMESPACE or original.get("data_version") != DATA_VERSION
            or bundle.get("namespace") != NAMESPACE or bundle.get("data_version") != DATA_VERSION
            or len(case_ids) != 18 or len(set(case_ids)) != 18
            or original.get("case_ids") != case_ids
            or original.get("selected_count") != 18 or original.get("member_count") != 18
            or resolve_project_path(original.get("run_dir", "")) != source_run
            or resolve_project_path(original.get("seed_store") or "") != source_store
            or DIFFICULTY_PROMPT_VERSION != "difficulty_v2"
            or original.get("difficulty_prompt_version") != DIFFICULTY_PROMPT_VERSION):
        raise ValueError("须使用当前 difficulty_v2 的完整18条原始种子及相同选种清单")
    model_rows = original.get("difficulty_results")
    if (not isinstance(model_rows, list) or [row.get("case_id") for row in model_rows] != case_ids
            or any(row.get("status") != "assessed" or row.get("level") not in DIFFICULTY_LEVELS
                   or row.get("missing_information") for row in model_rows)):
        raise ValueError("原摘要的逐题模型难度不完整")
    expected_levels = {row["case_id"]: row["level"] for row in model_rows}
    changes = {}
    for change in review["changes"]:
        if (not isinstance(change, dict) or set(change) != {"case_id", "from_level", "to_level", "reason"}
                or change["case_id"] not in expected_levels or change["case_id"] in changes
                or change["from_level"] != expected_levels[change["case_id"]]
                or change["to_level"] not in DIFFICULTY_LEVELS
                or change["to_level"] == change["from_level"]
                or not isinstance(change["reason"], str) or not 1 <= len(change["reason"].strip()) <= 300):
            raise ValueError("人工更正必须唯一绑定原等级、有效新等级及简短理由")
        changes[change["case_id"]] = change
        expected_levels[change["case_id"]] = change["to_level"]

    prompt_path = Path(pool_models.__file__).resolve().parent / "prompts" / "difficulty_v2.txt"
    source_hashes = original.get("source_sha256")
    required = (seeds_path, retrieval_path, Path(pool_models.__file__).resolve(),
                Path(difficulty.__file__).resolve(), prompt_path)
    if not isinstance(source_hashes, dict) or any(str(path) not in source_hashes for path in required):
        raise ValueError("原摘要缺少种子、检索配置或难度口径的来源绑定")
    protected = {str(resolve_project_path(path)): value for path, value in source_hashes.items()}
    protected.update({str(review_path): review_sha, str(source_summary): summary_sha,
                      str(source_store): store_sha, str(Path(__file__).resolve()): _sha(__file__)})

    def verify_sources():
        for path, expected in protected.items():
            if _sha(path) != expected:
                raise ValueError("人工复核应用期间来源发生变化：" + path)

    verify_sources()
    scope = {"namespace": NAMESPACE, "data_version": DATA_VERSION}
    source_pool = SQLitePoolStore(source_store, record_runs=False)
    members = source_pool.load_active_members(**scope)
    execution_ids = [entry["identity"]["execution_id"] for entry in members]
    groups, model_counts = _verify_members(source_pool, bundle, execution_ids)
    if groups != original.get("group_counts") or model_counts != original.get("model_difficulty_counts"):
        raise ValueError("原库分组或模型难度分布与摘要不一致")
    with source_pool.connection() as connection:
        config = source_pool._read_config(connection, **scope)
    if config.strategy != "fifo" or config.fifo_capacity != 18:
        raise ValueError("人工复核仅适用于原始 FIFO18 共同种子")
    for seed, member, model in zip(bundle["entries"], members, model_rows):
        row = seed["record"]
        case = {"id": seed["case_id"], "question": row["question"], "history": row["context"],
                "session": row["session_id"], "turn": row["turn"]}
        historical = history_from_query_result(
            case, _actual_record(case, row, "initial-seed-v1"),
            resolve_project_path(seed["source"]["path"]), row["query_id"], **scope)
        content = member["quality"].content
        assessment = member["difficulty"].assessment
        if (member["identity"]["execution_id"] != historical.task.task_id
                or content.get("question") != historical.task.question
                or content.get("context_text") != historical.task.context_text
                or content.get("operation") != json.loads(historical.task.evidence_text)[0]
                or content.get("source_sha256") != seed["source"]["sha256"]
                or assessment.level != model["level"] or assessment.reason != model["reason"]
                or list(assessment.missing_information) != model["missing_information"]
                or assessment.prompt_version != DIFFICULTY_PROMPT_VERSION
                or assessment.prompt_sha256 != source_hashes[str(prompt_path)]
                or assessment.model_signature.startswith("manual:")
                or source_pool.get(assessment.cache_key) != assessment):
            raise ValueError("原种子实际问答、顺序或模型标注与绑定来源不一致：" + seed["case_id"])
    retrieval = RetrievalConfig.from_json(retrieval_path)
    source_index = SQLiteRetrievalIndex(SQLiteVectorCache(source_pool), retrieval)
    prior_index = source_index.inspect(**scope)
    if (not prior_index["complete"] or prior_index["member_count"] != 18
            or prior_index["expected_binding_count"] != 18 * len(REPRESENTATIONS)
            or not prior_index["ready_binding_count"]):
        raise ValueError("原种子的双表示索引或向量缓存不完整")
    verify_sources()
    run = resolve_project_path(run_dir)
    if run.exists() or run.is_relative_to(source_run):
        raise FileExistsError("请指定原种子目录之外的全新运行目录：" + str(run))
    run.mkdir(parents=True)
    partial, target = run / "seed.partial.sqlite3", run / "seed.sqlite3"
    stage = "copy_seed"
    error = None
    effective_counts = None
    try:
        with closing(sqlite3.connect(source_store.as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(partial)) as destination:
                source.backup(destination)
        pool = SQLitePoolStore(partial, record_runs=False)
        stage = "apply_manual_review"
        for seed, member in zip(bundle["entries"], members):
            change = changes.get(seed["case_id"])
            if change is None:
                continue
            previous = member["difficulty"].assessment
            assessment = DifficultyAssessment(
                cache_key="manual:" + sha256_text(canonical_json({
                    "review_sha256": review_sha, "source_seed_sha256": store_sha,
                    "identity": member["identity"], "change": change,
                })),
                prompt_version=previous.prompt_version, prompt_sha256=previous.prompt_sha256,
                model_signature="manual:assistant_static_review_v1", status="assessed",
                level=change["to_level"], reason=change["reason"].strip(),
                missing_information=(), origin_usage=DifficultyUsage(),
            )
            pool.put(assessment)
            pool.save_annotation(member, grouping=member["grouping"], difficulty=DifficultyResult(
                **member["identity"], assessment=assessment, cache_hit=False, usage=DifficultyUsage()))
        stage = "rebind_index"
        updated = pool.load_active_members(**scope)
        updated_groups, effective_counts = _verify_members(pool, bundle, execution_ids)
        if updated_groups != groups:
            raise ValueError("人工复核改变了种子分组")
        for seed, old, new in zip(bundle["entries"], members, updated):
            prior = old["difficulty"].assessment
            current = new["difficulty"].assessment
            if (new["identity"] != old["identity"] or new["quality"] != old["quality"]
                    or new["grouping"] != old["grouping"]
                    or current.level != expected_levels[seed["case_id"]]
                    or pool.get(prior.cache_key) != prior
                    or (seed["case_id"] not in changes and current != prior)
                    or (seed["case_id"] in changes
                        and current.model_signature != "manual:assistant_static_review_v1")):
                raise ValueError("人工更正之外的种子内容或原模型缓存发生变化")
        index = SQLiteRetrievalIndex(SQLiteVectorCache(pool), retrieval)
        old_bindings = {(item["identity"]["execution_id"], item["representation"]): item
                        for item in prior_index["bindings"]}
        bindings = [{**deepcopy(old_bindings[(member["identity"]["execution_id"], representation)]),
                     **index.descriptor(member, representation)}
                    for member in updated for representation in REPRESENTATIONS]
        publication = index.publish(**scope, members=updated, bindings=bindings)
        inspection = index.inspect(**scope)
        if (not publication["published"] or not inspection["complete"]
                or inspection["expected_binding_count"] != prior_index["expected_binding_count"]
                or inspection["ready_binding_count"] != prior_index["ready_binding_count"]
                or inspection["vectors"] != prior_index["vectors"]):
            raise ValueError("已有向量的索引重绑定未完整就绪")
        stage = "publication"
        _verify_members(pool, bundle, execution_ids)
        verify_sources()
        os.link(partial, target)
        try:
            partial.unlink()
        except OSError:
            pass
    except Exception as exc:
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:1000]}
    ready = error is None and target.is_file()
    summary = deepcopy(original)
    summary.update({
        "ready": ready, "run_dir": str(run), "seed_store": str(target) if ready else None,
        "partial_store": str(partial) if partial.is_file() else None,
        "member_count": 18 if ready else None, "error": error,
        "effective_difficulty_counts": effective_counts if ready else None,
        "difficulty_results": [{**row, "effective_level": expected_levels[row["case_id"]] if ready else None}
                               for row in model_rows],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manual_review": {
            "actor": review["actor"], "changes": deepcopy(review["changes"]), "applied": ready,
            "review_file": {"path": str(review_path), "sha256": review_sha},
            "source_run": {"path": str(source_run), "summary": str(source_summary),
                           "summary_sha256": summary_sha, "seed": str(source_store), "seed_sha256": store_sha},
            "application_sha256": protected[str(Path(__file__).resolve())],
            "cost": {"elapsed_seconds": perf_counter() - started, "model_call_count": 0,
                     "http_request_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
                     "embedding_input_tokens": 0, "embedding_model_call_count": 0,
                     "token_usage_complete": True},
        },
    })
    write_json(run / "summary.json", summary)
    return summary
