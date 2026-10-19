"""直接SQL比较与单题记录；外部请求仅在用户执行run时发生。"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from src.project_paths import resolve_project_path
from .common import POLICY, load_cases, load_config, load_public_context, read_json, write_json
from .comparison import compare_results, evaluate_result

IMPLEMENTATION_VERSION = "dev_v0.1"


def _validate_inputs(config):
    import yaml
    from src.chatflow_agent.prompt_resources import parse_prompt_config

    document, all_cases = load_cases(config["questions"])
    truth = read_json(config["ground_truth"])
    business = yaml.safe_load(Path(config["business_config"]).read_text(encoding="utf-8"))
    if (document.get("data_version") != config["data_version"]
            or truth.get("data_version") != config["data_version"]
            or business["data"]["expected_baseline_id"] != config["data_version"]):
        raise ValueError("问题、标准与业务配置的数据版本须一致")
    if (document.get("status") == "DRAFT" or document.get("reviewed_by_user") is not True
            or truth.get("reviewed_by_user") is not True):
        raise ValueError("须使用已经审核完成的问题与标准")
    excluded = set(config["exclude_case_ids"])
    if not excluded.issubset({c["id"] for c in all_cases}):
        raise ValueError("排除清单含未知题号")
    excluded_sessions = {c["session"] for c in all_cases if c["id"] in excluded}
    cases = [c for c in all_cases if c["id"] not in excluded]
    if not cases or any(c["session"] in excluded_sessions for c in cases):
        raise ValueError("不得清空测试集或拆分多轮会话")
    facts = {}
    for fact in truth.get("cases", []):
        case_id = fact.get("id", fact.get("question_id"))
        if not case_id or case_id in facts:
            raise ValueError("标准题号须非空且唯一")
        facts[case_id] = fact
    standards = {}
    for case in cases:
        fact = facts.get(case["id"], {})
        expected = fact.get("expected_result")
        if fact.get("response_type") == "clarify":
            raise ValueError("澄清题应在运行前从SQL测试清单排除")
        if (not isinstance(expected, dict) or expected.get("is_complete", True) is not True
                or not compare_results(expected, expected)):
            raise ValueError("缺少有效、完整的标准SQL结果：" + case["id"])
        standards[case["id"]] = deepcopy({k: expected[k] for k in ("columns", "rows")})
    baseline = resolve_project_path(business["data"]["baseline_dir"])
    manifest = read_json(baseline / "manifest.json")
    schema = str(business["data"].get("expected_schema_version", ""))
    if manifest.get("baseline_id") != config["data_version"] or str(manifest.get("schema_version")) != schema:
        raise ValueError("业务数据版本或Schema不一致")
    if config.get("require_dataset_acceptance", True):
        if not config.get("acceptance_report"):
            raise ValueError("缺少已完成的数据库验收报告")
        acceptance = read_json(config["acceptance_report"])
        required = {"status": "PASS", "baseline_id": config["data_version"], "schema_version": schema,
                    "database": manifest["mysql"]["database"], "readonly": True,
                    "schema_contract_exact_match": True, "all_fields_exact_match": True,
                    "decimal_exact_match": True, "candidate_manifest_unchanged": True}
        if (any(acceptance.get(k) != v for k, v in required.items())
                or acceptance.get("offline_structure_business_scenario_pollution", {}).get("status") != "PASS"):
            raise ValueError("数据库验收报告未通过或版本不符")
    business["data"]["baseline_dir"] = str(baseline)
    business["prompt"] = parse_prompt_config(business.get("prompt"))
    for key in ("rules_file", "field_mapping_file"):
        if business.get("prompt", {}).get(key):
            business["prompt"][key] = str(resolve_project_path(business["prompt"][key]))
    public = load_public_context(config.get("public_context"), data_version=config["data_version"])
    return cases, standards, business, public


def prepare(config_path, run_dir):
    """只准备输入与独立经验库；不请求模型、不查询业务数据库。"""
    import yaml
    from ..integration.bootstrap import initialize_empty_store
    from ..integration.common import copy_database
    from ..pool.pool_models import GROUP_RULE_VERSION, PoolConfig
    from ..pool.pool_store import SQLitePoolStore
    from ..retrieval.index_store import SQLiteRetrievalIndex
    from ..retrieval.retrieval_models import RetrievalConfig
    from ..retrieval.vector_cache import SQLiteVectorCache

    config = load_config(config_path)
    cases, standards, business, public = _validate_inputs(config)
    retrieval = RetrievalConfig.from_json(config["retrieval_config"])
    capacities = PoolConfig(**read_json(config["pool_config"]))
    if capacities.strategy != "adaptive":
        raise ValueError("pool_config须提供adaptive分组容量；FIFO容量单独由fifo_capacity指定")
    pool_configs = {
        mode: (PoolConfig(strategy="fifo", fifo_capacity=config["fifo_capacity"])
               if mode == "fifo" else capacities)
        for mode in config["conditions"]
    }
    run = resolve_project_path(run_dir)
    if run.exists():
        raise FileExistsError("实验目录已存在；请使用全新运行目录")
    for key in ("questions", "ground_truth", "business_config", "retrieval_config", "pool_config", "source_store"):
        source = config.get(key)
        if source and Path(source).is_relative_to(run):
            raise ValueError("运行目录不得包含输入文件")
    run.mkdir(parents=True)
    # 各组共享相同问题和公共运行输入，分别保存功能性工作库。
    inputs = run / "inputs"
    inputs.mkdir()
    baseline = Path(business["data"]["baseline_dir"])
    for name in ("manifest.json", "database_schema.sql"):
        shutil.copyfile(baseline / name, inputs / name)
    business["data"]["baseline_dir"] = str(inputs)
    for key, name in (("rules_file", "rules.txt"), ("field_mapping_file", "field_mapping.json")):
        if business.get("prompt", {}).get(key):
            shutil.copyfile(business["prompt"][key], inputs / name)
            business["prompt"][key] = str(inputs / name)
    business["output"] = {"directory": str(run)}
    business["learning"] = {**business.get("learning", {}), "mode": "standard"}
    (run / "business.yaml").write_text(yaml.safe_dump(business, allow_unicode=True, sort_keys=False), encoding="utf-8")
    shutil.copyfile(config["retrieval_config"], run / "retrieval.json")
    shutil.copyfile(config["pool_config"], run / "pool.json")
    initial = run / ".initial.sqlite3"
    try:
        if config["initial_state"] == "snapshot":
            copy_database(config["source_store"], initial)
            pool = SQLitePoolStore(initial, record_runs=False)
            pool.initialize()
            index = SQLiteRetrievalIndex(SQLiteVectorCache(pool), retrieval)
            inspection = index.inspect(namespace=config["namespace"], data_version=config["data_version"])
            if (not inspection["complete"] or not inspection["member_count"]
                    or not inspection["ready_binding_count"]):
                raise ValueError("种子池须有有效成员及当前版本的完整索引")
            seeds = pool.load_active_members(namespace=config["namespace"], data_version=config["data_version"])
            seed_keys = {(entry["identity"]["execution_id"], entry["identity"]["content_version"])
                         for entry in seeds}
            eligible_keys = {(entry["identity"]["execution_id"], entry["identity"]["content_version"])
                             for entry in pool.load_inputs(namespace=config["namespace"], data_version=config["data_version"])
                             if entry["quality"].admission_state == "eligible"}
            if seed_keys != eligible_keys:
                raise ValueError("共同种子库须只包含选定的有效经验，不能夹带池外eligible候选")
            if any(entry["grouping"].rule_version != GROUP_RULE_VERSION for entry in seeds):
                raise ValueError("种子经验须先按当前SQL规则归组")
            if "fifo" in config["conditions"] and len(seeds) > config["fifo_capacity"]:
                raise ValueError("共同种子数超过FIFO容量；请先准备两组都能完整容纳的同一份种子")
        for mode in config["conditions"]:
            folder = run / mode
            folder.mkdir()
            store = folder / "history.sqlite3"
            if config["initial_state"] == "empty":
                initialize_empty_store(store, namespace=config["namespace"], data_version=config["data_version"],
                                       retrieval_config=retrieval, pool_config=pool_configs[mode], record_runs=False)
            else:
                copy_database(initial, store)
                # 仅改变保留策略，保留完全相同的种子成员、内容及现有向量。
                SQLitePoolStore(store, record_runs=False).configure(
                    namespace=config["namespace"], data_version=config["data_version"],
                    config=pool_configs[mode],
                )
            _set_completed(store, False)
    finally:
        initial.unlink(missing_ok=True)
    prepared = {"policy": POLICY, "run_id": "direct-" + uuid4().hex[:16],
                "implementation_version": IMPLEMENTATION_VERSION,
                "namespace": config["namespace"], "data_version": config["data_version"],
                "conditions": config["conditions"], "initial_state": config["initial_state"],
                "business_config": str(run / "business.yaml"), "retrieval_config": str(run / "retrieval.json"),
                "pool_config": str(run / "pool.json"),
                "pool_configs": {mode: value.to_dict() for mode, value in pool_configs.items()},
                "public_context": public,
                "cases": cases, "standards": standards}
    write_json(run / "config.json", prepared)
    return {"run_dir": str(run), "conditions": config["conditions"], "case_count": len(cases)}


def load_run(run_dir):
    run = resolve_project_path(run_dir)
    path = run / "config.json"
    if not path.is_file():
        raise ValueError("不是新版直接比较目录；旧实验只读保留，请新建目录")
    config = read_json(path)
    if config.get("policy") != POLICY:
        raise ValueError("运行目录不是sql_result_direct_v1")
    return run, config, config["cases"]


def _set_completed(path, completed):
    # 仅保存执行完整性标记；题目评分始终只有pass/fail。
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS direct_validation_state(completed INTEGER NOT NULL)")
        connection.execute("DELETE FROM direct_validation_state")
        connection.execute("INSERT INTO direct_validation_state VALUES (?)", (int(completed),))


def condition_completed(path):
    if not Path(path).is_file():
        return False
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='direct_validation_state'").fetchone()
        return bool(exists and connection.execute("SELECT completed FROM direct_validation_state").fetchone() == (1,))


def _make_retriever(config, store):
    from ..retrieval.retrieval_models import RetrievalConfig
    from ..retrieval.service import PSURetriever
    retriever = PSURetriever.from_local(store, retrieval_config=RetrievalConfig.from_json(config["retrieval_config"]),
                                      model_config_path=config["business_config"])
    retriever.index.pool_store.record_runs = False
    return retriever


def _make_service(config, mode, retriever):
    from ..integration.local_query import LocalValidationQueryService
    return LocalValidationQueryService(config["business_config"], experiment_id=config["run_id"] + "-" + mode,
                                       condition=mode, namespace=config["namespace"],
                                       data_version=config["data_version"], retriever=retriever)


def _make_updater(config, service, retriever):
    from ..integration.direct_update import DirectExperienceUpdater
    from ..pool.pool_models import PoolConfig
    pool_config = PoolConfig(**config["pool_configs"][service.condition])
    return DirectExperienceUpdater(retriever, namespace=config["namespace"], data_version=config["data_version"],
                                   model_config=config["business_config"], pool_config=pool_config,
                                   schema_context="\n\n".join(service.schemas.values()))


def run_condition(run_dir, mode):
    """整组顺序运行，无人工审核、裁判或旧式断点恢复。"""
    from src.lifelong_learning.core.models import ChatHistory
    from ..integration.direct_update import DirectUpdateError

    run, config, cases = load_run(run_dir)
    if mode not in config["conditions"]:
        raise ValueError("该条件不在本轮配置中")
    folder, store = run / mode / "results", run / mode / "history.sqlite3"
    if not store.is_file():
        raise FileNotFoundError(store)
    if folder.exists():
        if condition_completed(store):
            from .reporting import summarize
            return summarize(run)["conditions"][mode]
        raise ValueError("本组曾启动但未完整结束；请使用新目录和原始种子重跑")
    if config.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("运行目录由其他源码版本准备；请用dev_v0.1创建新目录")
    folder.mkdir()
    _set_completed(store, False)
    retriever = service = updater = None
    try:
        if mode != "baseline":
            retriever = _make_retriever(config, store)
        service = _make_service(config, mode, retriever)
        service.run_id = config["run_id"]
        for position, case in enumerate(cases, 1):
            if case["turn"] == 1:
                service.reset()
            service._turn_index = case["turn"] - 1
            service._history = ChatHistory.model_validate({"messages": [*config["public_context"], *case["history"]]})
            service.session_history = deepcopy(case["history"])
            service.case_id = case["id"]
            print(f"[{mode} {position}/{len(cases)}] {case['question']}", flush=True)
            try:
                record = service.query(case["question"])
            except Exception as error:
                print("查询失败：" + type(error).__name__, file=sys.stderr, flush=True)
                record = {"status": "failed", "operations": [], "result_query_id": None}
            expected = config["standards"][case["id"]]
            result = evaluate_result(record, expected)
            row = {"case_id": case["id"], "condition": mode, "question": case["question"],
                   "query_id": result["query_id"], "sql": result["sql"],
                   "actual": result["actual"], "expected": expected, "verdict": result["verdict"]}
            path = folder / (case["id"] + ".json")
            # 每题只写一次；PSU引用此不可变文件，但不导入其中的标准结果。
            with path.open("x", encoding="utf-8") as stream:
                json.dump(row, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            print("  " + result["verdict"], flush=True)
            if mode in ("fifo", "adaptive") and result["verdict"] == "pass":
                if updater is None:
                    updater = _make_updater(config, service, retriever)
                try:
                    updater.apply(case=case, record=record, record_path=path, query_id=result["query_id"])
                except DirectUpdateError as error:
                    print("本题经验更新失败且已隔离：" + type(error).__name__, file=sys.stderr, flush=True)
        _set_completed(store, True)
    finally:
        for resource in (updater, service, retriever):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    print("资源关闭失败：" + type(error).__name__, file=sys.stderr)
    from .reporting import summarize
    return summarize(run)["conditions"][mode]
