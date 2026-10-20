"""由用户执行预编写 SQL，收集真实结果并保存直接 PASS 的成功候选。

没有查询模型、经验检索、难度评分或在线更新；私有标准仅在内存中比较。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from src.project_paths import resolve_project_path
from .common import POLICY, load_cases, write_json
from .comparison import compare_results
from .splits import select_cases


VERSION = "psu_authored_formation_v1"
DATA_VERSION = "synthetic_enterprise_v30_120_v1.0.2"
DATABASE = "synthetic_enterprise_v30_120"
NAMESPACE = "validation-v30-120-v1.0.2-natural-v0.3.0-split-v1"
DEFAULT_BUNDLE = "data/experiences/psu_memory/enterprise_v30_120_v1_0_2_authored_v1/bundle.json"
SOURCE_NAMES = {"questions", "ground_truth", "reference_validation", "split",
                "dataset_manifest", "business_context"}


@dataclass
class AuthoredBundle:
    path: Path
    sha256: str
    namespace: str
    cases: list[dict]
    statements: dict[str, str]
    standards: dict[str, dict]
    record_counts: dict[str, int]
    source_hashes: dict[str, str]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def load_bundle(bundle_path=DEFAULT_BUNDLE):
    """核对绑定后一次读入 SQL、公开题和标准；执行期间不再读取这些输入。"""
    root = resolve_project_path(".")
    path = resolve_project_path(bundle_path)
    snapshots = {}

    def read_bytes(source):
        content = source.read_bytes()
        _require(source not in snapshots or snapshots[source] == content,
                 "准备期间输入发生变化：" + str(source))
        snapshots[source] = content
        return content

    raw_bundle = read_bytes(path)
    bundle = json.loads(raw_bundle)
    _require(bundle.get("version") == VERSION, "预编写经验包版本不符")
    _require(bundle.get("data_version") == DATA_VERSION, "经验包数据版本不符")
    _require(bundle.get("namespace") == NAMESPACE, "经验包须使用固定划分的种子命名域")
    bindings = bundle.get("sources")
    _require(isinstance(bindings, dict) and SOURCE_NAMES.issubset(bindings), "经验包来源绑定不完整")
    documents, paths, hashes = {}, {}, {}
    for name, binding in bindings.items():
        _require(isinstance(binding, dict) and isinstance(binding.get("path"), str),
                 "经验包来源路径无效：" + name)
        relative = Path(binding["path"])
        source = (root / relative).resolve()
        _require(not relative.is_absolute() and source.is_relative_to(root) and source.is_file(),
                 "来源必须是项目内的现有文件：" + name)
        content = read_bytes(source)
        _require(_hash(content) == binding.get("sha256"), "来源文件已变化：" + name)
        paths[name], hashes[name] = source, _hash(content)
        if name in SOURCE_NAMES:
            documents[name] = json.loads(content)

    questions, all_cases = load_cases(paths["questions"])
    truth, split, dataset = (documents[name] for name in ("ground_truth", "split", "dataset_manifest"))
    _require(questions == documents["questions"], "准备期间题库发生变化")
    for document in (questions, truth):
        _require(document.get("data_version") == DATA_VERSION
                 and document.get("reviewed_by_user") is True, "须使用用户已核验的题库及标准")
    _require(documents["business_context"].get("data_version") == DATA_VERSION,
             "公共背景数据版本不符")
    config = {"data_version": DATA_VERSION, "split_file": str(paths["split"]),
              "case_partition": "formation", "questions": str(paths["questions"]),
              "ground_truth": str(paths["ground_truth"]),
              "reference_validation": str(paths["reference_validation"]),
              "exclude_case_ids": split.get("excluded_case_ids", [])}
    cases = select_cases(config, all_cases)
    _require(dataset.get("baseline_id") == DATA_VERSION and dataset.get("schema_version") == "3.0"
             and dataset.get("mysql", {}).get("database") == DATABASE, "数据集身份不符")
    counts = dataset.get("record_counts")
    _require(isinstance(counts, dict) and len(counts) == 7
             and all(isinstance(k, str) and re.fullmatch(r"[a-z][a-z0-9_]*", k)
                     and type(v) is int and v >= 0 for k, v in counts.items()), "数据集记录数无效")

    fact_list = truth.get("cases", [])
    facts = {fact.get("id"): fact for fact in fact_list}
    _require(len(facts) == len(fact_list), "标准题号重复")
    standards = {}
    for case in cases:
        fact = facts.get(case["id"], {})
        expected = fact.get("expected_result")
        _require(fact.get("response_type") != "clarify"
                 and isinstance(expected, dict) and expected.get("is_complete", True) is True
                 and compare_results(expected, expected), "标准结果缺失或无效：" + case["id"])
        # 不读取、执行或导出 reference_sql，候选接口也不接收标准结果。
        standards[case["id"]] = deepcopy({key: expected[key] for key in ("columns", "rows")})

    authored = bundle.get("cases")
    _require(isinstance(authored, list) and all(isinstance(item, dict) for item in authored)
             and [item.get("case_id") for item in authored] == [case["id"] for case in cases],
             "预编写 SQL 须按固定顺序覆盖全部形成题")
    statements = {}
    for item in authored:
        _require(isinstance(item.get("sql_file"), str), "SQL 文件路径缺失")
        relative = Path(item["sql_file"])
        source = (path.parent / relative).resolve()
        _require(not relative.is_absolute() and source.is_relative_to(path.parent)
                 and source.is_file(), "SQL 文件必须位于经验包目录内")
        content = read_bytes(source)
        _require(_hash(content) == item.get("sql_sha256"), "预编写 SQL 已变化：" + item["case_id"])
        sql = content.decode("utf-8").strip()
        _require(bool(sql), "SQL 不能为空：" + item["case_id"])
        statements[item["case_id"]] = sql
    # select_cases/load_cases 会读取同一输入；最终核对避免准备过程中产生混合版本。
    for source, content in snapshots.items():
        _require(source.read_bytes() == content, "准备期间输入发生变化：" + str(source))
    return AuthoredBundle(path, _hash(raw_bundle), bundle["namespace"], cases,
                          statements, standards, counts, hashes)


def _make_environment(**options):
    from src.chatflow_agent.database import BusinessMySQLReadOnlyEnvironment
    return BusinessMySQLReadOnlyEnvironment(**options, max_total_rows=10000,
                                           fetch_batch_size=500, statement_timeout_ms=20000)


def _make_collector(path, namespace):
    from ..integration.direct_update import DirectCandidateCollector
    return DirectCandidateCollector(path, namespace, DATA_VERSION)


def _connection_live(environment, connection):
    try:
        return (connection is not None and environment.connection is connection
                and connection.is_connected())
    except Exception:
        return False


def _diagnostic(error):
    return {"error_type": type(error).__name__, "message": str(error)[:1000],
            "mysql_errno": getattr(error, "errno", None)}


def _actual_record(case, row, run_id):
    """仅把实际执行证据交给已有候选保存器。"""
    actual = row["actual"]
    return {
        "question": case["question"], "answer": "", "baseline_id": DATA_VERSION,
        "session_id": case["session"], "turn_index": case["turn"],
        "status": "completed", "mode": "authored_sql", "experiment_id": run_id,
        "result_query_id": row["query_id"],
        "operations": [{"kind": "sql", "query_id": row["query_id"], "sql": row["sql"],
                        "success": True, "is_complete": True,
                        "columns": deepcopy(actual["columns"]), "rows": deepcopy(actual["rows"]),
                        "row_count": len(actual["rows"])}],
        "completed_task_usage": {"elapsed_seconds": row["cost"]["database_seconds"],
                                 "model_call_count": 0, "prompt_tokens": 0,
                                 "completion_tokens": 0, "embedding_input_tokens": 0},
    }


def run_authored_formation(*, run_dir, bundle_path=DEFAULT_BUNDLE,
                           host="127.0.0.1", port=3306, user="synthetic", database=DATABASE,
                           password_env="ENTERPRISE_MYSQL_PASSWORD"):
    """用户调用的执行入口；一份只读事务，不重连，不重试，不覆盖旧目录。"""
    from ..integration.direct_update import DirectUpdateIntegrityError

    _require(database == DATABASE, "必须使用指定的数据集数据库")
    loaded = load_bundle(bundle_path)
    run = resolve_project_path(run_dir)
    if run.exists():
        raise FileExistsError("运行目录已存在，请指定全新目录：" + str(run))
    run.mkdir(parents=True)
    results = run / "results"
    results.mkdir()
    run_id = "authored-" + uuid4().hex[:16]
    started = datetime.now(timezone.utc).isoformat()
    wall_start = perf_counter()
    collector = environment = connection = None
    stop_reason = None
    stop_detail = {}
    candidate_errors = []
    rows = []
    costs = {"database_setup_seconds": 0.0, "database_query_seconds": 0.0,
             "comparison_seconds": 0.0, "candidate_seconds": 0.0}
    try:
        start = perf_counter()
        try:
            collector = _make_collector(run / "candidates.sqlite3", loaded.namespace)
        except Exception as error:
            stop_reason, stop_detail = "candidate_setup_failed", _diagnostic(error)
        finally:
            costs["candidate_seconds"] += perf_counter() - start
        if stop_reason is None:
            start = perf_counter()
            try:
                environment = _make_environment(database=database, host=host, port=port,
                                                user=user, password_env=password_env)
                connection = environment.connection
                connection.start_transaction(isolation_level="REPEATABLE READ",
                                             consistent_snapshot=True, readonly=True)
                environment.verify_record_counts(loaded.record_counts)
                if not _connection_live(environment, connection):
                    raise RuntimeError("数据库快照连接已丢失")
            except Exception as error:
                stop_reason, stop_detail = "database_setup_failed", _diagnostic(error)
            finally:
                costs["database_setup_seconds"] += perf_counter() - start

        for position, case in enumerate(loaded.cases, 1):
            print(f"[authored {position}/{len(loaded.cases)}] {case['question']}", flush=True)
            row = {"case_id": case["id"], "question": case["question"],
                   "context": deepcopy(case["history"]), "data_version": DATA_VERSION,
                   "session_id": case["session"], "turn": case["turn"],
                   "query_id": case["id"] + "-authored", "sql": loaded.statements[case["id"]],
                   "actual": None, "verdict": "fail", "executed": False, "error": None,
                   "cost": {"database_seconds": 0.0, "comparison_seconds": 0.0}}
            if stop_reason is None and not _connection_live(environment, connection):
                stop_reason = "snapshot_connection_lost"
            if stop_reason is not None:
                row["error"] = {"reason": "not_executed", "cause": stop_reason, **stop_detail}
            else:
                start = perf_counter()
                row["executed"] = True
                try:
                    row["actual"] = environment.execute_query(row["sql"])
                except Exception as error:
                    row["error"] = {"reason": "query_error", **_diagnostic(error)}
                finally:
                    row["cost"]["database_seconds"] = perf_counter() - start
                if row["error"] is None:
                    start = perf_counter()
                    actual = row["actual"]
                    if not isinstance(actual, dict) or actual.get("is_complete") is not True:
                        row["error"] = {"reason": "incomplete_result"}
                    elif not compare_results(actual, loaded.standards[case["id"]]):
                        row["error"] = {"reason": "result_mismatch"}
                    else:
                        row["verdict"] = "pass"
                    row["cost"]["comparison_seconds"] = perf_counter() - start
                if not _connection_live(environment, connection):
                    stop_reason = "snapshot_connection_lost"
            costs["database_query_seconds"] += row["cost"]["database_seconds"]
            costs["comparison_seconds"] += row["cost"]["comparison_seconds"]
            path = results / (case["id"] + ".json")
            # PASS 来源先落盘，再保存候选；之后不改写，保持来源哈希有效。
            write_json(path, row)
            rows.append(row)
            if row["verdict"] == "pass":
                start = perf_counter()
                try:
                    collector.apply(case, _actual_record(case, row, run_id), path, row["query_id"])
                except Exception as error:
                    candidate_errors.append({"case_id": case["id"], **_diagnostic(error)})
                    if isinstance(error, DirectUpdateIntegrityError):
                        stop_reason = "candidate_integrity_error"
                finally:
                    costs["candidate_seconds"] += perf_counter() - start
            detail = "" if row["error"] is None else " (" + row["error"]["reason"] + ")"
            print("  " + row["verdict"] + detail, flush=True)
    finally:
        if environment is not None:
            try:
                if environment.connection is connection and connection is not None:
                    connection.rollback()
            except Exception:
                pass
            environment.close()
        if collector is not None:
            collector.close()

    candidates = {"candidate_count": 0, "execution_ids": [], "source_refs": []}
    if collector is not None:
        try:
            candidates = collector.summary()
        except Exception as error:
            candidate_errors.append({"case_id": None, **_diagnostic(error)})
            candidates = {"candidate_count": None, "execution_ids": [], "source_refs": []}
    passed = sum(row["verdict"] == "pass" for row in rows)
    executed = sum(row["executed"] for row in rows)
    summary = {
        "version": VERSION, "policy": POLICY, "run_id": run_id, "run_dir": str(run),
        "data_version": DATA_VERSION, "namespace": loaded.namespace,
        "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
        "bundle": {"path": str(loaded.path), "sha256": loaded.sha256},
        "source_sha256": loaded.source_hashes,
        "case_count": len(rows), "executed_count": executed,
        "not_executed_count": len(rows) - executed, "pass_count": passed,
        "fail_count": len(rows) - passed, "execution_complete": executed == len(rows),
        "candidate_save_complete": (collector is not None and not candidate_errors
                                    and candidates["candidate_count"] == passed),
        "stop_reason": stop_reason, "candidate_errors": candidate_errors,
        **candidates,
        "cost": {**costs, "elapsed_seconds": perf_counter() - wall_start,
                 "model_call_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
                 "embedding_input_tokens": 0,
                 "scope": "本次数据库执行、比较与候选保存；不包含预先编写SQL的成本"},
    }
    write_json(run / "summary.json", summary)
    return summary
