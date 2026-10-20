"""按固定规则把已核验的逐题JSON整理为共同种子；不访问数据库或模型。"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import sqlglot

from src.project_paths import resolve_project_path
from ..pool.classification import classify
from ..pool.pool_models import GROUP_IDS, GROUP_RULE_VERSION, GroupingInput
from .authored_formation import DATA_VERSION, NAMESPACE, VERSION as FORMATION_VERSION, load_bundle
from .common import POLICY
from .comparison import compare_results


VERSION = "psu_initial_seed_v1"
DEFAULT_SEED_BUNDLE = "data/experiences/psu_memory/enterprise_v30_120_v1_0_2_initial_seed_v1/seeds.json"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value):
    # 不折叠SQL内部空白，以免改变字符串字面量；与固定划分中的去重规则一致。
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def build_seed_bundle(formation_run):
    """读取真实PASS文件，按预定优先序、精确去重及每组容量选择。"""
    _require(sqlglot.__version__ == "26.7.0", "请使用requirements-psu.txt固定的sqlglot==26.7.0")
    root = resolve_project_path(".")
    run = resolve_project_path(formation_run)
    snapshots = {}

    def read(path):
        path = path.resolve()
        _require(path.is_relative_to(root), "种子来源须位于项目内")
        content = path.read_bytes()
        if path in snapshots:
            _require(content == snapshots[path], "选种期间来源发生变化")
        snapshots[path] = content
        return content

    def binding(path):
        path = path.resolve()
        return {"path": path.relative_to(root).as_posix(), "sha256": _sha(read(path))}

    summary_path = run / "summary.json"
    summary = json.loads(read(summary_path))
    _require(summary.get("version") == FORMATION_VERSION and summary.get("policy") == POLICY
             and summary.get("data_version") == DATA_VERSION and summary.get("namespace") == NAMESPACE,
             "须使用当前预编写SQL形成记录")
    _require(summary.get("execution_complete") is True and summary.get("candidate_save_complete") is True
             and not summary.get("candidate_errors") and summary.get("stop_reason") is None,
             "形成执行或候选保存尚未完整完成")
    authored_path = resolve_project_path(summary["bundle"]["path"])
    _require(_sha(read(authored_path)) == summary["bundle"]["sha256"], "形成SQL包已变化")
    loaded = load_bundle(authored_path)
    _require(summary.get("source_sha256") == loaded.source_hashes, "形成来源哈希不一致")
    authored = json.loads(read(authored_path))
    split_path = resolve_project_path(authored["sources"]["split"]["path"])
    split = json.loads(read(split_path))
    rules = split["seed_selection"]
    _require(rules["max_total"] == 18 and rules["max_per_group"] == 8
             and rules["deduplication"] == "exact_question_context_sql_v1", "共同种子规则不符")
    priority = rules["priority_case_ids"]
    case_ids = [case["id"] for case in loaded.cases]
    _require(len(priority) == len(set(priority)) and set(priority) == set(case_ids), "优先序未覆盖形成题")
    _require({p.stem for p in (run / "results").glob("*.json")} == set(case_ids), "形成结果文件不完整")
    records, sources, execution_ids = {}, {}, []
    for case in loaded.cases:
        case_id = case["id"]
        path = run / "results" / (case_id + ".json")
        row = json.loads(read(path))
        _require(row.get("case_id") == case_id and row.get("question") == case["question"]
                 and row.get("context") == case["history"] and row.get("session_id") == case["session"]
                 and row.get("turn") == case["turn"] and row.get("data_version") == DATA_VERSION
                 and row.get("sql") == loaded.statements[case_id]
                 and row.get("query_id") == case_id + "-authored", "形成内容与固定来源不一致：" + case_id)
        _require(row.get("executed") is True and row.get("verdict") in ("pass", "fail"), "形成记录无效")
        if row["verdict"] != "pass":
            continue
        actual = row.get("actual") or {}
        _require(row.get("error") is None and actual.get("is_complete") is True
                 and actual.get("row_count") == len(actual.get("rows", []))
                 and compare_results(actual, loaded.standards[case_id]), "PASS未绑定完整一致的实际结果：" + case_id)
        records[case_id], sources[case_id] = row, binding(path)
        identity = json.dumps([NAMESPACE, DATA_VERSION, case["session"], case["turn"]],
                              ensure_ascii=False, separators=(",", ":"))
        execution_ids.append("business:" + _sha(identity.encode()))
    _require(summary.get("case_count") == len(case_ids) and summary.get("executed_count") == len(case_ids)
             and summary.get("not_executed_count") == 0 and summary.get("pass_count") == len(records)
             and summary.get("fail_count") == len(case_ids) - len(records)
             and summary.get("candidate_count") == len(records)
             and summary.get("execution_ids") == execution_ids
             and [resolve_project_path(p) for p in summary.get("source_refs", [])]
                 == [resolve_project_path(source["path"]) for source in sources.values()], "形成汇总与逐题候选不一致")

    selected, seen = [], set()
    group_counts = Counter({group: 0 for group in GROUP_IDS})
    skipped = {"duplicate": [], "group_full": [], "not_pass": []}
    scanned = 0
    for case_id in priority:
        scanned += 1
        if case_id not in records:
            skipped["not_pass"].append(case_id)
            continue
        row = records[case_id]
        key = _sha(_canonical([_text(row["question"]),
                               [_text(message["content"]) for message in row["context"][-2:]],
                               _text(row["sql"])]).encode())
        if key in seen:
            skipped["duplicate"].append(case_id)
            continue
        seen.add(key)
        decision = classify(GroupingInput(execution_id=case_id, content_version=sources[case_id]["sha256"],
                                         namespace=NAMESPACE, data_version=DATA_VERSION,
                                         question=row["question"], sql=row["sql"]))
        group = decision.group_id
        if group_counts[group] >= rules["max_per_group"]:
            skipped["group_full"].append(case_id)
            continue
        group_counts[group] += 1
        selected.append({"case_id": case_id, "group": group, "source": sources[case_id],
                         "record": deepcopy(row)})
        if len(selected) == rules["max_total"]:
            break
    _require(bool(selected), "没有满足规则的有效共同种子")
    metadata = {case["id"]: case for case in loaded.cases}
    distributions = {"group": dict(group_counts)}
    for label, field in (("category", "category"), ("learning_stage", "learning_stage"),
                         ("question_bank_difficulty", "difficulty")):
        distributions[label] = dict(sorted(Counter(metadata[item["case_id"]][field] for item in selected).items()))
    payload = {
        "version": VERSION, "data_version": DATA_VERSION, "namespace": NAMESPACE,
        "sources": {"summary": binding(summary_path), "authored_bundle": binding(authored_path),
                    "split": binding(split_path),
                    "classification": binding(root / "src/psu_memory/pool/classification.py"),
                    "selection": binding(Path(__file__))},
        "classification": {"rule_version": GROUP_RULE_VERSION, "sqlglot_version": sqlglot.__version__},
        "selection": {"max_total": rules["max_total"], "max_per_group": rules["max_per_group"],
                      "deduplication": rules["deduplication"], "scanned_count": scanned, "skipped": skipped},
        "distribution": distributions, "entries": selected,
    }
    for path, content in snapshots.items():
        _require(path.read_bytes() == content, "选种期间来源发生变化：" + str(path))
    return payload


def load_seed_bundle(path=DEFAULT_SEED_BUNDLE):
    """初始化前重新核对选择结果，不能修改题号/分组或夹带未选中的候选。"""
    path = resolve_project_path(path)
    payload = json.loads(path.read_bytes())
    _require(payload.get("version") == VERSION, "共同种子包版本不符")
    summary_path = resolve_project_path(payload["sources"]["summary"]["path"])
    expected = build_seed_bundle(summary_path.parent)
    _require(payload == expected, "共同种子包与固定规则、当前代码或原始形成来源不一致")
    return payload


def prepare_seed_bundle(formation_run, destination=DEFAULT_SEED_BUNDLE):
    path = resolve_project_path(destination)
    payload = build_seed_bundle(formation_run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return payload
