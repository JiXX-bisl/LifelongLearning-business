#!/usr/bin/env python3
"""用户运行的题库结构及参考结果核验；不调用模型，不修改候选基线。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PUBLIC = ROOT / "data/questions/psu_memory/enterprise_v30_120_v1_0_2_draft"
PRIVATE = ROOT / "data/evaluation/psu_memory/enterprise_v30_120_v1_0_2_draft"
BASELINE = ROOT / "data/sql/synthetic_enterprise_v30_120_v1.0.2"
CATEGORIES = ("A1", "A2", "A4", "A5", "B2", "B3", "C1", "C2", "C4", "C6")
BASELINE_ID = "synthetic_enterprise_v30_120_v1.0.2"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    """授权记录绑定完整对象；与文件字节哈希分别校验。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_authorized_case_changes(original, current, authorizations, *, reference=False):
    """只接受逐题、逐字段及完整旧/新签名均匹配的用户授权差异。"""
    require(isinstance(authorizations, dict), "人工授权差异必须为逐题对象")
    require(set(original) == set(current), "原目标或参考SQL题号发生变化")
    require(set(authorizations).issubset(original), "人工授权差异含未知题号")
    prefix = "reference" if reference else "goal"
    changed_ids = []
    for case_id, old in original.items():
        new = current[case_id]
        require(isinstance(old, dict) and isinstance(new, dict), "目标或参考SQL必须为结构化对象：" + case_id)
        # 原业务目标允许追加自然题的私有注释；原有字段仍逐项保留。
        # reference是执行契约，完整比较，不允许静默增删其字段。
        keys = set(old) | set(new) if reference else set(old)
        changed = {key: {"old": old.get(key), "new": new.get(key)}
                   for key in keys if key not in old or key not in new or old[key] != new[key]}
        authorization = authorizations.get(case_id)
        if not changed:
            require(authorization is None, "人工授权记录已失配或没有对应差异：" + case_id)
            continue
        require(isinstance(authorization, dict), "未授权改写原目标或参考SQL：" + case_id)
        require(all(isinstance(authorization.get(key), str) and authorization[key].strip()
                    for key in ("reason", "source")), "人工授权差异缺少理由或用户审阅来源：" + case_id)
        require(authorization.get("old_" + prefix + "_sha256") == canonical_sha256(old)
                and authorization.get("new_" + prefix + "_sha256") == canonical_sha256(new),
                "人工授权差异的旧或新完整对象签名失配：" + case_id)
        require(authorization.get("field_changes") == changed,
                "人工授权字段差异与实际改动不完全一致：" + case_id)
        if reference:
            require(set(changed) == {"sql"}, "人工授权参考契约仅允许显式SQL修订：" + case_id)
        else:
            require(not (set(changed) & {"id", "question_id", "category", "learning_stage",
                                        "difficulty", "difficulty_definition", "response_type"}),
                    "人工目标授权不能改变题号、类别、阶段、难度或响应契约：" + case_id)
        changed_ids.append(case_id)
    return sorted(changed_ids)


def validate_goal_preservation(original_facts, facts, original_rules, rules, approval=None):
    """保持原目标及响应契约，仅逐项允许已绑定用户证据的业务口径修订。"""
    if approval is None:
        goals, references = {}, {}
    else:
        require(isinstance(approval, dict) and approval.get("status") == "HUMAN_REVIEW_APPROVED",
                "人工审核报告缺少明确通过状态")
        goals = approval.get("authorized_goal_changes", {})
        references = approval.get("authorized_reference_changes", {})
    goal_changes = validate_authorized_case_changes(original_facts, facts, goals)
    for name in ("expected_types", "sql_requirements"):
        require(rules.get(name) == original_rules.get(name), "自然题改写了原响应或SQL契约：" + name)
    reference_changes = validate_authorized_case_changes(original_rules.get("references", {}),
                                                        rules.get("references", {}), references,
                                                        reference=True)
    return {"authorized_goal_change_ids": goal_changes,
            "authorized_reference_change_ids": reference_changes}


def static_check(question_dir=PUBLIC, private_dir=PRIVATE):
    from src.psu_memory.validation.common import load_cases, load_public_context
    from src.psu_memory.question_bank_contracts import bind_case_contracts, validate_intent_contracts
    from src.lifelong_learning.environments.fixed_mysql_environment import validate_read_only_sql

    question_dir, private_dir = Path(question_dir), Path(private_dir)
    document, cases = load_cases(question_dir / "questions.json")
    truth = read(private_dir / "ground_truth.json")
    rules = read(private_dir / "review_rules.json")
    manifest = read(private_dir / "bundle_manifest.json")
    require(document["data_version"] == BASELINE_ID, "题库数据版本不匹配")
    for private in (truth, rules, manifest):
        require(private.get("data_version") == BASELINE_ID, "私有依据数据版本不匹配")
    for private in (document, truth, manifest):
        require(private.get("schema_version") == "3.0", "题库Schema版本不匹配")
    ids = {case["id"] for case in cases}
    facts = {item["id"]: item for item in truth["cases"]}
    require(len(facts) == len(truth["cases"]), "标准依据题号重复")
    require(set(facts) == ids == set(rules["rubrics"]) == set(rules["expected_types"]), "题目、标准依据与审核类型未完整对应")
    require(set(rules["sql_requirements"]) == ids, "SQL审核契约未覆盖全部题目")
    require(set(rules.get("references", {})).issubset(ids), "参考SQL出现未知题号")
    bind_case_contracts(cases, rules)
    intent_summary = None
    if rules.get("intent_contract_version") is not None:
        contracts = validate_intent_contracts(read(private_dir / "intent_contracts.json"), cases,
                                             data_version=BASELINE_ID)
        load_public_context(question_dir / "business_context.json", data_version=BASELINE_ID)
        original_truth = read(private_dir / "source_snapshot/ground_truth.json")
        original_facts = {item["id"]: item for item in original_truth["cases"]}
        original_questions = {item["id"]: item for item in read(private_dir / "source_snapshot/questions.json")["cases"]}
        public_questions = {item["id"]: item for item in cases}
        require(set(original_facts) == ids, "原业务目标快照未完整对应自然题")
        original_rules = read(private_dir / "source_snapshot/review_rules.json")
        approval_path = private_dir / "approval_report.json"
        approval = read(approval_path) if approval_path.exists() else None
        if approval is not None:
            report_relative = str(approval_path.resolve().relative_to(ROOT))
            require(manifest.get("artifact_sha256", {}).get(report_relative) == digest(approval_path),
                    "人工审核报告必须以文件哈希绑定到当前题库包")
        preservation = validate_goal_preservation(original_facts, facts, original_rules, rules, approval)
        for contract in contracts["cases"]:
            case_id = contract["id"]
            require(contract.get("actual_question") == public_questions[case_id]["question"],
                    "私有意图合同与实际自然问句不同：" + case_id)
            require(contract.get("standard_question") == original_questions[case_id]["question"],
                    "原标准问题与源快照不同：" + case_id)
            require(isinstance(contract.get("expected_intent"), str) and contract["expected_intent"].strip(),
                    "自然题缺少预期业务目的：" + case_id)
            columns = set(facts[case_id].get("expected_result", {}).get("columns", []))
            target_columns = set()
            for target in contract["required_business_targets"]:
                mapped = target.get("standard_result_columns")
                require(isinstance(mapped, list) and all(isinstance(column, str) for column in mapped),
                        "业务目标必须显式映射标准结果列：" + case_id)
                target_columns.update(mapped)
            technical_columns = set()
            for excluded in contract.get("technical_reference_columns", []):
                if isinstance(excluded, str):
                    name, reason = excluded, contract.get("technical_reference_column_notes", {}).get(excluded)
                else:
                    require(isinstance(excluded, dict), "技术参考列须明确列名与理由：" + case_id)
                    name, reason = excluded.get("column"), excluded.get("reason")
                require(isinstance(name, str) and isinstance(reason, str) and reason.strip(),
                        "技术参考列排除缺少说明：" + case_id)
                technical_columns.add(name)
            require(target_columns | technical_columns == columns,
                    "标准业务结果列未完整对应目标或技术列依据：" + case_id)
            require(not (target_columns & technical_columns), "同一标准列不能同时归为业务目标及技术列：" + case_id)
        intent_summary = {"contract_version": contracts["intent_contract_version"],
                          "artifact_version": contracts["version"],
                          "case_count": len(contracts["cases"]),
                          "pending_review_count": len(contracts["pending_reviews"]),
                          "reviewed_by_user": contracts.get("reviewed_by_user") is True,
                          "original_business_targets_preserved": not preservation["authorized_goal_change_ids"],
                          "original_business_targets_preserved_except_user_authorized_changes": True,
                          **preservation,
                          "execution_ready": not contracts["pending_reviews"]
                          and contracts.get("reviewed_by_user") is True}
    for relative, expected in manifest["artifact_sha256"].items():
        require(digest(ROOT / relative) == expected, "题库产物已变化：" + relative)
    for relative, expected in manifest["source_sha256"].items():
        require(digest(ROOT / relative) == expected, "题库源依据已变化：" + relative)
    tables = read(BASELINE / "tables.json")
    scenario_ids = {item["scenario_id"] for item in read(ROOT / "data/evaluation" / BASELINE_ID / "scenarios.json")}
    coverage = Counter()
    response_counts = Counter()
    reference_count = 0
    for case in cases:
        case_id = case["id"]
        item = facts[case_id]
        require(case["category"] in CATEGORIES, "未知业务类别：" + case_id)
        require(case["learning_stage"] in ("L1", "L2", "L3"), "未知学习阶段：" + case_id)
        require(case["difficulty"] in ("L1", "L2", "L3", "L4"), "无效旧难度枚举：" + case_id)
        require(item["response_type"] in ("answer", "clarify", "insufficient"), "未知响应类型：" + case_id)
        require(item["response_type"] == rules["expected_types"][case_id], "响应类型不同：" + case_id)
        require(item.get("target_scope") and item.get("expected_behavior"), "缺少业务范围或标准行为：" + case_id)
        require(item.get("public_evidence"), "缺少公开证据：" + case_id)
        require(item.get("scenario_ids") and set(item["scenario_ids"]).issubset(scenario_ids), "无效私有场景索引：" + case_id)
        for evidence in item["public_evidence"]:
            table = evidence["table"]
            require(table in tables, "未知公开表：" + table)
            key = evidence.get("primary_key", evidence.get("business_key", {}))
            require(key, "公开证据缺少主键或业务键：" + case_id)
            matched = [row for row in tables[table] if all(row.get(k) == v for k, v in key.items())]
            require(matched, "公开证据未找到：" + case_id)
            for field in evidence.get("fields", []):
                require(all(field in row for row in matched), "公开证据字段不存在：" + field)
        for partner in item.get("paired_question_ids", []):
            require(partner in ids and partner != case_id, "无效配对题号：" + case_id)
        reference = rules.get("references", {}).get(case_id)
        requirement = rules["sql_requirements"][case_id]
        require(requirement in ("required", "optional", "not_applicable"), "无效SQL要求：" + case_id)
        if requirement == "required":
            require(reference is not None, "必需SQL题缺少参考：" + case_id)
        if reference is not None:
            validate_read_only_sql(reference["sql"])
            require(reference["sql"] == item.get("reference_sql"), "两处参考SQL不同：" + case_id)
            expected = item.get("expected_result")
            require(isinstance(expected, dict) and isinstance(expected.get("columns"), list)
                    and isinstance(expected.get("rows"), list), "标准结果结构无效：" + case_id)
            require(all(len(row) == len(expected["columns"]) for row in expected["rows"]), "标准结果列宽不同：" + case_id)
            require(isinstance(reference.get("aliases"), dict), "列别名映射格式错误：" + case_id)
            reference_count += 1
        if case["category"] == "C6":
            require(item.get("state") is not None and item.get("operations") is not None, "C6缺少条件状态或操作：" + case_id)
        require(not any(key in case for key in ("response_type", "expected_type", "reference_sql", "expected_result", "public_evidence", "scenario_ids", "state", "operations", "expected_intent", "required_business_targets", "intent_rubric", "intent_contract_version", "intent_alignment_status")), "私有依据进入公开题目：" + case_id)
        coverage[case["category"], case["learning_stage"]] += 1
        response_counts[item["response_type"]] += 1
    require(all(coverage[category, stage] >= 2 for category in CATEGORIES for stage in ("L1", "L2", "L3")), "首批业务类别与学习阶段覆盖不足")
    return {"status": "PASS", "baseline_id": BASELINE_ID, "case_count": len(cases),
            "reference_count": reference_count, "response_type_counts": dict(response_counts),
            "coverage": {f"{category}:{stage}": count for (category, stage), count in sorted(coverage.items())},
            "intent_contracts": intent_summary,
            "database_checked": False, "formal_business_acceptance": "user_pending",
            "question_review": "user_approved" if document.get("reviewed_by_user") is True else "user_pending"}, cases, facts, rules


def database_check(report, facts, rules, args):
    from src.chatflow_agent.database import BusinessMySQLReadOnlyEnvironment
    from src.lifelong_learning.tasks.enterprise_query_bench.evaluator import compare_query_results
    from src.lifelong_learning.tasks.enterprise_query_bench.dataset import RowOrder

    require(args.database == "synthetic_enterprise_v30_120", "参考核验只允许目标新数据库")
    environment = BusinessMySQLReadOnlyEnvironment(
        database=args.database, host=args.host, port=args.port, user=args.user,
        password_env="ENTERPRISE_MYSQL_PASSWORD", max_total_rows=10000,
        fetch_batch_size=500, statement_timeout_ms=20000,
    )
    checks = []
    try:
        environment.connection.start_transaction(consistent_snapshot=True, readonly=True)
        environment.verify_record_counts(read(BASELINE / "manifest.json")["record_counts"])
        for case_id, reference in rules["references"].items():
            expected = facts[case_id]["expected_result"]
            try:
                actual = environment.execute_query(reference["sql"])
                columns_equal = actual["columns"] == expected["columns"]
                matched = (actual["is_complete"] and columns_equal
                           and compare_query_results(actual["rows"], expected["rows"], RowOrder.UNORDERED))
                checks.append({"id": case_id, "status": "PASS" if matched else "FAIL",
                               "columns_equal": columns_equal, "actual": actual,
                               "expected": expected})
                if not actual["is_complete"] or environment.connection is None:
                    # No reconnect is permitted inside a single-snapshot run.
                    break
            except Exception as error:
                checks.append({"id": case_id, "status": "ERROR", "error_type": type(error).__name__})
                # The environment may discard a failed connection. A retry
                # would lose the single-snapshot contract, so stop this run.
                break
    finally:
        if environment.connection is not None:
            environment.connection.rollback()
        environment.close()
    report.update(database_checked=True, readonly=True, reference_checks=checks,
                  all_references_checked=len(checks) == len(rules["references"]))
    if not report["all_references_checked"] or any(item["status"] != "PASS" for item in checks):
        report["status"] = "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-dir", type=Path, default=PUBLIC, help="公开题库目录；默认保留旧精确问法草案")
    parser.add_argument("--private-dir", type=Path, default=PRIVATE, help="私有验证依据目录；自然题必须包含意图合同和源快照")
    parser.add_argument("--check-database", action="store_true", help="用户显式启动全部参考SELECT与静态标准结果比对")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", default="synthetic")
    parser.add_argument("--database", default="synthetic_enterprise_v30_120")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    destination = args.report if args.report.is_absolute() else ROOT / args.report
    require(destination.resolve().is_relative_to(ROOT / "data/reports"), "报告须保存到独立data/reports目录")
    require(not destination.exists(), "报告路径已经存在，请使用新文件名")
    try:
        question_dir = args.question_dir if args.question_dir.is_absolute() else ROOT / args.question_dir
        private_dir = args.private_dir if args.private_dir.is_absolute() else ROOT / args.private_dir
        report, _, facts, rules = static_check(question_dir, private_dir)
        if args.check_database:
            report = database_check(report, facts, rules, args)
    except Exception as error:
        report = {"status": "FAIL", "error_type": type(error).__name__}
        if isinstance(error, (ValueError, FileNotFoundError)):
            report["message"] = str(error)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "database_checked": report.get("database_checked", False),
                      "report": str(destination)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
