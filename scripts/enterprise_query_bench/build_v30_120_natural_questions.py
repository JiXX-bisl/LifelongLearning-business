#!/usr/bin/env python3
"""部署自然提问v0.2及意图依据，原标准答案从稳定快照逐项保留。"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "enterprise_v30_120_v1_0_2_natural_v0_2_draft"
PUBLIC = ROOT / "data/questions/psu_memory" / NAME
PRIVATE = ROOT / "data/evaluation/psu_memory" / NAME
SNAPSHOT = PRIVATE / "source_snapshot"
BASELINE_ID = "synthetic_enterprise_v30_120_v1.0.2"
BASELINE = ROOT / "data/sql" / BASELINE_ID
PUBLIC_FIELDS = ("id", "session", "turn", "question", "history", "category", "learning_stage", "difficulty")
GOAL_FIELDS = ("expected_result", "expected_behavior", "response_type", "target_scope", "reference_sql",
               "state", "operations", "enterprise_ids", "sql_requirement", "numeric_tolerances")
ALIGNMENTS = ("supported", "convention_needs_review", "scope_conflict")
BASIS_KINDS = ("question_wording", "public_dictionary", "public_records", "conversation_context", "business_convention")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def goal_signature(item):
    value = {field: item.get(field) for field in GOAL_FIELDS}
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def user_revisions():
    result = {}
    review = (SNAPSHOT / "user_review.md").read_text(encoding="utf-8")
    for section in re.split(r"(?m)^## ", review)[1:]:
        case_id = section.split(" · ", 1)[0]
        match = re.search(r"\*\*修改版：(.*?)\*\*", section, re.S)
        if match:
            result[case_id] = match.group(1).strip()
    return result


def build():
    old_questions = read(SNAPSHOT / "questions.json")
    old_truth = read(SNAPSHOT / "ground_truth.json")
    old_rules = read(SNAPSHOT / "review_rules.json")
    source_matrix = read(SNAPSHOT / "coverage_matrix.json")
    source_manifest = read(SNAPSHOT / "source_manifest.json")
    for record in source_manifest["files"]:
        if sha((SNAPSHOT / record["snapshot_file"]).read_bytes()) != record["sha256"]:
            raise ValueError("源快照已变化：" + record["snapshot_file"])
    by_id = {case["id"]: case for case in old_questions["cases"]}
    old_goals = {item["id"]: item for item in old_truth["cases"]}
    natural, contracts = {}, {}
    overlay_paths = [PRIVATE / "natural_ab_overlay.json", PRIVATE / "natural_c_overlay.json"]
    for path in overlay_paths:
        overlay = read(path)
        for item in overlay["cases"]:
            if item["id"] in natural:
                raise ValueError("自然问法题号重复：" + item["id"])
            natural[item["id"]] = item["question"].strip()
        for contract in overlay["intent_contracts"]:
            if contract["id"] in contracts:
                raise ValueError("意图契约题号重复：" + contract["id"])
            contracts[contract["id"]] = copy.deepcopy(contract)
    if set(natural) != set(contracts) or set(natural) != set(by_id) or set(by_id) != set(old_goals):
        raise ValueError("自然问法、意图契约和原标准目标须完整对应")
    revisions = user_revisions()
    for case_id, question in revisions.items():
        if natural[case_id] != question:
            raise ValueError("用户加粗修改必须原样保留：" + case_id)

    common = {"version": "psu_enterprise_v30_120_natural_v0.2.0", "data_version": BASELINE_ID,
              "schema_version": "3.0", "status": "DRAFT", "reviewed_by_user": False}
    context = {"version": "enterprise_risk_public_context_v0.2", "data_version": BASELINE_ID,
               "messages": [{"role": "user", "content":
                 "本次工作是企业背景、经营情况和风险记录核查。业务观察日为2026年8月31日，"
                 "可查询资料收集至2026年9月15日23:59:59，日期时间按北京时间理解。"
                 "对话中的今年和当前按本次业务观察日理解，回答请交代所用数据的期间及单位。"}]}
    cases, goals, intent_items, pending = [], [], [], []
    history = {}
    for old_case in old_questions["cases"]:
        case_id = old_case["id"]
        question = natural[case_id]
        if not question:
            raise ValueError("自然提问不能为空：" + case_id)
        case = {field: copy.deepcopy(old_case[field]) for field in PUBLIC_FIELDS}
        case["question"] = question
        previous = history.setdefault(case["session"], [])
        if case["turn"] != len(previous) + 1:
            raise ValueError("源会话轮次不连续：" + case_id)
        case["history"] = [{"role": "user", "content": content} for content in previous]
        previous.append(question)
        cases.append(case)
        contract = contracts[case_id]
        contract["standard_question"] = old_case["question"]
        contract["actual_question"] = question
        contract["category"] = case["category"]
        contract["learning_stage"] = case["learning_stage"]
        contract["user_revision_preserved"] = case_id in revisions
        contract["basis_reviewed_by_user"] = False
        contract["public_context_reference"] = str((PUBLIC / "business_context.json").relative_to(ROOT))
        targets = contract.get("required_business_targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("每题必须明确业务目标及依据：" + case_id)
        seen_targets, covered_columns = set(), set()
        result_columns = set((old_goals[case_id].get("expected_result") or {}).get("columns", []))
        for target in targets:
            target_id = target["target_id"]
            if target_id in seen_targets:
                raise ValueError("同题意图目标编号重复：" + case_id)
            seen_targets.add(target_id)
            if not target.get("description") or type(target.get("implicit")) is not bool:
                raise ValueError("目标描述与隐含标记缺失：" + case_id)
            columns = target.get("standard_result_columns", [])
            if not set(columns).issubset(result_columns):
                raise ValueError("目标对应未知标准列：" + case_id)
            covered_columns.update(columns)
            basis = target.get("restoration_basis")
            if not isinstance(basis, list) or not basis:
                raise ValueError("隐含目标缺少还原依据：" + case_id)
            for item in basis:
                if (item.get("kind") not in BASIS_KINDS or not item.get("text")
                        or not item.get("source") or type(item.get("observable")) is not bool):
                    raise ValueError("还原依据须说明类型、内容、来源及可见性：" + case_id)
            status = target.get("alignment_status")
            if status not in ALIGNMENTS:
                raise ValueError("未知意图对齐状态：" + case_id)
            if status == "supported" and not any(item["observable"] for item in basis):
                raise ValueError("已支持目标必须有可见还原依据：" + case_id)
            if status != "supported":
                pending.append({"id": case_id, "target_id": target_id, "status": status,
                                "reason": target["description"], "review_notes": contract.get("review_notes", [])})
        source_technical = contract.get("technical_reference_columns", [])
        technical = {item["column"] if isinstance(item, dict) else item for item in source_technical}
        # Locator and display columns remain in the unchanged oracle. They
        # need not force an exact output shape on the natural-language answer.
        locators = {"enterprise_id", "enterprise_name", "person_entity_id", "person_name", "business_object_ref"}
        unmapped = result_columns - covered_columns - technical - locators
        if unmapped:
            raise ValueError("原业务结果列未映射到意图目标：" + case_id + ":" + ",".join(sorted(unmapped)))
        contract["technical_reference_columns"] = [{"column": column,
            "reason": "企业、人员或事项定位标记；原标准数据仍保留，自然回答不强制技术ID或原列名。"}
            for column in sorted(technical | (result_columns - covered_columns))]
        statuses = {target["alignment_status"] for target in targets}
        contract["overall_alignment_status"] = ("scope_conflict" if "scope_conflict" in statuses else
                                                 "convention_needs_review" if "convention_needs_review" in statuses
                                                 else "supported")
        contract["original_goal_sha256"] = goal_signature(old_goals[case_id])
        intent_items.append(contract)
        goal = copy.deepcopy(old_goals[case_id])
        goal.update(expected_intent=contract["expected_intent"],
                    intent_contract_source=str((PRIVATE / "intent_contracts.json").relative_to(ROOT)),
                    required_business_targets=copy.deepcopy(targets),
                    intent_alignment_status=contract["overall_alignment_status"],
                    original_goal_sha256=contract["original_goal_sha256"])
        if goal_signature(goal) != goal_signature(old_goals[case_id]):
            raise ValueError("原标准答案或处理目标发生变化：" + case_id)
        goals.append(goal)

    questions = {**common, "cases": cases}
    truth = {**copy.deepcopy(old_truth), **common,
             "basis": "自然业务提问、共有背景和公开依据支持意图还原；原标准答案保留为业务目标；待审依据单独记录。",
             "cases": goals}
    intent_document = {**common, "intent_contract_version": "natural_intent_v1",
                       "principles": ["所有实际提问采用自然问法", "原标准答案作为业务目标",
                                      "每项隐含目标记录支持其还原的依据"],
                       "case_count": len(cases), "user_revision_count": len(revisions),
                       "pending_review_target_count": len(pending), "pending_reviews": pending, "cases": intent_items}
    rules = copy.deepcopy(old_rules)
    rules.update(version="psu_enterprise_v30_120_natural_review_v0.2.0", schema_version="3.0",
                 intent_contract_version="natural_intent_v1",
                 intent_contracts=str((PRIVATE / "intent_contracts.json").relative_to(ROOT)),
                 intent_rubrics={item["id"]: "预期业务目的：" + item["expected_intent"] + "。目标覆盖：" +
                                "；".join(target["description"] for target in item["required_business_targets"])
                                for item in intent_items})
    rules["general"] = ["实际提问使用自然问法，原标准答案保持为业务目标。",
                        "逐题意图与还原依据只供验证端；依据未审定的题不能当作模型可唯一还原的已验收题。",
                        "意图理解、查询事实和业务回答分别审核，相关审核完成后才沿正常机制更新经验。",
                        "私有审核理由、标准数值及SQL不得写入PSU反馈内容。",
                        *rules.get("general", [])]

    matrix = {**copy.deepcopy(source_matrix), **common,
              "question_changed_count": sum(case["question"] != by_id[case["id"]]["question"] for case in cases),
              "preserved_original_goal_count": len(goals), "user_revision_count": len(revisions),
              "intent_alignment_counts": dict(Counter(item["overall_alignment_status"] for item in intent_items)),
              "pending_review_target_count": len(pending),
              "evaluation_dimensions": ["intent_understanding", "query_and_calculation", "business_answer", "response_behavior"]}
    contract_map = {item["id"]: item for item in intent_items}
    for item in matrix["cases"]:
        item["actual_question"] = natural[item["id"]]
        item["expected_intent"] = contract_map[item["id"]]["expected_intent"]
        item["intent_alignment_status"] = contract_map[item["id"]]["overall_alignment_status"]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["id", "category", "learning_stage", "difficulty", "response_type", "natural_question",
                     "expected_intent", "alignment_status", "pending_target_count", "paired_question_ids"])
    for case in cases:
        contract = contract_map[case["id"]]
        writer.writerow([case["id"], case["category"], case["learning_stage"], case["difficulty"],
                         old_goals[case["id"]]["response_type"], case["question"], contract["expected_intent"],
                         contract["overall_alignment_status"], sum(target["alignment_status"] != "supported"
                         for target in contract["required_business_targets"]),
                         ";".join(old_goals[case["id"]].get("paired_question_ids", []))])

    statuses_zh = {"supported": "有还原依据", "convention_needs_review": "隐含业务习惯待审", "scope_conflict": "范围冲突待审"}
    lines = ["# 120家 v1.0.2 自然提问与业务意图对照稿", "", "状态：DRAFT，待用户审核。", "",
             "本版采用已确认的三条原则：所有提问自然化；原标准答案作为业务目标；每项隐含目标记录还原依据。",
             "原题库、用户加粗修改及原标准数据均保留在来源快照。模型输入只有自然提问、公开用户历史和共有业务背景。", "",
             "## 共有业务背景", "", context["messages"][0]["content"], "",
             "## 本版记录", "",
             f"72个提问回合；原标准目标保留{len(goals)}项；用户修改原样采用{len(revisions)}处。",
             "意图对齐状态：" + json.dumps(matrix["intent_alignment_counts"], ensure_ascii=False) + "。",
             f"共有{len(pending)}个目标需审核其业务习惯或范围依据；这些记录供审阅，未宣称正式验收。", "",
             "审核方式：确认自然问法是否表达原业务目的，核对隐含目标依据；范围冲突单独处理。自然语言回答接受等价表述，不要求原SQL列名和表格形式。"]
    for case in cases:
        case_id = case["id"]
        goal = old_goals[case_id]
        contract = contract_map[case_id]
        lines.extend(["", f"## {case_id} · {goal['response_type']} · 难度 {case['difficulty']}", "",
                      "**自然提问：" + case["question"] + "**", "",
                      "原标准问题（验证端）：" + by_id[case_id]["question"], "",
                      "预期业务目的：" + contract["expected_intent"], "",
                      "意图依据状态：" + statuses_zh[contract["overall_alignment_status"]] +
                      ("；已原样采用用户修改。" if case_id in revisions else "。"), "",
                      "| 业务目标 | 隐含 | 可支持还原的依据 | 审核状态 |", "|---|---|---|---|"])
        sources = {}
        for target in contract["required_business_targets"]:
            for basis in target["restoration_basis"]:
                if basis["source"] not in sources:
                    sources[basis["source"]] = "S" + str(len(sources) + 1)
            basis_text = "；".join(item["text"] + "〔" + sources[item["source"]] + ";" +
                                  ("可见" if item["observable"] else "业务约定待确认") + "〕"
                                  for item in target["restoration_basis"])
            cells = [target["description"], "是" if target["implicit"] else "否", basis_text,
                     statuses_zh[target["alignment_status"]]]
            lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in cells) + " |")
        lines.extend(["", "依据来源：", ""])
        for source, label in sources.items():
            file_part, separator, location = source.partition("::")
            source_path = ROOT / file_part
            if source_path.is_file():
                rendered = f"[{source_path.name}]({source_path})" + ("：" + location if separator else "")
            else:
                rendered = source
            lines.append("- " + label + "：" + rendered)
        notes = contract.get("review_notes", [])
        if isinstance(notes, str):
            notes = [notes]
        if notes:
            lines.extend(["", "审核注意：" + "；".join(notes)])
        behavior = goal["expected_behavior"]
        lines.extend(["", "原标准行为：" + (behavior if isinstance(behavior, str) else "；".join(behavior))])
        if goal.get("expected_result") is not None:
            lines.extend(["", "原标准结果（未更改）：", "", "```json",
                          json.dumps(goal["expected_result"], ensure_ascii=False, indent=2), "```"])
        if case["category"] == "C6":
            lines.extend(["", "本轮目标条件：`" + json.dumps(goal.get("state"), ensure_ascii=False) + "`",
                          "", "继承／替换／取消操作：`" + json.dumps(goal.get("operations"), ensure_ascii=False) + "`"])
        if goal.get("paired_question_ids"):
            lines.extend(["", "配对题：" + "、".join(goal["paired_question_ids"])])
    pending_lines = ["# 自然提问隐含目标与范围待审清单", "", "状态：待审核。原标准答案保持，以下项目需要确认还原依据或调整自然表达。", "",
                     "| 题号 | 目标编号 | 状态 | 原业务目标 |", "|---|---|---|---|"]
    for item in pending:
        pending_lines.append("| " + " | ".join(str(item[key]).replace("|", "\\|")
                             for key in ("id", "target_id", "status", "reason")) + " |")

    artifacts = {PUBLIC / "questions.json": encoded(questions), PUBLIC / "business_context.json": encoded(context),
                 PRIVATE / "ground_truth.json": encoded(truth), PRIVATE / "intent_contracts.json": encoded(intent_document),
                 PRIVATE / "review_rules.json": encoded(rules), PRIVATE / "coverage_matrix.json": encoded(matrix),
                 PRIVATE / "coverage_matrix.csv": buffer.getvalue().encode("utf-8-sig"),
                 PRIVATE / "自然提问与业务意图对照稿.md": ("\n".join(lines) + "\n").encode("utf-8"),
                 PRIVATE / "隐含目标与范围待审清单.md": ("\n".join(pending_lines) + "\n").encode("utf-8")}
    source_paths = [SNAPSHOT / record["snapshot_file"] for record in source_manifest["files"]]
    source_paths.extend([SNAPSHOT / "source_manifest.json", *overlay_paths, Path(__file__).resolve(),
                         BASELINE / "manifest.json", BASELINE / "tables.json", BASELINE / "data_dictionary.json",
                         ROOT / "data/reports/enterprise_v30_120/database_acceptance.json"])
    bundle = {**common, "created_date": "2026-10-13", "case_count": len(cases),
              "principles_confirmed_by_user": True, "question_review": "user_pending",
              "intent_basis_review": "user_pending", "independent_question_validation": "user_pending",
              "model_experiments": "user_pending", "pending_review_target_count": len(pending),
              "source_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in source_paths},
              "artifact_sha256": {str(path.relative_to(ROOT)): sha(data) for path, data in artifacts.items()},
              "preserved_original_goal_sha256": {item["id"]: goal_signature(item) for item in old_truth["cases"]},
              "database_acceptance": {"report": "data/reports/enterprise_v30_120/database_acceptance.json",
                                      "status": read(ROOT / "data/reports/enterprise_v30_120/database_acceptance.json")["status"]}}
    artifacts[PRIVATE / "bundle_manifest.json"] = encoded(bundle)
    for path, data in artifacts.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError("已有不同草案，请保留旧版并使用新版本目录：" + str(path))
    for path, data in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    return {"status": "DRAFT", "case_count": len(cases), "original_goal_count": len(goals),
            "user_revision_count": len(revisions), "intent_alignment_counts": matrix["intent_alignment_counts"],
            "pending_review_target_count": len(pending), "private_directory": str(PRIVATE)}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(), ensure_ascii=False, indent=2))
