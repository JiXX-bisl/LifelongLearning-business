#!/usr/bin/env python3
"""将用户已确认的自然题审核快照发布为v0.2.1，不运行数据库或模型。"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT_NAME = "enterprise_v30_120_v1_0_2_natural_v0_2_draft"
NAME = "enterprise_v30_120_v1_0_2_natural_v0_2_1_reviewed"
PRIVATE = ROOT / "data/evaluation/psu_memory" / NAME
PUBLIC = ROOT / "data/questions/psu_memory" / NAME
SNAPSHOT = PRIVATE / "source_snapshot"
BASELINE_ID = "synthetic_enterprise_v30_120_v1.0.2"
GOAL_CHANGES = {"A4-L2-01", "A4-L2-02"}
GOAL_FIELDS = ("expected_result", "expected_behavior", "response_type", "target_scope", "reference_sql",
               "state", "operations", "enterprise_ids", "sql_requirement", "numeric_tolerances")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def goal_signature(value):
    return sha(json.dumps({key: value.get(key) for key in GOAL_FIELDS}, ensure_ascii=False,
                          sort_keys=True).encode("utf-8"))


def render_source(source):
    file_part, separator, location = source.partition("::")
    path = ROOT / file_part
    return (f"[{path.name}]({path})" + ("：" + location if separator else "")) if path.is_file() else source


def parse_review():
    result = {}
    text = (SNAPSHOT / "user_edited_review.md").read_text(encoding="utf-8")
    for section in re.split(r"(?m)^## ", text)[1:]:
        if " · " not in section.split("\n", 1)[0]:
            continue
        case_id = section.split(" · ", 1)[0]
        match = re.search(r"\*\*自然提问：(.*?)\*\*", section, re.S)
        if match is None:
            raise ValueError("审阅稿缺少自然提问：" + case_id)
        status = re.search(r"^意图依据状态：(.*)$", section, re.M)
        raw_status = status.group(1).strip() if status else ""
        reason = raw_status.split("。", 1)[1].strip() if "。" in raw_status else ""
        result[case_id] = {"question": match.group(1).strip(), "raw_status": raw_status,
                           "user_reason": reason}
    return result


def latest_public_month(tables, enterprise_id):
    # Offline oracle preparation reads only public observations. Independent
    # database reference execution remains a separate user-operated check.
    ranked = {}
    for row in tables["enterprise_operation_monthly"]:
        if row["enterprise_id"] != enterprise_id or row["ingested_at"] > "2026-09-15 23:59:59":
            continue
        key = (row["source_system"], row["source_record_ref"])
        rank = (row["report_revision"], row["ingested_at"], row["monthly_record_id"])
        if key not in ranked or rank > ranked[key][0]:
            ranked[key] = (rank, row)
    formal = [row for _, row in ranked.values() if row["report_status"] == "正式"]
    latest_month = max(row["stat_month"] for row in formal)
    candidates = [row for row in formal if row["stat_month"] == latest_month]
    if len(candidates) != 1 or candidates[0]["operating_revenue"] is None:
        raise ValueError("最新月报需独立业务处理，不能编造确定值：" + enterprise_id)
    row = candidates[0]
    multiplier = Decimal("10000") if row["amount_unit"] == "万元" else Decimal("1")
    return row, format(Decimal(row["operating_revenue"]) * multiplier, ".2f")


def build():
    request = read(SNAPSHOT / "user_review_request.json")
    if request.get("approve_remaining_cases") is not True:
        raise ValueError("缺少用户对其他题目的明确审核确认")
    parent_questions = read(SNAPSHOT / "parent_natural_questions.json")
    parent_truth = read(SNAPSHOT / "parent_natural_ground_truth.json")
    parent_rules = read(SNAPSHOT / "parent_natural_review_rules.json")
    parent_contracts = read(SNAPSHOT / "parent_natural_intent_contracts.json")
    source_truth = read(SNAPSHOT / "ground_truth.json")
    source_rules = read(SNAPSHOT / "review_rules.json")
    old_facts = {item["id"]: item for item in source_truth["cases"]}
    parent_map = {item["id"]: item for item in parent_questions["cases"]}
    goals = {item["id"]: copy.deepcopy(item) for item in parent_truth["cases"]}
    contracts = {item["id"]: copy.deepcopy(item) for item in parent_contracts["cases"]}
    review = parse_review()
    if set(review) != set(parent_map) or not set(request["requested_case_ids"]).issubset(review):
        raise ValueError("用户审阅稿与题目清单不完整对应")
    findings = read(SNAPSHOT / "review_findings_ab.json") + read(SNAPSHOT / "review_findings_c6.json")["cases"]
    if (SNAPSHOT / "review_findings_additional.json").is_file():
        findings += read(SNAPSHOT / "review_findings_additional.json")
    finding_map = {item["id"]: item for item in findings}
    changed_ids = [case_id for case_id in parent_map if review[case_id]["question"] != parent_map[case_id]["question"]]
    additional_changed_ids = sorted(set(changed_ids) - set(request["requested_case_ids"]))
    if any(case_id not in finding_map for case_id in additional_changed_ids):
        raise ValueError("其余已认可题目的文字变化须补充核查记录")

    reason_latest = review["A4-L2-01"]["user_reason"]
    if not all("数据库中最新" in review[case_id]["user_reason"] for case_id in GOAL_CHANGES):
        raise ValueError("最新月份业务口径修订缺少用户明确理由")
    tables = read(ROOT / "data/sql" / BASELINE_ID / "tables.json")
    rows = {}
    for case_id in sorted(GOAL_CHANGES):
        goal = goals[case_id]
        row, amount = latest_public_month(tables, goal["enterprise_ids"][0])
        rows[case_id] = {"month": row["stat_month"], "amount_yuan": amount, "record_id": row["monthly_record_id"]}
        goal["target_scope"] = "当前可用数据库记录中的最新经营月份及营业收入，资料截止2026-09-15"
        goal["reference_sql"] = goal["reference_sql"].replace(
            "ingested_at <= '2026-08-31 23:59:59'", "ingested_at <= '2026-09-15 23:59:59'")
        if "ingested_at <= '2026-08-31 23:59:59'" in goal["reference_sql"]:
            raise ValueError("历史入库截止条件未移除：" + case_id)
        goal["expected_result"]["rows"] = [[row["stat_month"], amount]]
        goal["expected_behavior"] = [
            "未明确历史知识时点的最近营收，使用当前可用数据库记录中的最新经营月份，并说明月份与单位。",
            f"最新可用经营月份为{row['stat_month'][:7]}，营业收入{amount}元；报送或入库月份不替代业务经营月份。",
        ]
        goal["answer_keypoints"] = copy.deepcopy(goal["expected_behavior"])
        goal["forbidden_behavior"] = ["擅自恢复用户已删除的8月31日入库截止条件", "将报送或入库月份当作经营所属月",
                                      "不检查当前可用记录而固定复用历史月份答案"]
        axis = ["企业与数值迁移", "最近指当前可用数据库最新经营月份", "与新报送及新入库期间区分"]
        goal["change_axes"] = axis
        goal["change_axis"] = axis
        goal["paired_result_difference"] = "L2查当前最新经营月份；L1指定经营月份；L3盘点报送或入库期间。旧历史知识口径保留审计。"
        goal["authorized_goal_change_source"] = str((PRIVATE / "approval_report.json").relative_to(ROOT)) + "#" + case_id

    source_review = str((SNAPSHOT / "user_edited_review.md").relative_to(ROOT))
    source_request = str((SNAPSHOT / "user_review_request.json").relative_to(ROOT))
    history, cases, per_case_reviews = {}, [], []
    for old_case in parent_questions["cases"]:
        case_id = old_case["id"]
        case = copy.deepcopy(old_case)
        case["question"] = review[case_id]["question"]
        previous = history.setdefault(case["session"], [])
        case["history"] = [{"role": "user", "content": text} for text in previous]
        previous.append(case["question"])
        cases.append(case)
        contract = contracts[case_id]
        contract.update(actual_question=case["question"], basis_reviewed_by_user=True,
                        review_status="APPROVED", review_date="2026-10-13", user_review_reason=review[case_id]["user_reason"],
                        public_context_reference=str((PUBLIC / "business_context.json").relative_to(ROOT)),
                        reviewer="用户（本次对话明确确认）", review_source=source_request)
        resolution = {"reason": "用户明确确认其余问题无问题；本题自然表达与保留的业务目标通过人工审核。",
                      "source": source_request + "#" + case_id}
        if case_id in request["requested_case_ids"]:
            resolution["reason"] = "用户提交本题修改并授权核查后标为通过；新自然表达和业务意图已核对。"
            resolution["source"] = source_review + "#" + case_id
        if review[case_id]["user_reason"]:
            resolution["reason"] += "用户理由：" + review[case_id]["user_reason"]
        for target in contract["required_business_targets"]:
            bases = target["restoration_basis"]
            # Quoted input must reflect the actual new wording. Public prior
            # questions remain public evidence; human reasons stay private.
            for basis in bases:
                if basis["kind"] == "question_wording":
                    basis["text"] = case["question"]
                    basis["source"] = "public_question:" + case_id
                elif basis["kind"] == "conversation_context":
                    match = re.search(r"(?:public_question:|case:)([A-Z0-9-]+)", basis["source"])
                    if match and match.group(1) in review:
                        basis["text"] = review[match.group(1)]["question"]
            if case_id in finding_map and "replacement_restoration_basis" in finding_map[case_id]:
                replacement = finding_map[case_id]["replacement_restoration_basis"].get(target["target_id"])
                if replacement:
                    bases = copy.deepcopy(replacement)
                    for basis in bases:
                        if basis["kind"] == "reviewed_business_convention":
                            basis["kind"] = "business_convention"
                            basis["source"] = source_review + "#" + case_id
                    target["restoration_basis"] = bases
                    target["implicit"] = finding_map[case_id].get("implicit_targets", {}).get(target["target_id"], True)
            elif case_id in changed_ids:
                target["implicit"] = True
            target.update(review_status="APPROVED", review_resolution=copy.deepcopy(resolution))
            # Retain the old evidence diagnostic while recording the human
            # resolution explicitly, rather than manufacturing public cues.
            target["pre_review_alignment_status"] = target["alignment_status"]
        if case_id in GOAL_CHANGES:
            contract["expected_intent"] = "理解最近为当前可用数据库中的最新经营月份，报告该月营收及其业务月份。"
            for target in contract["required_business_targets"]:
                target["description"] = ("确定当前可用资料中该企业最新经营月份。" if "stat_month" in target["standard_result_columns"]
                                         else "报告该最新经营月份的营业收入及单位，不限定用户未提出的历史入库时点。")
                target["implicit"] = True
                target["restoration_basis"] = [
                    {"kind": "question_wording", "text": case["question"], "source": "public_question:" + case_id, "observable": True},
                    {"kind": "public_dictionary", "text": "月报经营月份与报送及入库时间是独立维度；金额按记录单位解释。",
                     "source": "data/sql/" + BASELINE_ID + "/business_dictionary.md::enterprise_operation_monthly", "observable": True},
                    {"kind": "public_records", "text": "当前可用月报包含该企业8月经营记录，按经营月份可识别最新可用期。",
                     "source": "data/sql/" + BASELINE_ID + "/tables.json::enterprise_operation_monthly.monthly_record_id=" + rows[case_id]["record_id"], "observable": True},
                    {"kind": "business_convention", "text": reason_latest, "source": source_review + "#" + case_id,
                     "observable": False, "reviewed_by_user": True},
                ]
            contract["review_notes"] = ["用户明确取消历史知识截止，活动标准改为数据库当前最新可用经营月份；旧问题、旧7月标准及SQL均保留在审核授权审计。"]
        else:
            contract["review_notes"] = [*contract.get("review_notes", []), "原业务目标保留；本次用户已确认通过，上述审核前差异诊断保留用于追溯。"]
        if case_id in changed_ids and case_id not in GOAL_CHANGES:
            convention = {"kind": "business_convention", "text": resolution["reason"],
                          "source": resolution["source"], "observable": False, "reviewed_by_user": True}
            for target in contract["required_business_targets"]:
                target["restoration_basis"].append(copy.deepcopy(convention))
        goal = goals[case_id]
        goal.update(expected_intent=contract["expected_intent"], required_business_targets=copy.deepcopy(contract["required_business_targets"]),
                    intent_contract_source=str((PRIVATE / "intent_contracts.json").relative_to(ROOT)),
                    question_review_status="APPROVED", intent_basis_review_status="APPROVED",
                    review_source=source_request, original_goal_sha256=goal_signature(old_facts[case_id]))
        assessment = finding_map.get(case_id, {}).get("assessment")
        per_case_reviews.append({"id": case_id, "status": "APPROVED", "source": resolution["source"],
                                 "listed_by_user": case_id in request["requested_case_ids"],
                                 "question_changed": case_id in changed_ids,
                                 "old_question": old_case["question"], "new_question": case["question"],
                                 "user_reason": review[case_id]["user_reason"], "assistant_assessment": assessment,
                                 "business_goal_changed": case_id in GOAL_CHANGES})

    common = {"version": "psu_enterprise_v30_120_natural_v0.2.1", "data_version": BASELINE_ID,
              "schema_version": "3.0", "status": "REVIEWED", "reviewed_by_user": True,
              "review_date": "2026-10-13"}
    question_doc = {**copy.deepcopy(parent_questions), **common, "cases": cases}
    goal_doc = {**copy.deepcopy(parent_truth), **common, "cases": [goals[case["id"]] for case in cases],
                "basis": "自然问法与用户审核确认的业务目的；70题保留原目标，2题最近营收按用户明确理由修订，均有追溯。"}
    intent_doc = {**copy.deepcopy(parent_contracts), **common,
                  "pending_reviews": [], "pending_review_target_count": 0,
                  "case_count": len(cases), "cases": [contracts[case["id"]] for case in cases]}
    rules = copy.deepcopy(parent_rules)
    rules.update(version="psu_enterprise_v30_120_natural_review_v0.2.1", reviewed_by_user=True,
                 intent_contracts=str((PRIVATE / "intent_contracts.json").relative_to(ROOT)))
    for case_id in GOAL_CHANGES:
        rules["references"][case_id]["sql"] = goals[case_id]["reference_sql"]
        rules["rubrics"][case_id] = "回答范围：" + goals[case_id]["target_scope"] + "。" + "；".join(goals[case_id]["expected_behavior"])
    rules["intent_rubrics"] = {case["id"]: "已审核预期目的：" + contracts[case["id"]]["expected_intent"] + "。必要业务目标：" +
                              "；".join(target["description"] for target in contracts[case["id"]]["required_business_targets"])
                              for case in cases}
    rules["general"] = ["题目及隐含目标已获用户人工审核；原诊断记录保留，不将私有审核理由输入模型或PSU。",
                        "只有A4-L2两题按用户明确理由更新最近营收口径；其他70题原业务目标不变。",
                        "人工题库审核通过不代表参考SQL独立核验或模型实验已完成。", *rules.get("general", [])]

    authorizations, reference_authorizations = {}, {}
    for case_id in sorted(GOAL_CHANGES):
        old, new = old_facts[case_id], goals[case_id]
        field_changes = {key: {"old": value, "new": new.get(key)} for key, value in old.items() if new.get(key) != value}
        authorizations[case_id] = {"old_goal_sha256": canonical(old), "new_goal_sha256": canonical(new),
                                   "field_changes": field_changes, "reason": review[case_id]["user_reason"],
                                   "source": source_review + "#" + case_id}
        old_ref, new_ref = source_rules["references"][case_id], rules["references"][case_id]
        reference_authorizations[case_id] = {"old_reference_sha256": canonical(old_ref), "new_reference_sha256": canonical(new_ref),
                                              "field_changes": {"sql": {"old": old_ref["sql"], "new": new_ref["sql"]}},
                                              "reason": review[case_id]["user_reason"], "source": source_review + "#" + case_id}
    for case_id in set(goals) - GOAL_CHANGES:
        if any(goals[case_id].get(key) != value for key, value in old_facts[case_id].items()):
            raise ValueError("未经用户授权的原标准目标变化：" + case_id)
    approval = {"version": "natural_question_human_approval_v0.2.1", "status": "HUMAN_REVIEW_APPROVED",
                "data_version": BASELINE_ID, "reviewer": "用户（本次对话明确确认）", "review_date": "2026-10-13",
                "source": source_request, "reviewed_case_count": len(cases), "case_reviews": per_case_reviews,
                "additional_visible_question_changes": additional_changed_ids,
                "authorized_goal_changes": authorizations, "authorized_reference_changes": reference_authorizations,
                "latest_revenue_oracle_from_public_records": rows,
                "independent_reference_validation": "user_pending", "model_experiments": "user_pending",
                "formal_database_business_acceptance": "user_pending"}

    matrix = read(SNAPSHOT / "parent_natural_coverage_matrix.json")
    matrix.update(**common, pending_review_target_count=0, reviewed_case_count=len(cases),
                  original_goals_unchanged_count=len(cases)-len(GOAL_CHANGES),
                  authorized_goal_change_ids=sorted(GOAL_CHANGES), review_status_counts={"APPROVED": len(cases)})
    for item in matrix["cases"]:
        case_id = item["id"]
        item.update(actual_question=review[case_id]["question"], expected_intent=contracts[case_id]["expected_intent"],
                    review_status="APPROVED", business_goal_changed=case_id in GOAL_CHANGES)
        if case_id in GOAL_CHANGES:
            item["target_scope"] = goals[case_id]["target_scope"]
            item["change_axis"] = goals[case_id]["change_axis"]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["id", "category", "learning_stage", "difficulty", "response_type", "natural_question", "expected_intent", "human_review", "business_goal_changed"])
    for case in cases:
        writer.writerow([case["id"], case["category"], case["learning_stage"], case["difficulty"], goals[case["id"]]["response_type"],
                         case["question"], contracts[case["id"]]["expected_intent"], "APPROVED", case["id"] in GOAL_CHANGES])
    context = read(SNAPSHOT / "parent_business_context.json")
    lines = ["# 120家 v1.0.2 自然提问与业务意图对照稿（已审核v0.2.1）", "",
             "状态：REVIEWED。72题及隐含目标已通过用户人工审核；参考SQL独立核验、正式数据库业务确认和模型实验另行记录。", "",
             "本次用户复核12题，其他60题明确认可。保留自然表达，70题原业务目标不变；A4-L2两题按用户明确理由改为当前数据库最新可用营收。", "",
             "## 共有业务背景", "", context["messages"][0]["content"], "",
             "## 本轮学习的提问习惯", "",
             "- 最近营收未指定历史知识时点时，主动识别当前可用数据库中的最新经营月份。",
             "- 是否亏损等简短问题应给出结论及有依据的月度事实，省略指标或金额要求不自动缩减业务目标。",
             "- 逐月净利润的自然续问可恢复完整期间，包括盈利月份；不要求用户逐项明说取消筛选。",
             "- 完整过程审计应主动核对时间线；年度缴税核查应主动识别缺报或未知，不依赖用户先提示缺漏。",
             "- 以上为审核端记录的业务目的依据，不作为私有必答字段清单输入模型或预填PSU。"]
    for case in cases:
        case_id = case["id"]
        contract, goal = contracts[case_id], goals[case_id]
        lines.extend(["", f"## {case_id} · {goal['response_type']} · 难度 {case['difficulty']}", "",
                      "**自然提问：" + case["question"] + "**", "",
                      "原标准问题（追溯）：" + contract["standard_question"], "",
                      "预期业务目的：" + contract["expected_intent"], "",
                      "**意图依据状态：通过审核。**", "", "人工审核：用户明确确认；本轮问句与依据同步完成。"])
        if review[case_id]["user_reason"]:
            lines.extend(["", "用户修改理由：" + review[case_id]["user_reason"]])
        sources = {}
        lines.extend(["", "| 业务目标 | 隐含 | 还原依据 | 审核 |", "|---|---|---|---|"])
        for target in contract["required_business_targets"]:
            for basis in target["restoration_basis"]:
                sources.setdefault(basis["source"], "S" + str(len(sources)+1))
            text = "；".join(basis["text"] + "〔" + sources[basis["source"]] + ("；可见〕" if basis["observable"] else "；已审核私有依据〕")
                            for basis in target["restoration_basis"])
            cells = [target["description"], "是" if target["implicit"] else "否", text, "通过审核"]
            lines.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in cells) + " |")
        lines.extend(["", "依据来源：", ""])
        lines.extend("- " + label + "：" + render_source(source) for source, label in sources.items())
        behavior = goal["expected_behavior"]
        lines.extend(["", "已审核标准行为：" + (behavior if isinstance(behavior,str) else "；".join(behavior)), "",
                      "已审核标准结果：", "", "```json", json.dumps(goal["expected_result"], ensure_ascii=False, indent=2), "```"])
        if case_id in GOAL_CHANGES:
            lines.extend(["", "本题经用户明确修改最近口径；历史标准结果保留追溯：`" +
                          json.dumps(old_facts[case_id]["expected_result"], ensure_ascii=False) + "`"])
        if case["category"] == "C6":
            lines.extend(["", "目标条件：`" + json.dumps(goal["state"], ensure_ascii=False) + "`",
                          "", "条件操作：`" + json.dumps(goal["operations"], ensure_ascii=False) + "`"])
    artifacts = {PUBLIC / "questions.json": encode(question_doc), PUBLIC / "business_context.json": encode(context),
                 PRIVATE / "ground_truth.json": encode(goal_doc), PRIVATE / "intent_contracts.json": encode(intent_doc),
                 PRIVATE / "review_rules.json": encode(rules), PRIVATE / "approval_report.json": encode(approval),
                 PRIVATE / "coverage_matrix.json": encode(matrix), PRIVATE / "coverage_matrix.csv": buffer.getvalue().encode("utf-8-sig"),
                 PRIVATE / "自然提问与业务意图对照稿.md": ("\n".join(lines)+"\n").encode("utf-8"),
                 PRIVATE / "隐含目标与范围待审清单.md": "# 隐含目标与范围审核记录\n\n72题均已通过用户人工审核，当前无待审项。原差异诊断、用户修改及理由保留在意图合同、审核报告及来源快照。\n".encode("utf-8")}
    source_paths = [*sorted(SNAPSHOT.rglob("*")), Path(__file__).resolve(), ROOT / "data/sql" / BASELINE_ID / "tables.json",
                    ROOT / "data/sql" / BASELINE_ID / "manifest.json", ROOT / "data/reports/enterprise_v30_120/database_acceptance.json"]
    manifest = {**common, "case_count": len(cases), "question_review": "APPROVED", "intent_basis_review": "APPROVED",
                "pending_review_target_count": 0, "authorized_goal_change_ids": sorted(GOAL_CHANGES),
                "original_goals_unchanged_count": len(cases)-len(GOAL_CHANGES),
                "independent_question_validation": "user_pending", "formal_database_business_acceptance": "user_pending",
                "model_experiments": "user_pending",
                "artifact_sha256": {str(path.relative_to(ROOT)): sha(data) for path,data in artifacts.items()},
                "source_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in source_paths if path.is_file()},
                "database_acceptance": {"report": "data/reports/enterprise_v30_120/database_acceptance.json", "status": "PASS"}}
    artifacts[PRIVATE / "bundle_manifest.json"] = encode(manifest)
    for path,data in artifacts.items():
        if path.exists() and path.read_bytes()!=data:
            raise ValueError("既有审核发布内容不同，请保留并建立新版本：" + str(path))
    for path,data in artifacts.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(data)
    return {"status": "REVIEWED", "reviewed_cases": len(cases), "user_listed_cases": len(request["requested_case_ids"]),
            "visible_question_changes": len(changed_ids), "unchanged_business_goals": len(cases)-len(GOAL_CHANGES),
            "authorized_goal_changes": rows, "review_document": str(PRIVATE / "自然提问与业务意图对照稿.md")}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(),ensure_ascii=False,indent=2))
