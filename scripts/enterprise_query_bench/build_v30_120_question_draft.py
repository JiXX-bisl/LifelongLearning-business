#!/usr/bin/env python3
"""从经公开记录推导的设计稿部署120家题库草案；不连接数据库或模型。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "enterprise_v30_120_v1_0_2_draft"
PUBLIC = ROOT / "data/questions/psu_memory" / NAME
PRIVATE = ROOT / "data/evaluation/psu_memory" / NAME
BASELINE_ID = "synthetic_enterprise_v30_120_v1.0.2"
BASELINE = ROOT / "data/sql" / BASELINE_ID
CATEGORIES = ("A1", "A2", "A4", "A5", "B2", "B3", "C1", "C2", "C4", "C6")
PUBLIC_FIELDS = ("id", "session", "turn", "question", "history", "category", "learning_stage", "difficulty")


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def as_json(value):
    # SQL boolean expressions return integers. Make the static oracle explicit.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, list):
        return [as_json(item) for item in value]
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    return value


def normalize_result(result):
    """Match connector normalization for typed DATETIME result columns."""
    columns = result["columns"]
    for row in result["rows"]:
        for index, value in enumerate(row):
            if (columns[index] in ("state_effective_at", "reported_at", "ingested_at")
                    and isinstance(value, str)
                    and re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", value)):
                row[index] = value.replace(" ", "T", 1)
    return result


def build():
    sources = [PRIVATE / "ab_design.json", PRIVATE / "c_design.json"]
    cases, facts = [], []
    for path in sources:
        design = read(path)
        cases.extend({key: case.get(key, [] if key == "history" else None) for key in PUBLIC_FIELDS}
                     for case in design["cases"])
        facts.extend(design["ground_truth"])
    cases.sort(key=lambda case: (case["learning_stage"], CATEGORIES.index(case["category"]), case["session"], case["turn"]))
    by_id = {case["id"]: case for case in cases}
    if len(by_id) != len(cases) or len({item["id"] for item in facts}) != len(facts):
        raise ValueError("设计稿题号重复")
    if set(by_id) != {item["id"] for item in facts}:
        raise ValueError("公开题目和私有依据题号不同")
    normalized, rubrics, expected_types, references, requirements = [], {}, {}, {}, {}
    pairs = defaultdict(list)
    for item in facts:
        if item.get("pair_id"):
            pairs[item["pair_id"]].append(item["id"])
    for raw in facts:
        item = as_json(dict(raw))
        case_id = item["id"]
        item["response_type"] = item.pop("expected_type", item.get("response_type"))
        if item["response_type"] not in ("answer", "clarify", "insufficient"):
            raise ValueError("设计稿响应类型无效：" + case_id)
        item.update(category=by_id[case_id]["category"], learning_stage=by_id[case_id]["learning_stage"],
                    difficulty=by_id[case_id]["difficulty"])
        if item.get("pair_id"):
            item["paired_question_ids"] = sorted(set(item.get("paired_question_ids", []))
                                                | (set(pairs[item["pair_id"]]) - {case_id}))
        expected_types[case_id] = item["response_type"]
        behavior = item["expected_behavior"]
        rubric = behavior if isinstance(behavior, str) else json.dumps(behavior, ensure_ascii=False)
        rubrics[case_id] = "回答范围：" + item["target_scope"] + "。" + rubric
        sql = item.get("reference_sql")
        requirements[case_id] = item.get("sql_requirement", "required" if item["response_type"] == "answer"
                                        else "not_applicable" if item["response_type"] == "clarify"
                                        else "optional")
        if sql:
            result = normalize_result(item["expected_result"])
            aliases = item.get("aliases", item.get("column_aliases", {}))
            if isinstance(aliases, list):
                aliases = {column: [] for column in aliases}
            if not isinstance(aliases, dict) or any(not isinstance(values, list) for values in aliases.values()):
                raise ValueError("列别名须为column->[aliases]映射：" + case_id)
            references[case_id] = {"sql": sql, "aliases": aliases,
                                   "mismatch_is_failure": True, "row_order": "unordered",
                                   "numeric_tolerances": item.get("numeric_tolerances", {})}
            if not isinstance(result, dict) or "columns" not in result or "rows" not in result:
                raise ValueError("设计稿标准结果结构无效：" + case_id)
        normalized.append(item)
    normalized.sort(key=lambda item: list(by_id).index(item["id"]))
    common = {"version": "psu_enterprise_v30_120_questions_v0.1.0", "data_version": BASELINE_ID,
              "schema_version": "3.0", "status": "DRAFT", "reviewed_by_user": False}
    questions = {**common, "cases": cases}
    truth = {**common, "basis": "公开七表、字典及题目指定口径；私有场景仅提供题目设计索引",
             "comparison": {"row_order": "unordered", "duplicate_policy": "multiset_preserve_multiplicity",
                            "sql_text_equality_required": False, "business_answer_review_required": True,
                            "amounts": "Decimal，按行amount_unit归一；参考结果保留十进制字符串",
                            "numeric_tolerances": "仅私有reference.numeric_tolerances明确列出的列使用绝对容差；其余精确比较",
                            "datetime_encoding": "MySQL DATETIME结果使用ISO 8601的T分隔符；公开查询条件仍按原数据库语义"},
             "cases": normalized}
    rules = {"version": "psu_enterprise_v30_120_review_v0.1.0", "data_version": BASELINE_ID,
             "general": ["标准结果、参考SQL、响应类型、场景和条件状态只供验证端读取。",
                         "接受等价只读SQL；结果按多重集比较，保留重复次数。",
                         "参考投影匹配仅证明该投影；最终业务回答和响应类型必须人工审核。",
                         "列出的金额和比例容差按numeric_tolerances执行；身份、计数、NULL和日期精确比较。",
                         "未知、明确零值、缺报与未覆盖分别解释，缺少数据不能填零。",
                         "L1/L2/L3为学习阶段，difficulty保留原L1—L4任务复杂度语义。",
                         "C6以公开user-only历史作为主对照输入，私有state仅核验继承、替换和取消。",
                         "经验使用、准入、分组、更新与淘汰沿用现有机制。"],
             "expected_types": expected_types, "sql_requirements": requirements,
             "rubrics": rubrics, "references": references}
    coverage = Counter((case["category"], case["learning_stage"]) for case in cases)
    matrix = {**common, "case_count": len(cases), "session_count": len({case["session"] for case in cases}),
              "response_type_counts": dict(Counter(expected_types.values())), "reference_count": len(references),
              "difficulty_counts": dict(Counter(case["difficulty"] for case in cases)),
              "category_stage_counts": {category: {stage: coverage[category, stage] for stage in ("L1", "L2", "L3")}
                                        for category in CATEGORIES},
              "pair_groups": dict(sorted(pairs.items())),
              "cases": [{key: item.get(key) for key in ("id", "category", "learning_stage", "difficulty", "enterprise_ids",
                         "response_type", "scenario_ids", "pair_id", "paired_question_ids", "change_axis", "target_scope")}
                        for item in normalized]}
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["id", "category", "learning_stage", "difficulty", "response_type", "enterprise_ids", "pair_id", "paired_question_ids", "change_axis", "target_scope"])
    for item in normalized:
        writer.writerow([item["id"], item["category"], item["learning_stage"], item["difficulty"], item["response_type"],
                         ";".join(item.get("enterprise_ids", [])), item.get("pair_id", ""),
                         ";".join(item.get("paired_question_ids", [])), json.dumps(item.get("change_axis"), ensure_ascii=False), item["target_scope"]])
    lines = ["# 120家 v1.0.2 首批题目审阅稿", "", "状态：DRAFT。标准依据仅供用户审核；模型输入取公开问题与公开对话历史。", "",
             "| 类别 | L1 形成 | L2 迁移 | L3 反例 |", "|---|---:|---:|---:|"]
    lines.extend(f"| {category} | {coverage[category, 'L1']} | {coverage[category, 'L2']} | {coverage[category, 'L3']} |" for category in CATEGORIES)
    for item in normalized:
        case = by_id[item["id"]]
        lines.extend(["", f"## {item['id']} · {item['response_type']} · 难度 {case['difficulty']}", "", case["question"], "",
                      "审核范围：" + item["target_scope"], "", "标准行为：" + rubrics[item["id"]]])
        if item.get("expected_result"):
            lines.extend(["", "标准结果：", "", "```json", json.dumps(item["expected_result"], ensure_ascii=False, indent=2), "```"])
        if case["category"] == "C6":
            lines.extend(["", "本轮条件状态：`" + json.dumps(item.get("state"), ensure_ascii=False) + "`",
                          "", "条件操作：`" + json.dumps(item.get("operations"), ensure_ascii=False) + "`"])
        if item.get("paired_question_ids"):
            lines.extend(["", "配对题：" + "、".join(item["paired_question_ids"])])
    artifacts = {PUBLIC / "questions.json": encoded(questions), PRIVATE / "ground_truth.json": encoded(truth),
                 PRIVATE / "review_rules.json": encoded(rules), PRIVATE / "coverage_matrix.json": encoded(matrix),
                 PRIVATE / "coverage_matrix.csv": buffer.getvalue().encode("utf-8-sig"),
                 PRIVATE / "题目审阅稿.md": ("\n".join(lines) + "\n").encode("utf-8")}
    acceptance_path = ROOT / "data/reports/enterprise_v30_120/database_acceptance.json"
    acceptance = read(acceptance_path)
    source_paths = [*sources, BASELINE / "manifest.json", BASELINE / "tables.json", BASELINE / "data_dictionary.json",
                    ROOT / "data/evaluation" / BASELINE_ID / "scenarios.json", acceptance_path]
    bundle = {**common, "created_date": "2026-10-13", "case_count": len(cases),
              "source_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in source_paths},
              "artifact_sha256": {str(path.relative_to(ROOT)): sha(data) for path, data in artifacts.items()},
              "database_acceptance": {"report": "data/reports/enterprise_v30_120/database_acceptance.json",
                                      "status": acceptance.get("status"), "baseline_id": acceptance.get("baseline_id")},
              "formal_business_acceptance": "user_pending", "question_review": "user_pending",
              "independent_question_validation": "user_pending", "model_experiments": "user_pending"}
    artifacts[PRIVATE / "bundle_manifest.json"] = encoded(bundle)
    # A published draft is reviewable input. Never overwrite a different draft.
    for path, data in artifacts.items():
        if path.exists() and path.read_bytes() != data:
            raise ValueError("既有草案内容不同，请保留旧版并使用新版本目录：" + str(path))
    for path, data in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
    return {"status": "DRAFT", "case_count": len(cases), "response_type_counts": matrix["response_type_counts"],
            "reference_count": len(references), "public_questions": str(PUBLIC / "questions.json"),
            "private_support": str(PRIVATE)}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(build(), ensure_ascii=False, indent=2))
