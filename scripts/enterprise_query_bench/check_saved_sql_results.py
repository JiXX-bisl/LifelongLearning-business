#!/usr/bin/env python3
"""由用户运行：只读重比已保存 SQL 结果，不重建正式评分或在线经验路径。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"JSON 字段重复：{path}: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None


def check_saved(run_dir):
    """Return diagnostic counts only; never read stores or change saved records."""
    run = Path(run_dir).resolve()
    config = read_json(run / "config.json")
    require(isinstance(config, dict) and config.get("policy") == "sql_result_direct_v1",
            "仅支持 sql_result_direct_v1 的已保存结果")
    old_version = config.get("comparison_version")
    require(isinstance(old_version, str) and old_version, "缺少原比较器版本")
    conditions, cases = config.get("conditions"), config.get("cases")
    require(isinstance(conditions, list) and conditions
            and all(identifier(mode) for mode in conditions)
            and len(set(conditions)) == len(conditions), "组别缺失、重复或无效")
    require(isinstance(cases, list) and cases
            and all(isinstance(case, dict) and identifier(case.get("id"))
                    and isinstance(case.get("question"), str) and case["question"].strip()
                    for case in cases), "题目列表无效")
    ids = {case["id"] for case in cases}
    require(len(ids) == len(cases), "题号重复")
    standards, contexts = config.get("standards"), config.get("comparison_contexts")
    require(isinstance(standards, dict) and set(standards) == ids, "标准结果题号不匹配")
    require(isinstance(contexts, dict) and set(contexts) == ids
            and all(isinstance(contexts[key], dict) for key in ids),
            "缺少逐题辅助列核验依据")
    records = []
    for mode in conditions:
        folder = run / mode / "results"
        paths = list(folder.iterdir())
        require({path.name for path in paths} == {key + ".json" for key in ids}
                and all(path.is_file() for path in paths), f"{mode} 结果文件缺失或多余")
        for case in cases:
            key = case["id"]
            record = read_json(folder / (key + ".json"))
            require(isinstance(record, dict)
                    and record.get("case_id") == key and record.get("condition") == mode
                    and record.get("question") == case["question"]
                    and record.get("expected") == standards[key],
                    f"{mode}/{key} 题号、组别、问题或标准结果不一致")
            require(record.get("verdict") in ("pass", "fail"), f"{mode}/{key} 原评分无效")
            records.append((mode, key, record))
    # Keep this command independent of runner, database, model and store modules.
    sys.dont_write_bytecode = True
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.psu_memory.validation.comparison import (
        COMPARISON_VERSION, compare_submission_results,
    )
    from src.psu_memory.validation.comparison_versions import COVERAGE_COMPARISON_VERSION
    if old_version == COVERAGE_COMPARISON_VERSION:
        from src.psu_memory.validation.coverage_comparison import compare_coverage_results
        current_version = COVERAGE_COMPARISON_VERSION
    else:
        current_version = COMPARISON_VERSION

    output = {
        "diagnostic_only": True,
        "scope": "saved_sql_results_only",
        "limitation": "原记录未保存任务状态、SQL 成功及完整标记；本次仅比较保存结果，"
                      "不是正式重评，不重建经验更新后的执行路径。",
        "run_dir": str(run),
        "policy": config["policy"],
        "comparison_version": {"original": old_version, "current": current_version},
        "conditions": {mode: {
            "total": len(cases), "original": {"pass": 0, "fail": 0},
            "fixed_output_diagnostic": {"pass": 0, "fail": 0}, "changed_cases": [],
        } for mode in conditions},
    }
    for mode, key, record in records:
        available = (identifier(record.get("query_id")) and record["query_id"] != "null"
                     and isinstance(record.get("sql"), str) and bool(record["sql"].strip())
                     and isinstance(record.get("actual"), dict))
        if old_version == COVERAGE_COMPARISON_VERSION:
            require(record.get("comparison_version") == old_version,
                    f"{mode}/{key} 评分版本与运行配置不一致")
            matched = available and compare_coverage_results(
                record["actual"], standards[key], context=contexts[key], sql=record["sql"])
        else:
            matched = available and compare_submission_results(
                record["actual"], standards[key], context=contexts[key])
        diagnostic = "pass" if matched else "fail"
        group = output["conditions"][mode]
        group["original"][record["verdict"]] += 1
        group["fixed_output_diagnostic"][diagnostic] += 1
        if diagnostic != record["verdict"]:
            group["changed_cases"].append({
                "case_id": key, "original": record["verdict"], "diagnostic": diagnostic,
            })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="已完成运行目录，只读取 JSON")
    args = parser.parse_args(argv)
    try:
        output = check_saved(args.run_dir)
    except (OSError, ValueError, TypeError) as error:
        print(f"结果核验失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
