"""User-run, JSON-only diagnostic rescoring; never opens an experiment database."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from ..validation.comparison import COMPARISON_VERSION, assess_result, compare_results


def _read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def rescore(run_dir, report_path):
    run, destination = Path(run_dir).resolve(), Path(report_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("复评报告已存在，请指定新文件：" + str(destination))
    if destination.is_relative_to(run):
        raise ValueError("复评报告必须保存在原实验目录之外")
    config, config_sha = _read(run / "config.json")
    source_version = config.get("comparison_version")
    if not isinstance(source_version, str) or not source_version.strip():
        raise ValueError("缺少原运行的评分版本标识")
    # A historical version is provenance only. No retired evaluator is loaded.
    cases = config["cases"]
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, dict)
                   or not isinstance(case.get("id"), str) or not case["id"]
                   or "/" in case["id"] or "\\" in case["id"]
                   or case["id"] in (".", "..")
                   or not isinstance(case.get("question"), str)
                   for case in cases)):
        raise ValueError("无效题目清单")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("题目清单重复")
    standards = config.get("standards")
    if (not isinstance(standards, dict) or set(standards) != set(ids)
            or any(not compare_results(value, value) for value in standards.values())):
        raise ValueError("标准结果无效或题号不匹配")
    modes = config["conditions"]
    if (not isinstance(modes, list) or not modes
            or any(mode not in ("baseline", "fifo", "adaptive", "fixed") for mode in modes)
            or len(modes) != len(set(modes))):
        raise ValueError("无效实验条件")
    rows, summaries = [], {}
    submission_failures = {"invalid_submission", "submission_not_completed"}
    for mode in modes:
        folder = run / mode / "results"
        if {p.stem for p in folder.glob("*.json")} != set(ids):
            raise ValueError("结果尚未齐全或含额外题号：" + mode)
        transitions, reasons = Counter(), Counter()
        for case in cases:
            case_id = case["id"]
            path = folder / (case_id + ".json")
            old, source_sha = _read(path)
            expected = standards[case_id]
            if (not isinstance(old, dict)
                    or old.get("case_id") != case_id or old.get("condition") != mode
                    or old.get("question") != case["question"]
                    or old.get("expected") != expected
                    or old.get("comparison_version", source_version) != source_version
                    or old.get("verdict") not in ("pass", "fail")):
                raise ValueError("记录与冻结配置不一致：" + str(path))
            old_reason = old.get("comparison_reason")
            if old_reason is not None and not isinstance(old_reason, str):
                raise ValueError("原记录失败原因格式无效：" + str(path))
            if old_reason in submission_failures:
                assessed = {"verdict": "fail", "reason": old_reason, "alignment": None}
            else:
                # Use the original frozen question and dialogue, not today's
                # question bank or the model's answer, to determine requirements.
                assessed = assess_result(old.get("actual"), expected,
                                         question=case["question"],
                                         history=case.get("history", []))
            transitions[old["verdict"] + "->" + assessed["verdict"]] += 1
            reasons[assessed["reason"]] += 1
            rows.append({"case_id": case_id, "condition": mode,
                         "source": str(path), "source_sha256": source_sha,
                         "old_verdict": old["verdict"], "old_reason": old_reason,
                         "verdict": assessed["verdict"], "comparison_reason": assessed["reason"],
                         "comparison_alignment": assessed["alignment"]})
        summaries[mode] = {"total": len(cases), "transitions": dict(transitions),
                           "reasons": dict(reasons)}
    report = {
        "version": "saved_result_subset_rescore_v2", "comparison_version": COMPARISON_VERSION,
        "source_comparison_version": source_version, "source_run": str(run),
        "source_config_sha256": config_sha, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "仅复评已保存结果；未重放查询、提交完整性或在线经验更新，不构成新一轮在线实验。",
        "conditions": summaries, "results": rows,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"report": str(destination), "conditions": summaries}


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读旧JSON，按冻结题目及新版结果包含准则另写重评报告，无模型或数据库操作")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(rescore(args.run_dir, args.report), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
