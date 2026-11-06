"""只读取单题结果；汇总返回调用方，不另写报告文件。"""
from __future__ import annotations

from .common import read_json
from .comparison import COMPARISON_VERSION
from .runner import condition_completed, load_run


def summarize(run_dir):
    run, config, cases = load_run(run_dir)
    expected_cases = {case["id"]: case for case in cases}
    modes = {}
    for mode in config["conditions"]:
        folder = run / mode / "results"
        passed = failed = 0
        for path in sorted(folder.glob("*.json")):
            row = read_json(path)
            case_id = row.get("case_id")
            if (case_id not in expected_cases or path.stem != case_id
                    or row.get("condition") != mode
                    or row.get("question") != expected_cases[case_id]["question"]
                    or row.get("expected") != config["standards"][case_id]
                    or row.get("verdict") not in ("pass", "fail")):
                raise ValueError("单题记录与本轮配置不符：" + str(path))
            comparison_version = config.get("comparison_version")
            # Historical results remain readable; their stored verdicts are never
            # re-evaluated here. New results must carry the current audit fields.
            if comparison_version == COMPARISON_VERSION:
                if (row.get("comparison_version") != comparison_version
                        or not isinstance(row.get("comparison_reason"), str)
                        or "comparison_alignment" not in row
                        or (row["comparison_alignment"] is not None
                            and not isinstance(row["comparison_alignment"], dict))
                        or (row["verdict"] == "pass"
                            and not isinstance(row["comparison_alignment"], dict))):
                    raise ValueError("单题记录与本轮评分版本不符：" + str(path))
            elif "comparison_version" in row and row["comparison_version"] != comparison_version:
                raise ValueError("历史单题记录与本轮评分版本不符：" + str(path))
            passed += row["verdict"] == "pass"
            failed += row["verdict"] == "fail"
        complete = condition_completed(run / mode / "history.sqlite3")
        if complete and passed + failed != len(cases):
            raise ValueError("完成标记与单题记录数量不符")
        modes[mode] = {"total": len(cases), "recorded": passed + failed,
                       "pass": passed, "fail": failed,
                       "pass_rate": passed / len(cases) if complete else None,
                       "completed": complete}
    return {"run_dir": str(run), "comparison_version": config.get("comparison_version"),
            "conditions": modes}
