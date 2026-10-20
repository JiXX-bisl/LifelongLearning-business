"""读取固定题目分区；只核对文件，不请求模型或数据库。"""
from __future__ import annotations

from pathlib import Path

from src.project_paths import resolve_project_path
from ..integration.common import file_hash, read_json


def select_cases(config, cases):
    split_path = config.get("split_file")
    if not split_path:
        if config.get("case_partition"):
            raise ValueError("case_partition必须配合split_file")
        return [case for case in cases if case["id"] not in config["exclude_case_ids"]]
    split = read_json(split_path)
    if (split.get("version") != "psu_formation_test_split_v1"
            or split.get("data_version") != config["data_version"]):
        raise ValueError("题目分区版本不符")
    partition = config.get("case_partition")
    if partition not in ("formation", "test"):
        raise ValueError("case_partition仅支持formation/test")
    for name in ("questions", "ground_truth", "reference_validation"):
        source = split.get("sources", {}).get(name, {})
        path = resolve_project_path(source.get("path", ""))
        if not path.is_file() or file_hash(path) != source.get("sha256"):
            raise ValueError("分区来源文件已变化：" + name)
        if name in config and path != Path(config[name]).resolve():
            raise ValueError("配置输入与分区来源不一致：" + name)
    report = read_json(resolve_project_path(split["sources"]["reference_validation"]["path"]))
    if (report.get("verdict") != "pass" or report.get("approved") is not True
            or report.get("fail_count") != 0
            or report.get("data_version") != config["data_version"]
            or report.get("ground_truth_sha256_after") != split["sources"]["ground_truth"]["sha256"]):
        raise ValueError("分区所用标准尚未通过用户数据库核验")
    owners = {}
    for group in ("formation", "test", "excluded"):
        ids = split.get(group + "_case_ids")
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise ValueError("题目分区须为题号列表")
        for case_id in ids:
            if case_id in owners:
                raise ValueError("题目分区重复：" + case_id)
            owners[case_id] = group
    if set(owners) != {case["id"] for case in cases}:
        raise ValueError("题目分区必须完整覆盖题库")
    if set(config["exclude_case_ids"]) != set(split["excluded_case_ids"]):
        raise ValueError("排除清单与固定分区不符")
    if (set(report.get("excluded_ids", [])) != set(split["excluded_case_ids"])
            or report.get("pass_count") != len(owners) - len(split["excluded_case_ids"])):
        raise ValueError("参考SQL核验数量与固定分区不符")
    sessions = {}
    for case in cases:
        owner = owners[case["id"]]
        if sessions.setdefault(case["session"], owner) != owner:
            raise ValueError("不得跨分区拆开多轮会话")
    selected = [case for case in cases if owners[case["id"]] == partition]
    if not selected or [case["id"] for case in selected] != split[partition + "_case_ids"]:
        raise ValueError("分区不能为空，且必须保持原题库顺序")
    return selected
