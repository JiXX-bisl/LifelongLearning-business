"""直接SQL验证的配置与公开问题读取；导入时无外部操作。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from src.project_paths import resolve_project_path
from ..integration.common import read_json, write_json

MODES = ("baseline", "fifo", "adaptive", "fixed")
POLICY = "sql_result_direct_v1"
DEFAULT_CONFIG = "configs/validation/enterprise_v30_120_v1_0_2_natural_direct_v1.json"


def load_public_context(path, *, data_version=None):
    if path is None:
        return []
    value = read_json(path)
    if data_version is not None and value.get("data_version") != data_version:
        raise ValueError("公开背景与数据版本不同")
    messages = value.get("messages")
    if not isinstance(messages, list) or any(
        not isinstance(m, dict) or set(m) != {"role", "content"}
        or m["role"] != "user" or not isinstance(m["content"], str) or not m["content"].strip()
        for m in messages
    ):
        raise ValueError("公开背景只接受用户消息")
    return deepcopy(messages)


def load_cases(path):
    document = read_json(path)
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("题库不能为空")
    ids, sessions, previous = set(), {}, None
    normalized = []
    for raw in cases:
        case = deepcopy(raw)
        case_id, session = case.get("id"), case.get("session")
        if (not isinstance(case_id, str) or not case_id.strip() or case_id in ids
                or case_id in (".", "..") or "/" in case_id or "\\" in case_id):
            raise ValueError("题号必须唯一且不能包含路径")
        if not isinstance(session, str) or not session.strip():
            raise ValueError("会话标识不能为空")
        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        questions = sessions.setdefault(session, [])
        if (type(case.get("turn")) is not int or case["turn"] != len(questions) + 1
                or (questions and previous != session)):
            raise ValueError("完整会话须连续排列并从第一轮开始")
        history = [{"role": "user", "content": q} for q in questions]
        if "history" in case and case["history"] != history:
            raise ValueError("公开history须等于本会话先前的用户问题")
        case["history"] = history
        questions.append(question)
        ids.add(case_id)
        previous = session
        normalized.append(case)
    return document, normalized


def load_config(path=DEFAULT_CONFIG):
    value = read_json(resolve_project_path(path))
    if value.get("policy") != POLICY:
        raise ValueError("仅支持sql_result_direct_v1，请使用新的直接比较配置")
    config = deepcopy(value)
    conditions = config.get("conditions", ["baseline", "fifo", "adaptive"])
    if (not isinstance(conditions, list) or not conditions
            or len(set(conditions)) != len(conditions)
            or any(mode not in MODES for mode in conditions)):
        raise ValueError("conditions仅支持不重复的baseline/fifo/adaptive；fixed保留旧版只检索语义")
    config["conditions"] = conditions
    fifo_capacity = config.setdefault("fifo_capacity", 18)
    if type(fifo_capacity) is not int or fifo_capacity < 1:
        raise ValueError("fifo_capacity必须是正整数")
    for key in ("namespace", "data_version"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(key + "不能为空")
    for key in ("questions", "ground_truth", "business_config", "retrieval_config", "pool_config"):
        if not isinstance(config.get(key), str):
            raise ValueError("缺少输入路径：" + key)
        config[key] = str(resolve_project_path(config[key]))
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    for key in ("public_context", "acceptance_report", "source_store"):
        if config.get(key) is not None:
            config[key] = str(resolve_project_path(config[key]))
            if not Path(config[key]).is_file():
                raise FileNotFoundError(config[key])
    initial = config.setdefault("initial_state", "empty")
    if initial not in ("empty", "snapshot"):
        raise ValueError("initial_state仅支持empty/snapshot")
    if (initial == "empty" and config.get("source_store") is not None
            or initial == "snapshot" and not config.get("source_store")):
        raise ValueError("empty不得带source_store；snapshot必须提供source_store")
    excluded = config.setdefault("exclude_case_ids", [])
    if (not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded)
            or len(set(excluded)) != len(excluded)):
        raise ValueError("exclude_case_ids必须是不重复题号列表")
    return config
