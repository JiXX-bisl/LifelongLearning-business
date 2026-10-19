"""Versioned public prompt resources; no question bank or oracle inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.project_paths import resolve_project_path

LEGACY_PROMPT_PROFILE = "legacy_v1"
BASIC_PROMPT_PROFILE = "basic_field_mapping_v1"
DEFAULT_RULES_FILE = "src/chatflow_agent/prompts/enterprise_query_basic_v1.txt"
DEFAULT_FIELD_MAPPING_FILE = "src/chatflow_agent/prompts/enterprise_field_mapping_v1.json"


def parse_prompt_config(value: Any = None) -> dict[str, str]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("prompt配置必须为对象")
    if set(value) - {"profile", "rules_file", "field_mapping_file"}:
        raise ValueError("prompt配置包含未知字段")
    profile = value.get("profile", LEGACY_PROMPT_PROFILE)
    if profile not in (LEGACY_PROMPT_PROFILE, BASIC_PROMPT_PROFILE):
        raise ValueError("未知prompt.profile")
    if profile == LEGACY_PROMPT_PROFILE:
        if set(value) - {"profile"}:
            raise ValueError("legacy提示词不接受外部映射资源")
        return {"profile": profile}
    result = {"profile": profile}
    for key, default in (("rules_file", DEFAULT_RULES_FILE),
                         ("field_mapping_file", DEFAULT_FIELD_MAPPING_FILE)):
        path = value.get(key, default)
        if not isinstance(path, str) or not path.strip():
            raise ValueError("prompt资源路径不能为空：" + key)
        result[key] = str(resolve_project_path(path))
    return result


def load_prompt_resources(config: dict[str, str]) -> dict[str, Any]:
    config = parse_prompt_config(config)
    if config["profile"] != BASIC_PROMPT_PROFILE:
        raise ValueError("仅基础映射profile使用外部提示词资源")
    rules_path = Path(config["rules_file"])
    mapping_path = Path(config["field_mapping_file"])
    rules_bytes = rules_path.read_bytes()
    mapping_bytes = mapping_path.read_bytes()
    rules = rules_bytes.decode("utf-8").strip()
    mapping = json.loads(mapping_bytes.decode("utf-8"))
    if not rules:
        raise ValueError("基础规则资源不能为空")
    if (not isinstance(mapping, dict)
            or set(mapping) != {"version", "schema_version", "table_descriptions", "fields"}
            or mapping["version"] != "enterprise_field_mapping_v1"
            or mapping["schema_version"] != "3.0"):
        raise ValueError("基础字段映射资源版本或结构不正确")
    for key in ("fields", "table_descriptions"):
        values = mapping[key]
        if (not isinstance(values, dict) or not values
                or any(not isinstance(name, str) or not name.strip()
                       or not isinstance(label, str) or not label.strip()
                       or "\n" in label or "\r" in label or len(label) > 80
                       for name, label in values.items())):
            raise ValueError("基础映射须为非空短字段标签：" + key)
    return {
        "rules": rules,
        "field_mapping": dict(mapping["fields"]),
        "table_descriptions": dict(mapping["table_descriptions"]),
        "source_sha256": {
            "rules_file": hashlib.sha256(rules_bytes).hexdigest(),
            "field_mapping_file": hashlib.sha256(mapping_bytes).hexdigest(),
        },
    }
