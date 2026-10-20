from __future__ import annotations

import re
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from src.project_paths import resolve_project_path
from src.chatflow_agent.prompt_resources import (
    BASIC_PROMPT_PROFILE, load_prompt_resources, parse_prompt_config,
)

from src.lifelong_learning.configuration.yaml_loader import load_yaml_config
from src.lifelong_learning.environments.fixed_mysql_environment import (
    FixedMySQLReadOnlyEnvironment,
)
from src.lifelong_learning.tasks.enterprise_query_bench.schema_catalog import (
    get_public_date_context,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BusinessRuntimeError(RuntimeError):
    """The business runtime cannot start with the current configuration."""


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BusinessRuntimeError(f"配置项 {name} 必须是对象。")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BusinessRuntimeError(f"缺少配置项 {name}。")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise BusinessRuntimeError(f"配置项 {name} 必须是正整数。") from error
    if result <= 0:
        raise BusinessRuntimeError(f"配置项 {name} 必须是正整数。")
    return result

def parse_learning_config(value: Any) -> dict[str, Any]:
    """解析业务学习配置；未提供的字段使用默认值。"""
    learning = _mapping(value, "learning")

    mode = _text(
        learning.get("mode", "standard"),
        "learning.mode",
    )
    if mode not in {"standard", "psu", "gsc"}:
        raise BusinessRuntimeError(
            "配置项 learning.mode 只支持 standard、psu、gsc。"
        )

    experiment_id = _text(
        learning.get("experiment_id", "q05_dev"),
        "learning.experiment_id",
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", experiment_id) is None:
        raise BusinessRuntimeError(
            "配置项 learning.experiment_id 必须以英文字母或数字开头，"
            "且只能包含英文字母、数字、下划线或连字符。"
        )

    return {
        "mode": mode,
        "experiment_id": experiment_id,
        "utilized_sample_count": _positive_int(
            learning.get("utilized_sample_count", 4),
            "learning.utilized_sample_count",
        ),
        "group_count": _positive_int(
            learning.get("group_count", 2),
            "learning.group_count",
        ),
        "sample_count_per_group": _positive_int(
            learning.get("sample_count_per_group", 2),
            "learning.sample_count_per_group",
        ),
    }

def _project_path(value: Any, name: str) -> Path:
    return resolve_project_path(_text(value, name), project_root=PROJECT_ROOT)


def _load_manifest(baseline_dir: Path) -> dict[str, Any]:
    manifest_path = baseline_dir / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except FileNotFoundError as error:
        raise BusinessRuntimeError(
            f"数据基线manifest不存在：{manifest_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise BusinessRuntimeError(
            f"无法读取数据基线manifest：{error}"
        ) from error
    return _mapping(manifest, "data.manifest")


def _check_output_directory(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir) as temporary_file:
            temporary_file.write(b"runtime-check")
            temporary_file.flush()
    except OSError as error:
        raise BusinessRuntimeError(
            f"输出目录不可写：{output_dir}"
        ) from error


def check_business_runtime(config_path: str | Path) -> dict[str, Any]:
    """Validate the minimum dependencies required to start the business agent."""
    resolved_config_path = Path(config_path).expanduser()
    if not resolved_config_path.is_absolute():
        resolved_config_path = (PROJECT_ROOT / resolved_config_path).resolve()

    try:
        config = load_yaml_config(resolved_config_path)
    except FileNotFoundError as error:
        raise BusinessRuntimeError(
            f"业务配置不存在：{resolved_config_path}"
        ) from error
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        raise BusinessRuntimeError(
            "业务配置无法读取，请检查文件权限及YAML格式。"
        ) from error

    model = _mapping(config.get("model"), "model")
    database = _mapping(config.get("database"), "database")
    data = _mapping(config.get("data"), "data")
    output = _mapping(config.get("output"), "output")
    learning = parse_learning_config(config.get("learning", {}))
    try:
        prompt_config = parse_prompt_config(config.get("prompt"))
        if prompt_config["profile"] == BASIC_PROMPT_PROFILE:
            load_prompt_resources(prompt_config)
    except (ValueError, OSError) as error:
        raise BusinessRuntimeError("提示词配置或资源无效：" + str(error)) from error

    model_type = _text(model.get("type"), "model.type")
    if model_type != "openai_compatible":
        raise BusinessRuntimeError(f"不支持的模型类型：{model_type}。")
    model_name = _text(
        os.environ.get("OPENAI_MODEL_NAME", "").strip() or model.get("model_name"),
        "model.model_name",
    )
    api_key_env = _text(model.get("api_key_env"), "model.api_key_env")
    base_url_env = _text(model.get("base_url_env"), "model.base_url_env")
    password_env = _text(database.get("password_env"), "database.password_env")

    if (
        model.get("api_key") is not None
        or model.get("base_url") is not None
        or database.get("password") is not None
    ):
        raise BusinessRuntimeError(
            "模型地址和凭据不能写入业务配置，请使用环境变量。"
        )

    for environment_name in (api_key_env, base_url_env, password_env):
        if not os.environ.get(environment_name, "").strip():
            raise BusinessRuntimeError(f"缺少环境变量 {environment_name}。")

    baseline_dir = _project_path(data.get("baseline_dir"), "data.baseline_dir")
    expected_baseline_id = _text(
        data.get("expected_baseline_id"),
        "data.expected_baseline_id",
    )
    manifest = _load_manifest(baseline_dir)
    actual_baseline_id = _text(manifest.get("baseline_id"), "manifest.baseline_id")
    if actual_baseline_id != expected_baseline_id:
        raise BusinessRuntimeError(
            "数据基线版本不匹配："
            f"配置为 {expected_baseline_id}，实际为 {actual_baseline_id}。"
        )
    schema_version = _text(manifest.get("schema_version", "2.1"), "manifest.schema_version")
    if prompt_config["profile"] == BASIC_PROMPT_PROFILE and schema_version != "3.0":
        raise BusinessRuntimeError("基础字段映射提示词仅支持Schema 3.0")
    expected_schema_version = data.get("expected_schema_version")
    if expected_schema_version is not None:
        expected_schema_version = _text(expected_schema_version, "data.expected_schema_version")
        if schema_version != expected_schema_version:
            raise BusinessRuntimeError(
                "Schema版本不匹配："
                f"配置为 {expected_schema_version}，实际为 {schema_version}。"
            )
    try:
        public_date_context = get_public_date_context(schema_version, manifest=manifest)
    except (ValueError, TypeError):
        raise BusinessRuntimeError("基线Schema版本或公开日期背景无效。") from None

    mysql_manifest = _mapping(manifest.get("mysql"), "manifest.mysql")
    database_name = _text(mysql_manifest.get("database"), "manifest.mysql.database")
    output_dir = _project_path(output.get("directory"), "output.directory")
    _check_output_directory(output_dir)

    host = _text(database.get("host"), "database.host")
    port = _positive_int(database.get("port"), "database.port")
    user = _text(database.get("user"), "database.user")
    connection_timeout_seconds = _positive_int(
        database.get("connection_timeout_seconds"),
        "database.connection_timeout_seconds",
    )
    fetch_batch_size = _positive_int(
        database.get("fetch_batch_size", 50), 
        "database.fetch_batch_size",
    )
    max_total_rows = _positive_int(
        database.get("max_total_rows", 500),
        "database.max_total_rows",
    )

    environment: FixedMySQLReadOnlyEnvironment | None = None
    try:
        environment = FixedMySQLReadOnlyEnvironment(
            database=database_name,
            host=host,
            port=port,
            user=user,
            password_env=password_env,
            connection_timeout_seconds=connection_timeout_seconds,
        )
        if environment.execute("SELECT 1") != [(1,)]:
            raise RuntimeError("SELECT 1未返回预期结果")
    except Exception as error:
        raise BusinessRuntimeError(
            "数据库连接失败，请检查数据库服务和连接配置。"
        ) from error
    finally:
        if environment is not None:
            environment.close()

    return {
        "config_path": str(resolved_config_path),
        "model": {
            "type": model_type,
            "model_name": model_name,
            "api_key_env": api_key_env,
            "base_url_env": base_url_env,
            "probe_performed": False,
        },
        "database": {
            "host": host,
            "port": port,
            "user": user,
            "password_env": password_env,
            "database": database_name,
            "connection_timeout_seconds": connection_timeout_seconds,
            "fetch_batch_size": fetch_batch_size,
            "max_total_rows": max_total_rows,
        },
        "data": {
            "baseline_id": actual_baseline_id,
            "baseline_dir": str(baseline_dir),
            "schema_version": schema_version,
            "public_date_context": public_date_context,
        },
        "output_dir": str(output_dir),
        "learning": learning,
        "prompt": prompt_config,
    }
