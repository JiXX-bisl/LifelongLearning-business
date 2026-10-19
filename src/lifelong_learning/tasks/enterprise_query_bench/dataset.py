from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from src.project_paths import resolve_project_path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.lifelong_learning.tasks.enterprise_query_bench.schema_catalog import (
    GUIDED_INTERFACE_PROFILE,
    RESEARCH_INTERFACE_PROFILE,
    build_compact_schemas,
    build_table_catalog,
    expected_tables,
    get_public_business_rules,
    get_public_date_context,
    get_schema_prompt_version,
)


JSONScalar = int | float | str | None

EXPECTED_TABLES = {
    "enterprise_basic",
    "enterprise_person",
    "enterprise_operation_profile",
    "enterprise_operation_monthly",
    "enterprise_operation_event",
    "enterprise_negative_event",
}

FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|REPLACE|CREATE|ALTER|DROP|TRUNCATE|"
    r"RENAME|GRANT|REVOKE|LOCK|UNLOCK|CALL|LOAD)\b|"
    r"\bINTO\s+OUTFILE\b|\bFOR\s+UPDATE\b",
    re.IGNORECASE,
)


class DatasetFormatError(ValueError):
    """企业查询任务集格式错误。"""


class RowOrder(StrEnum):
    ORDERED = "ordered"
    UNORDERED = "unordered"


class Difficulty(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class ResultType(StrEnum):
    SINGLE_VALUE = "single_value"
    SINGLE_ROW = "single_row"
    MULTI_ROW = "multi_row"
    EMPTY = "empty"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    AUTO_VALIDATED = "auto_validated"
    REVIEWED = "reviewed"


class ComparisonPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_order: RowOrder


class AnswerInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direct: list[list[JSONScalar]]
    sql: str
    comparison_policy: ComparisonPolicy

    @field_validator("sql")
    @classmethod
    def validate_ground_truth_sql(cls, value: str) -> str:
        sql = value.strip()

        if not re.match(r"^(?:SELECT|WITH)\b", sql, re.IGNORECASE):
            raise ValueError(
                "Ground Truth SQL must start with SELECT or WITH."
            )

        if not sql.endswith(";") or sql.count(";") != 1:
            raise ValueError(
                "Ground Truth SQL must be one statement ending "
                "with one semicolon."
            )

        if FORBIDDEN_SQL_PATTERN.search(sql):
            raise ValueError(
                "Ground Truth SQL contains a forbidden keyword."
            )

        return sql


class SchemaInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    schema_ref: str = Field(min_length=1)


class TaskMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_query_type: str = Field(min_length=1)
    difficulty: Difficulty
    task_family_id: str = Field(min_length=1)
    ground_truth_tables: list[str] = Field(min_length=1)
    result_type: ResultType
    generation_source: str = Field(min_length=1)
    review_status: ReviewStatus


class EnterpriseQueryEntry(BaseModel):
    """企业查询任务的完整后台记录。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^ENT-[DQ]-\d{6}$")
    instruction: str = Field(min_length=1)
    answer_info: AnswerInfo
    schema_info: SchemaInfo
    skill_list: list[str] = Field(min_length=1)
    metadata: TaskMetadata


@dataclass(frozen=True)
class EnterpriseQueryDataset:
    """加载完成的企业查询任务集及其统一数据库结构。"""

    entries: dict[str, EnterpriseQueryEntry]
    baseline_id: str
    schema_version: str
    database_name: str
    expected_record_counts: dict[str, int]
    schema_sql: str
    table_schema_map: dict[str, str]
    table_catalog: str
    interface_profile: str
    schema_prompt_version: str
    data_file_path: Path
    baseline_dir: Path
    public_date_context: dict[str, Any] = field(default_factory=dict)

    def get_sample_index_list(self) -> list[str]:
        return list(self.entries.keys())

    def build_agent_instruction(
        self,
        sample_index: str,
    ) -> str:
        entry = self.entries[sample_index]
        dates = self.public_date_context or get_public_date_context(self.schema_version)
        date_background = (
            f"基准日：{dates['reference_date']}；"
            f"月度观察期：{dates['observation_window_start'][:7]}"
            f"至{dates['observation_window_end'][:7]}。\n"
        )
        if self.schema_version == "3.0":
            date_background += (
                f"Schema版本：3.0；可用知识截止：{dates['knowledge_cutoff_at']}；"
                f"日期与无时区时间戳：{dates['timezone']}。\n"
                "运营概况基础快照日期："
                + "、".join(dates["profile_snapshot_dates"])
                + "。\n公开数据口径：\n"
                + "\n".join(get_public_business_rules(self.schema_version))
                + "\n"
            )

        if (
            self.interface_profile
            == RESEARCH_INTERFACE_PROFILE
        ):
            return (
                f"当前数据库：{self.database_name}"
                "（MySQL 8.0）。\n"
                + (date_background if self.schema_version == "3.0" else "")
                + "可用表：\n"
                f"{self.table_catalog}\n\n"
                f"用户问题：\n{entry.instruction}"
            )

        return (
            f"当前数据库：{self.database_name}"
            "（MySQL 8.0）。\n"
            f"{date_background}"
            "可用表目录：\n"
            f"{self.table_catalog}\n\n"
            "根据问题自行选表；"
            "需要字段说明时使用 DESCRIBE 表名。\n"
            f"用户问题：\n{entry.instruction}"
        )

    def describe_table(
        self,
        table_name: str,
    ) -> str:
        if table_name not in self.table_schema_map:
            raise ValueError(
                f"Unknown table: {table_name}"
            )

        return self.table_schema_map[table_name]


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise DatasetFormatError(
                f"Duplicate JSON key detected: {key!r}"
            )
        result[key] = value

    return result


def _load_json_strict(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(
            file,
            object_pairs_hook=_reject_duplicate_keys,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _extract_create_table_sql(
    schema_text: str, *, schema_version: str = "2.1",
) -> str:
    """
    从mysqldump结构文件中提取CREATE TABLE语句。

    不向Agent提供DROP TABLE、SET和mysqldump控制语句。
    """
    pattern = re.compile(
        r"CREATE TABLE\s+`(?P<table_name>[^`]+)`\s*"
        r"\(.*?\)\s*ENGINE=.*?;",
        re.IGNORECASE | re.DOTALL,
    )

    statements: list[str] = []
    table_names: set[str] = set()
    schema_tables = expected_tables(schema_version)

    for match in pattern.finditer(schema_text):
        table_name = match.group("table_name")
        if table_name in table_names:
            raise DatasetFormatError(f"Duplicate Schema table: {table_name}")
        statements.append(match.group(0).strip())
        table_names.add(table_name)

    if table_names != schema_tables:
        missing = sorted(schema_tables - table_names)
        unexpected = sorted(table_names - schema_tables)

        raise DatasetFormatError(
            "Schema table set mismatch. "
            f"missing={missing}, unexpected={unexpected}"
        )

    return "\n\n".join(statements)


def load_enterprise_query_dataset(
    data_file_path: str | Path,
    baseline_dir: str | Path,
    interface_profile: str = GUIDED_INTERFACE_PROFILE,
) -> EnterpriseQueryDataset:
    """
    加载口语化企业查询Entry文件和冻结数据库Schema。

    baseline_dir应直接指向已由用户验收并冻结的基线目录；
    Schema 2.1使用原六表，Schema 3.0使用新七表。
    候选基线不会自动启用问题库或跳过冻结门槛。
    """
    data_path = resolve_project_path(data_file_path)
    baseline_path = resolve_project_path(baseline_dir)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset file does not exist: {data_path}"
        )

    manifest_path = baseline_path / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Baseline manifest does not exist: {manifest_path}"
        )

    raw_dataset = _load_json_strict(data_path)
    manifest = _load_json_strict(manifest_path)

    if not isinstance(raw_dataset, dict):
        raise DatasetFormatError(
            "Dataset root must be a JSON object."
        )

    if manifest.get("status") != "FROZEN":
        raise DatasetFormatError(
            "The referenced database baseline is not frozen."
        )

    try:
        sample_indices = sorted(
            raw_dataset.keys(),
            key=int,
        )
    except (TypeError, ValueError) as error:
        raise DatasetFormatError(
            "All top-level sample indices must be numeric strings."
        ) from error

    expected_indices = [
        str(index)
        for index in range(len(sample_indices))
    ]

    if sample_indices != expected_indices:
        raise DatasetFormatError(
            "Sample indices must be continuous and start from 0."
        )

    entries: dict[str, EnterpriseQueryEntry] = {}
    task_ids: set[str] = set()

    for sample_index in sample_indices:
        entry = EnterpriseQueryEntry.model_validate(
            raw_dataset[sample_index]
        )

        if entry.task_id in task_ids:
            raise DatasetFormatError(
                f"Duplicate task_id: {entry.task_id}"
            )

        task_ids.add(entry.task_id)
        entries[sample_index] = entry

    schema_identifiers = {
        (
            entry.schema_info.baseline_id,
            entry.schema_info.schema_version,
            entry.schema_info.schema_ref,
        )
        for entry in entries.values()
    }

    if len(schema_identifiers) != 1:
        raise DatasetFormatError(
            "All tasks must reference the same frozen Schema."
        )

    baseline_id, schema_version, schema_ref = next(
        iter(schema_identifiers)
    )

    if baseline_id != manifest.get("baseline_id"):
        raise DatasetFormatError(
            "Dataset baseline_id differs from manifest baseline_id."
        )

    if schema_version != str(manifest.get("schema_version")):
        raise DatasetFormatError(
            "Dataset schema_version differs from manifest."
        )
    try:
        public_date_context = get_public_date_context(schema_version, manifest=manifest)
    except (ValueError, TypeError) as error:
        raise DatasetFormatError("Manifest Schema version or public dates are invalid.") from error

    schema_path = (baseline_path / schema_ref).resolve()

    # 防止schema_ref越出冻结基线目录。
    if (
        schema_path != baseline_path
        and baseline_path not in schema_path.parents
    ):
        raise DatasetFormatError(
            "schema_ref points outside the baseline directory."
        )

    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Schema file does not exist: {schema_path}"
        )

    artifact_info = (
        manifest.get("artifacts", {}).get(schema_ref)
    )

    if artifact_info is None:
        raise DatasetFormatError(
            f"{schema_ref!r} is not registered in manifest artifacts."
        )

    expected_schema_hash = artifact_info.get("sha256")
    actual_schema_hash = _sha256_file(schema_path)

    if actual_schema_hash != expected_schema_hash:
        raise DatasetFormatError(
            "Schema SHA-256 does not match the frozen manifest."
        )

    ground_truth_tables = {
        table_name
        for entry in entries.values()
        for table_name in entry.metadata.ground_truth_tables
    }

    schema_tables = expected_tables(schema_version)
    unknown_tables = ground_truth_tables - schema_tables

    if unknown_tables:
        raise DatasetFormatError(
            "Tasks reference unknown Ground Truth tables: "
            f"{sorted(unknown_tables)}"
        )

    raw_schema_text = schema_path.read_text(
        encoding="utf-8"
    )
    agent_schema_sql = _extract_create_table_sql(
        raw_schema_text, schema_version=schema_version,
    )

    database_name = manifest["mysql"]["database"]
    expected_record_counts = {
        str(table_name): int(row_count)
        for table_name, row_count
        in manifest["record_counts"].items()
    }

    if set(expected_record_counts) != schema_tables:
        raise DatasetFormatError(
            f"Manifest record_counts does not match schema {schema_version}."
        )

    return EnterpriseQueryDataset(
        entries=entries,
        baseline_id=baseline_id,
        schema_version=schema_version,
        database_name=database_name,
        expected_record_counts=expected_record_counts,
        schema_sql=agent_schema_sql,
        table_schema_map=build_compact_schemas(
            agent_schema_sql,
            interface_profile=interface_profile,
            schema_version=schema_version,
        ),
        table_catalog=build_table_catalog(
            interface_profile=interface_profile,
            schema_version=schema_version,
        ),
        interface_profile=interface_profile,
        schema_prompt_version=(
            get_schema_prompt_version(
                interface_profile, schema_version=schema_version,
            )
        ),
        data_file_path=data_path,
        baseline_dir=baseline_path,
        public_date_context=public_date_context,
    )
