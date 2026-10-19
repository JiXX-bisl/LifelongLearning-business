from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Mapping


GUIDED_INTERFACE_PROFILE = "guided_v2"
RESEARCH_INTERFACE_PROFILE = "research_v3"

SCHEMA_PROMPT_VERSION_BY_PROFILE = {
    GUIDED_INTERFACE_PROFILE: (
        "on_demand_schema_v1"
    ),
    RESEARCH_INTERFACE_PROFILE: (
        "on_demand_schema_v2_research"
    ),
}


def get_schema_prompt_version(
    interface_profile: str,
    *,
    schema_version: str = "2.1",
) -> str:
    get_table_descriptions(schema_version)
    try:
        version = SCHEMA_PROMPT_VERSION_BY_PROFILE[
            interface_profile
        ]
    except KeyError as error:
        raise ValueError(
            "Unsupported enterprise query interface "
            f"profile: {interface_profile}"
        ) from error
    return version if schema_version == "2.1" else f"{version}_schema30"

TABLE_DESCRIPTIONS = {
    "enterprise_basic": "工商基本信息",
    "enterprise_person": "核心人员及任职信息",
    "enterprise_operation_profile": "基准日运营概况",
    "enterprise_operation_monthly": "月度经营数据",
    "enterprise_operation_event": "融资、借款、专利等运营事件",
    "enterprise_negative_event": "纠纷、投诉、报警等负面事件",
}

FIELD_MEANINGS = {
    "enterprise_id": "企业ID",
    "enterprise_name": "企业名称",
    "synthetic_credit_code": "虚拟统一社会信用代码",
    "enterprise_type": "企业类型",
    "registration_status": "经营状态",
    "industry": "所属行业",
    "registered_capital": "注册资本，元",
    "established_date": "成立日期",
    "registered_address": "注册地址",
    "registered_email": "登记邮箱",
    "registered_phone": "登记电话",
    "registered_business_scope": "工商登记经营范围",
    "main_business_description": "主营业务描述",
    "person_id": "人员ID",
    "person_name": "人员姓名",
    "age": "年龄，岁",
    "household_region": "户籍地区",
    "role": "人员角色",
    "start_date": "任职起始日期",
    "end_date": "任职结束日期",
    "is_current": "是否现任：1是、0否",
    "as_of_date": "基准日",
    "has_wechat_account": "是否开设公众号：1是、0否",
    "wechat_content_summary": "公众号内容摘要",
    "total_assets": "基准日总资产，元",
    "total_liabilities": "基准日总负债，元",
    "stat_month": "统计月份，以该月1日表示",
    "employee_count": "员工人数",
    "social_security_count": "社保缴纳人数",
    "social_security_amount": "社保缴纳金额，元",
    "tax_amount": "纳税金额，元",
    "invoice_purchase_amount": "进项发票金额，元",
    "invoice_sales_amount": "销项发票金额，元",
    "utility_invoice_amount": "运维发票金额，元",
    "travel_hospitality_invoice_amount": "差旅接待发票金额，元",
    "operating_revenue": "营业收入，元",
    "main_business_revenue": "主营业务收入，元",
    "total_profit": "利润总额，元",
    "net_profit": "净利润，元",
    "operation_event_id": "运营事件ID",
    "negative_event_id": "负面事件ID",
    "related_person_id": "相关人员ID",
    "event_date": "事件发生日期",
    "event_type": "事件类型",
    "amount": "事件金额，元",
    "valuation": "估值，元",
    "financing_stage": "融资阶段",
    "term_months": "期限，月",
    "quantity": "数量，件",
    "event_status": "事件状态",
    "description": "事件描述",
    "case_category": "案件类别",
    "handling_result": "处理结果",
}


def get_table_descriptions(schema_version: str = "2.1") -> dict[str, str]:
    """Select an explicit schema contract without changing the frozen v2.1 API."""
    if schema_version == "2.1":
        return dict(TABLE_DESCRIPTIONS)
    if schema_version == "3.0":
        from src.enterprise_dataset_v30.schema import TABLE_DESCRIPTIONS as descriptions
        return dict(descriptions)
    raise ValueError(f"Unsupported enterprise schema version: {schema_version}")


def get_field_meanings(schema_version: str = "2.1") -> dict[str, str]:
    get_table_descriptions(schema_version)
    if schema_version == "2.1":
        return dict(FIELD_MEANINGS)
    from src.enterprise_dataset_v30.schema import FIELD_MEANINGS as meanings
    return dict(meanings)


def expected_tables(schema_version: str = "2.1") -> set[str]:
    return set(get_table_descriptions(schema_version))


def get_public_date_context(
    schema_version: str = "2.1",
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the public business dates without exposing evaluation artifacts."""
    get_table_descriptions(schema_version)
    metadata = {} if manifest is None else manifest
    reference_date = str(metadata.get(
        "reference_date", metadata.get("dataset_reference_date", "2026-08-31"),
    ))
    anchor = date.fromisoformat(reference_date)
    previous_year, previous_month = divmod(anchor.year * 12 + anchor.month - 1 - 11, 12)
    legacy_window_start = date(previous_year, previous_month + 1, 1).isoformat()
    window = metadata.get("observation_window", {})
    if not isinstance(window, Mapping):
        raise ValueError("Public observation window must be an object.")
    start_date = str(metadata.get(
        "observation_window_start",
        window.get("start", "2024-09-01" if schema_version == "3.0" else legacy_window_start),
    ))
    end_date = str(metadata.get("observation_window_end", window.get("end", reference_date)))
    if date.fromisoformat(start_date) > date.fromisoformat(end_date):
        raise ValueError("Public observation window start exceeds its end.")
    context: dict[str, Any] = {
        "reference_date": reference_date,
        "observation_window_start": start_date,
        "observation_window_end": end_date,
        "timezone": str(metadata.get("timezone", "Asia/Shanghai")),
    }
    if schema_version == "3.0":
        cutoff = str(metadata.get("knowledge_cutoff_at", "2026-09-15 23:59:59"))
        datetime.fromisoformat(cutoff)
        snapshots = metadata.get("profile_snapshot_dates", [
            "2025-08-31", "2025-12-31", "2026-03-31", "2026-08-31",
        ])
        if not isinstance(snapshots, list) or any(not isinstance(value, str) for value in snapshots):
            raise ValueError("Public profile snapshot dates must be a list of date strings.")
        for value in snapshots:
            date.fromisoformat(value)
        context["knowledge_cutoff_at"] = cutoff
        context["profile_snapshot_dates"] = list(snapshots)
    return context


def get_public_business_rules(schema_version: str = "2.1") -> tuple[str, ...]:
    get_table_descriptions(schema_version)
    if schema_version == "2.1":
        return ()
    from src.enterprise_dataset_v30.schema import PUBLIC_RULES
    return tuple(PUBLIC_RULES)

TABLE_PATTERN = re.compile(
    r"^CREATE TABLE\s+`([^`]+)`\s*\(\s*\n"
    r"(.*?)"
    r"^\)\s*ENGINE=.*?;",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

COLUMN_PATTERN = re.compile(
    r"^`([^`]+)`\s+(.+)$"
)


def _validate_interface_profile(
    interface_profile: str,
) -> None:
    if (
        interface_profile
        not in SCHEMA_PROMPT_VERSION_BY_PROFILE
    ):
        raise ValueError(
            "Unsupported enterprise query interface "
            f"profile: {interface_profile}"
        )


def build_table_catalog(
    interface_profile: str = GUIDED_INTERFACE_PROFILE,
    *,
    schema_version: str = "2.1",
) -> str:
    """
    guided_v2返回表名及中文用途；
    research_v3只返回表名。
    """
    _validate_interface_profile(interface_profile)
    descriptions = get_table_descriptions(schema_version)

    if interface_profile == RESEARCH_INTERFACE_PROFILE:
        return "\n".join(
            f"- {name}"
            for name in descriptions
        )

    return "\n".join(
        f"- {name}：{description}"
        for name, description
        in descriptions.items()
    )


def build_compact_schemas(
    schema_sql: str,
    interface_profile: str = GUIDED_INTERFACE_PROFILE,
    *,
    schema_version: str = "2.1",
) -> dict[str, str]:
    """
    从冻结mysqldump中提取按需Schema。

    两种模式都保留字段、类型、可空性、枚举、
    主键、唯一键和外键。

    guided_v2额外提供中文字段含义；
    research_v3在Schema 2.1中保留原不提供字段解释的接口；
    Schema 3.0的两种模式均提供正常身份、时间、单位和报送语义。
    """
    _validate_interface_profile(interface_profile)
    descriptions = get_table_descriptions(schema_version)
    meanings = get_field_meanings(schema_version)

    result: dict[str, str] = {}

    for match in TABLE_PATTERN.finditer(schema_sql):
        table_name, body = match.groups()

        if (
            table_name not in descriptions
            or table_name in result
        ):
            raise ValueError(
                f"Unexpected or duplicate table: "
                f"{table_name}"
            )

        if interface_profile == RESEARCH_INTERFACE_PROFILE:
            lines = [table_name]
        else:
            lines = [
                f"{table_name}："
                f"{descriptions[table_name]}"
            ]

        column_count = 0

        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            column = COLUMN_PATTERN.fullmatch(line)

            if column:
                name, definition = column.groups()

                if interface_profile == GUIDED_INTERFACE_PROFILE or schema_version == "3.0":
                    if name not in meanings:
                        raise ValueError(
                            "Missing field description: "
                            f"{name}"
                        )

                    lines.append(
                        f"- {name}: {definition}；"
                        f"{meanings[name]}"
                    )
                else:
                    lines.append(
                        f"- {name}: {definition}"
                    )

                column_count += 1

            elif line.startswith(
                ("PRIMARY KEY", "UNIQUE KEY")
            ):
                lines.append(f"- {line}")

            elif "FOREIGN KEY" in line:
                relationship = re.search(
                    r"FOREIGN KEY\s*\([^)]*\)\s*"
                    r"REFERENCES\s*`[^`]+`\s*\([^)]*\)",
                    line,
                )

                if relationship is None:
                    raise ValueError(
                        "Cannot parse foreign key: "
                        f"{line}"
                    )

                lines.append(
                    f"- {relationship.group(0)}"
                )

        if column_count == 0:
            raise ValueError(
                f"No columns extracted for {table_name}"
            )

        result[table_name] = "\n".join(lines)

    if set(result) != set(descriptions):
        raise ValueError(
            f"Cannot extract the complete schema {schema_version} table set."
        )

    return result


def build_basic_schema_catalog(
    schema_sql: str,
    field_mapping: dict[str, str],
    *,
    schema_version: str = "3.0",
    table_descriptions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the seven-table structural dictionary without business guidance.

    Table membership and keys remain local to each table.  A shared column's
    DDL definition and short label are stored once.  The caller supplies the
    labels explicitly; this function never falls back to the guided meanings.
    """
    if schema_version != "3.0":
        raise ValueError("The basic field-mapping catalog requires Schema 3.0.")
    expected_descriptions = get_table_descriptions(schema_version)
    descriptions = dict(
        expected_descriptions if table_descriptions is None else table_descriptions
    )
    if set(descriptions) != set(expected_descriptions):
        raise ValueError("Basic table descriptions must cover exactly the seven Schema 3.0 tables.")
    if any(not isinstance(value, str) or not value.strip() for value in descriptions.values()):
        raise ValueError("Basic table descriptions must be nonempty strings.")

    table_fields: dict[str, list[str]] = {}
    field_definitions: dict[str, str] = {}
    table_keys: dict[str, list[str]] = {}
    for match in TABLE_PATTERN.finditer(schema_sql):
        table_name, body = match.groups()
        if table_name not in descriptions or table_name in table_fields:
            raise ValueError(f"Unexpected or duplicate table: {table_name}")
        fields: list[str] = []
        keys: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            column = COLUMN_PATTERN.fullmatch(line)
            if column:
                name, definition = column.groups()
                if name in fields:
                    raise ValueError(f"Duplicate field in {table_name}: {name}")
                if name in field_definitions and field_definitions[name] != definition:
                    raise ValueError(
                        f"Conflicting shared field definition for {name}: "
                        f"{field_definitions[name]!r} versus {definition!r} in {table_name}"
                    )
                fields.append(name)
                field_definitions.setdefault(name, definition)
            elif line.startswith(("PRIMARY KEY", "UNIQUE KEY")):
                keys.append(line)
            elif "FOREIGN KEY" in line:
                _basic_foreign_key_relationship(line)
                keys.append(line)
        if not fields:
            raise ValueError(f"No columns extracted for {table_name}")
        table_fields[table_name] = fields
        table_keys[table_name] = keys

    if set(table_fields) != set(expected_descriptions):
        raise ValueError("Cannot extract the complete schema 3.0 seven-table set.")
    missing = set(field_definitions) - set(field_mapping)
    extra = set(field_mapping) - set(field_definitions)
    if missing or extra:
        raise ValueError(
            "Basic field mapping must match the schema fields exactly; "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )
    if any(not isinstance(value, str) or not value.strip() for value in field_mapping.values()):
        raise ValueError("Basic field labels must be nonempty strings.")

    # Canonical schema order keeps prompts identical even when configuration
    # dictionaries or CREATE statements are presented in another order.
    ordered_tables = list(expected_descriptions)
    ordered_fields = list(dict.fromkeys(
        name for table_name in ordered_tables for name in table_fields[table_name]
    ))
    return {
        "table_fields": {name: table_fields[name] for name in ordered_tables},
        "field_definitions": {name: field_definitions[name] for name in ordered_fields},
        "field_mapping": {name: field_mapping[name] for name in ordered_fields},
        "table_descriptions": {name: descriptions[name] for name in ordered_tables},
        "table_keys": {name: table_keys[name] for name in ordered_tables},
    }


def _basic_foreign_key_relationship(key: str) -> str:
    relationship = re.search(
        r"FOREIGN KEY\s*\([^)]*\)\s*"
        r"REFERENCES\s*`[^`]+`\s*\([^)]*\)(?:\s+ON\s+(?:DELETE|UPDATE)\s+"
        r"(?:RESTRICT|CASCADE|SET NULL|NO ACTION))*$",
        key,
        re.IGNORECASE,
    )
    if relationship is None:
        raise ValueError(f"Cannot parse foreign key: {key}")
    return relationship.group(0)


def render_basic_schema_catalog(catalog: Mapping[str, Any]) -> str:
    """Render table membership and one global type/label line per field."""
    table_fields = catalog["table_fields"]
    definitions = catalog["field_definitions"]
    labels = catalog["field_mapping"]
    descriptions = catalog["table_descriptions"]
    table_keys = catalog["table_keys"]
    lines = ["七表及字段归属："]
    for table_name, fields in table_fields.items():
        lines.extend((
            "", f"{table_name}（{descriptions[table_name]}）：", ", ".join(fields),
        ))

    lines.extend(("", "键关系："))
    enterprise_relationship = (
        "FOREIGN KEY (`enterprise_id`) REFERENCES `enterprise_basic` (`enterprise_id`)"
    )
    shared_enterprise_tables = {
        table_name
        for table_name, keys in table_keys.items()
        if any(
            "FOREIGN KEY" in key
            and _basic_foreign_key_relationship(key) == enterprise_relationship
            for key in keys
        )
    }
    # Compress only the exact relationship that applies to every other table;
    # never infer or silently omit a missing/changed foreign key.
    shared_enterprise_keys = shared_enterprise_tables == (
        set(table_fields) - {"enterprise_basic"}
    )
    if shared_enterprise_keys:
        lines.append("除enterprise_basic外，各表enterprise_id外键均关联enterprise_basic.enterprise_id。")
    for table_name, keys in table_keys.items():
        for key in keys:
            relationship = _basic_foreign_key_relationship(key) if "FOREIGN KEY" in key else key
            if shared_enterprise_keys and relationship == enterprise_relationship:
                continue
            lines.append(f"{table_name}: {relationship}")

    lines.extend((
        "", "字段映射及类型（同名字段共用，类型包含可空性）：",
        "NULL：未知、未提供或不适用；0：已报送为零，标志字段按枚举解释。",
    ))
    lines.extend(
        f"{name} → {labels[name]} | {definition}"
        for name, definition in definitions.items()
    )
    return "\n".join(lines)


def basic_table_schemas(catalog: Mapping[str, Any]) -> dict[str, str]:
    """Supply DESCRIBE responses from the same short field dictionary."""
    definitions = catalog["field_definitions"]
    labels = catalog["field_mapping"]
    descriptions = catalog["table_descriptions"]
    result: dict[str, str] = {}
    for table_name, fields in catalog["table_fields"].items():
        lines = [f"{table_name}：{descriptions[table_name]}"]
        lines.extend(
            f"- {name}: {definitions[name]}；{labels[name]}" for name in fields
        )
        lines.extend(f"- {key}" for key in catalog["table_keys"][table_name])
        result[table_name] = "\n".join(lines)
    return result
