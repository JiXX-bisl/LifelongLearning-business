"""Schema and public field contract for the version 3.0 enterprise dataset.

The query schema contains observed business records. Ground truth, scenario
assignments and the  ledger belong to separate offline artifacts.
"""

from __future__ import annotations


SCHEMA_VERSION = "3.0"

TABLE_DESCRIPTIONS = {
    "enterprise_basic": "企业工商基本信息及其来源",
    "enterprise_person": "已登记关键人员的身份、任职事实与来源记录",
    "enterprise_operation_profile": "不同统计时点的运营概况及报送版本",
    "enterprise_operation_monthly": "月度经营数据及报送版本",
    "enterprise_operation_event": "融资、借款、专利等事件及状态记录",
    "enterprise_negative_event": "纠纷、投诉、报警等事件及处理状态记录",
    "enterprise_data_coverage": "业务数据的采集范围、完成状态与更新时间",
}

COMMON_COLUMNS = (
    ("source_system", "VARCHAR(40) NOT NULL"),
    ("source_record_ref", "VARCHAR(80) NOT NULL"),
    ("reported_at", "DATETIME NOT NULL"),
    ("ingested_at", "DATETIME NOT NULL"),
    ("report_revision", "INT UNSIGNED NOT NULL"),
    ("report_status", "VARCHAR(20) NOT NULL"),
)

AMOUNT_COLUMNS = (("amount_unit", "VARCHAR(10) NOT NULL"),)

COLUMN_DEFINITIONS = {
    "enterprise_basic": (
        ("enterprise_id", "VARCHAR(20) NOT NULL"),
        ("enterprise_name", "VARCHAR(120) NOT NULL"),
        ("synthetic_credit_code", "VARCHAR(24) NOT NULL"),
        ("enterprise_type", "VARCHAR(30) NOT NULL"),
        ("registration_status", "VARCHAR(20) NOT NULL"),
        ("industry", "VARCHAR(50) NOT NULL"),
        ("registered_capital", "DECIMAL(18,2) NOT NULL"),
        ("established_date", "DATE NOT NULL"),
        ("registered_address", "VARCHAR(255) NOT NULL"),
        ("registered_email", "VARCHAR(120) DEFAULT NULL"),
        ("registered_phone", "VARCHAR(30) DEFAULT NULL"),
        ("registered_business_scope", "TEXT NOT NULL"),
        ("main_business_description", "TEXT NOT NULL"),
        ("as_of_date", "DATE NOT NULL"),
        ("registration_status_effective_date", "DATE NOT NULL"),
        ("source_updated_at", "DATETIME NOT NULL"),
    ),
    "enterprise_person": (
        ("person_id", "VARCHAR(24) NOT NULL"),
        ("enterprise_id", "VARCHAR(20) NOT NULL"),
        ("person_entity_id", "VARCHAR(40) NOT NULL"),
        ("assignment_id", "VARCHAR(40) NOT NULL"),
        ("person_name", "VARCHAR(50) NOT NULL"),
        ("age", "SMALLINT DEFAULT NULL"),
        ("household_region", "VARCHAR(50) DEFAULT NULL"),
        ("role", "VARCHAR(30) NOT NULL"),
        ("start_date", "DATE NOT NULL"),
        ("end_date", "DATE DEFAULT NULL"),
        ("is_current", "TINYINT DEFAULT NULL"),
    ) + COMMON_COLUMNS,
    "enterprise_operation_profile": (
        ("profile_record_id", "VARCHAR(40) NOT NULL"),
        ("enterprise_id", "VARCHAR(20) NOT NULL"),
        ("as_of_date", "DATE NOT NULL"),
        ("has_wechat_account", "TINYINT DEFAULT NULL"),
        ("wechat_content_summary", "TEXT"),
        ("total_assets", "DECIMAL(22,6) DEFAULT NULL"),
        ("total_liabilities", "DECIMAL(22,6) DEFAULT NULL"),
    ) + COMMON_COLUMNS + AMOUNT_COLUMNS,
    "enterprise_operation_monthly": (
        ("monthly_record_id", "VARCHAR(40) NOT NULL"),
        ("enterprise_id", "VARCHAR(20) NOT NULL"),
        ("stat_month", "DATE NOT NULL"),
        ("period_start", "DATE NOT NULL"),
        ("period_end", "DATE NOT NULL"),
        ("employee_count", "INT DEFAULT NULL"),
        ("social_security_count", "INT DEFAULT NULL"),
        ("social_security_amount", "DECIMAL(22,6) DEFAULT NULL"),
        ("tax_amount", "DECIMAL(22,6) DEFAULT NULL"),
        ("invoice_purchase_amount", "DECIMAL(22,6) DEFAULT NULL"),
        ("invoice_sales_amount", "DECIMAL(22,6) DEFAULT NULL"),
        ("utility_invoice_amount", "DECIMAL(22,6) DEFAULT NULL"),
        ("travel_hospitality_invoice_amount", "DECIMAL(22,6) DEFAULT NULL"),
        ("operating_revenue", "DECIMAL(22,6) DEFAULT NULL"),
        ("main_business_revenue", "DECIMAL(22,6) DEFAULT NULL"),
        ("total_profit", "DECIMAL(22,6) DEFAULT NULL"),
        ("net_profit", "DECIMAL(22,6) DEFAULT NULL"),
    ) + COMMON_COLUMNS + AMOUNT_COLUMNS,
    "enterprise_operation_event": (
        ("operation_event_id", "VARCHAR(40) NOT NULL"),
        ("enterprise_id", "VARCHAR(20) NOT NULL"),
        ("business_event_ref", "VARCHAR(80) NOT NULL"),
        ("business_object_ref", "VARCHAR(80) NOT NULL"),
        ("event_date", "DATE NOT NULL"),
        ("state_effective_at", "DATETIME NOT NULL"),
        ("event_type", "VARCHAR(30) NOT NULL"),
        ("amount", "DECIMAL(22,6) DEFAULT NULL"),
        ("valuation", "DECIMAL(22,6) DEFAULT NULL"),
        ("financing_stage", "VARCHAR(20) DEFAULT NULL"),
        ("term_months", "SMALLINT DEFAULT NULL"),
        ("quantity", "INT DEFAULT NULL"),
        ("event_status", "VARCHAR(30) NOT NULL"),
        ("outstanding_principal", "DECIMAL(22,6) DEFAULT NULL"),
        ("overdue_principal", "DECIMAL(22,6) DEFAULT NULL"),
        ("balance_as_of_date", "DATE DEFAULT NULL"),
        ("description", "VARCHAR(1000) NOT NULL"),
    ) + COMMON_COLUMNS + AMOUNT_COLUMNS,
    "enterprise_negative_event": (
        ("negative_event_id", "VARCHAR(40) NOT NULL"),
        ("enterprise_id", "VARCHAR(20) NOT NULL"),
        ("related_person_id", "VARCHAR(24) DEFAULT NULL"),
        ("business_event_ref", "VARCHAR(80) NOT NULL"),
        ("business_object_ref", "VARCHAR(80) NOT NULL"),
        ("event_date", "DATE NOT NULL"),
        ("state_effective_at", "DATETIME NOT NULL"),
        ("event_type", "VARCHAR(30) NOT NULL"),
        ("case_category", "VARCHAR(50) DEFAULT NULL"),
        ("description", "VARCHAR(1000) NOT NULL"),
        ("handling_result", "VARCHAR(30) NOT NULL"),
    ) + COMMON_COLUMNS,
    "enterprise_data_coverage": (
        ("coverage_id", "VARCHAR(40) NOT NULL"),
        ("enterprise_id", "VARCHAR(20) NOT NULL"),
        ("table_name", "VARCHAR(50) NOT NULL"),
        ("source_system", "VARCHAR(40) NOT NULL"),
        ("coverage_start", "DATE NOT NULL"),
        ("coverage_end", "DATE NOT NULL"),
        ("coverage_status", "VARCHAR(30) NOT NULL"),
        ("last_synced_at", "DATETIME NOT NULL"),
    ),
}

TABLES = tuple(TABLE_DESCRIPTIONS)
INSERT_ORDER = TABLES
DELETE_ORDER = tuple(reversed(INSERT_ORDER))
TABLE_COLUMNS = {
    table: tuple(name for name, _definition in columns)
    for table, columns in COLUMN_DEFINITIONS.items()
}
COLUMN_TYPES = {
    table: {
        name: definition.split(" NOT NULL")[0].split(" DEFAULT NULL")[0]
        for name, definition in columns
    }
    for table, columns in COLUMN_DEFINITIONS.items()
}
NULLABLE_COLUMNS = {
    table: frozenset(name for name, definition in columns if "NOT NULL" not in definition)
    for table, columns in COLUMN_DEFINITIONS.items()
}
TABLE_DEFINITIONS = {
    table: {
        name: {"type": COLUMN_TYPES[table][name], "nullable": name in NULLABLE_COLUMNS[table]}
        for name in TABLE_COLUMNS[table]
    }
    for table in TABLES
}
PRIMARY_KEYS = {
    "enterprise_basic": ("enterprise_id",),
    "enterprise_person": ("person_id",),
    "enterprise_operation_profile": ("profile_record_id",),
    "enterprise_operation_monthly": ("monthly_record_id",),
    "enterprise_operation_event": ("operation_event_id",),
    "enterprise_negative_event": ("negative_event_id",),
    "enterprise_data_coverage": ("coverage_id",),
}
PRIMARY_KEY_FIELDS = PRIMARY_KEYS
MONEY_FIELDS = {
    "enterprise_basic": ("registered_capital",),
    "enterprise_operation_profile": ("total_assets", "total_liabilities"),
    "enterprise_operation_monthly": (
        "social_security_amount", "tax_amount", "invoice_purchase_amount",
        "invoice_sales_amount", "utility_invoice_amount",
        "travel_hospitality_invoice_amount", "operating_revenue",
        "main_business_revenue", "total_profit", "net_profit",
    ),
    "enterprise_operation_event": (
        "amount", "valuation", "outstanding_principal", "overdue_principal",
    ),
}
MONEY_PRECISION = {
    table: {
        name: {"precision": 18 if table == "enterprise_basic" else 22,
               "scale": 2 if table == "enterprise_basic" else 6}
        for name in names
    }
    for table, names in MONEY_FIELDS.items()
}

FIELD_MEANINGS = {
    "enterprise_id": "稳定企业身份；关联不同业务表的企业外键",
    "enterprise_name": "企业报送名称；名称相同或相近不单独证明身份相同",
    "synthetic_credit_code": "唯一虚拟统一社会信用代码",
    "enterprise_type": "报送的企业组织类型",
    "registration_status": "报送时企业工商经营状态",
    "industry": "报送的所属行业",
    "registered_capital": "工商登记注册资本；固定单位人民币元",
    "established_date": "企业成立日期",
    "registered_address": "报送的工商注册地址",
    "registered_email": "登记邮箱；NULL表示未提供或未知",
    "registered_phone": "登记电话；NULL表示未提供或未知",
    "registered_business_scope": "工商登记经营范围文本",
    "main_business_description": "实际主营业务说明",
    "person_id": "人员报送记录的物理主键；不是稳定人员身份或任职事实数",
    "person_entity_id": "稳定人员身份；同一人的不同任职及报送共享此身份",
    "assignment_id": "一段任职事实身份；同一任职的重复报送及修订共享此身份",
    "person_name": "报送姓名；同名不必为同一人",
    "age": "来源报送的数据基准日年龄；同基准日的版本变化可能为报送更正，并非按报送间隔自然增长；NULL表示未知",
    "household_region": "报送户籍省级地区；NULL表示未提供或未知",
    "role": "该任职事实的角色",
    "start_date": "任职开始日期，含当天",
    "end_date": "任职结束日期，含当天；NULL可能表示尚未记录离任",
    "is_current": "来源报送的基准日是否在任标志：1是、0否、NULL未知",
    "profile_record_id": "运营概况报送行的物理主键",
    "monthly_record_id": "月度经营报送行的物理主键",
    "as_of_date": "本行工商信息或运营概况所对应的业务统计日期",
    "registration_status_effective_date": "本行工商经营状态在业务上开始生效的日期",
    "source_updated_at": "工商信息来源的最后更新时间",
    "has_wechat_account": "报送公众号状态：1有、0无、NULL未知",
    "wechat_content_summary": "公众号内容说明；NULL本身不能证明没有公众号",
    "total_assets": "统计日资产存量；数值单位见本行amount_unit",
    "total_liabilities": "统计日负债存量；数值单位见本行amount_unit",
    "stat_month": "月度标签，以月份第一天表示；实际覆盖起止见period_start/end",
    "period_start": "业务统计或采集覆盖开始日，含当天",
    "period_end": "业务统计或采集覆盖结束日，含当天",
    "employee_count": "该期末劳动合同在册人数；NULL未知、0已报告为零",
    "social_security_count": "该期实际参保缴费人数；NULL未知、0已报告为零",
    "social_security_amount": "该期实际社保缴纳金额；单位见amount_unit",
    "tax_amount": "该期来源报送的实际缴纳税款金额；税种及统计范围结合企业正常报送说明解释，单位见amount_unit",
    "invoice_purchase_amount": "该期进项发票不含税金额；单位见amount_unit",
    "invoice_sales_amount": "该期销项发票不含税金额；单位见amount_unit",
    "utility_invoice_amount": "进项发票中的运维发票不含税金额；单位见amount_unit",
    "travel_hospitality_invoice_amount": "进项发票中的差旅餐饮住宿发票不含税金额；单位见amount_unit",
    "operating_revenue": "该期营业收入发生额，使用不含增值税价款口径；单位见amount_unit",
    "main_business_revenue": "该期主营业务收入发生额，使用不含增值税价款口径；单位见amount_unit",
    "total_profit": "该期利润总额发生额；可为负，单位见amount_unit",
    "net_profit": "该期净利润发生额；可为负，单位见amount_unit",
    "operation_event_id": "运营事件来源记录物理主键",
    "negative_event_id": "负面事件来源记录物理主键",
    "related_person_id": "该事件直接关联的本企业人员报送记录ID；通过其person_entity_id得到人员身份",
    "business_event_ref": "真实事件身份；同一事件的多来源、修订及状态记录共享此值",
    "business_object_ref": "业务事项身份，例如同一借款合同或纠纷事项；可能发生多个真实事件",
    "event_date": "实际业务事件发生日期，与状态生效、报送和入库时间分别记录",
    "state_effective_at": "本行事件状态或处理结果在业务上开始生效的时刻",
    "event_type": "报送事件类型；具体含义需结合业务事项和其他字段",
    "amount": "事件金额；借款为取得本金，借款逾期为该事件逾期本金；单位见amount_unit",
    "valuation": "融资对应企业估值，非实际取得资金；单位见amount_unit",
    "financing_stage": "融资阶段；其他事件不适用时为NULL",
    "term_months": "合同期限（月）；不适用或未知时为NULL",
    "quantity": "本次申请或授权批次涉及件数；不代表当前有效专利总量",
    "event_status": "来源报送的运营事件状态",
    "outstanding_principal": "对应balance_as_of_date的事项未偿本金存量；单位见amount_unit",
    "overdue_principal": "对应balance_as_of_date的事项逾期本金存量；单位见amount_unit",
    "balance_as_of_date": "本行余额金额的业务观察日期；不适用或未知时为NULL",
    "description": "事件事实、来源或处理过程说明",
    "case_category": "报送案件类别；NULL表示未提供或不适用",
    "handling_result": "来源报送的负面事项处理状态；投诉或调查不自动代表确认违法",
    "source_system": "业务来源系统名称，不代表人工可信度结论",
    "source_record_ref": "该来源内同业务事实和统计时点的原始记录引用；状态记录的引用范围包括状态生效时点；结合来源确定修订范围",
    "reported_at": "来源报送时刻；不是业务发生或状态生效时刻",
    "ingested_at": "记录进入可查询数据库的时刻，可用于知识截止时点",
    "report_revision": "同来源、同业务事实和统计时点的正整数报送修订版本，source_record_ref标识该范围；不同来源或时点版本不可直接横比",
    "report_status": "来源报送过程状态：正式或撤回；不代表数据质量或风险标签",
    "amount_unit": "本行观测金额单位：元或万元；人数和件数不受此字段影响",
    "coverage_id": "采集覆盖记录的物理主键",
    "table_name": "覆盖的数据类别，使用业务表完整表名",
    "coverage_start": "该来源采集覆盖开始日，含当天",
    "coverage_end": "该来源采集覆盖结束日，含当天",
    "coverage_status": "采集状态：完整、部分、未采集或不适用；完整表示采集过程完成，不保证每条内容正确",
    "last_synced_at": "该来源和范围最近一次采集同步的时刻",
}

PUBLIC_RULES = (
    "数据基准日2026-08-31；可用知识截止2026-09-15 23:59:59；业务状态时间和入库时间是不同维度。",
    "日期和无时区时间戳均按北京时间（Asia/Shanghai）解释；月度观察窗口为2024-09至2026-08。",
    "本数据集企业、人员、地址、联系方式和事件为虚构。",
    "观测金额按本行amount_unit解释；万元乘10000转换为元；注册资本始终人民币元。",
    "NULL表示未知、未提供或不适用；0表示来源明确报送为零；无记录需结合采集覆盖判断。",
    "同一来源同source_record_ref内report_revision区分修订；来源版本号不表示来源间的优先级。",
    "正式和撤回仅表达报送流程；正式记录仍可能存在字段矛盾或来源冲突。",
    "任职区间包含start_date和end_date；NULL结束日期需结合来源在任标志和更新信息理解。",
    "人员名单覆盖关键管理、财务、技术岗位及董事等登记任职范围；月报employee_count另按劳动合同在册员工口径统计。",
    "月度金额为period_start至period_end的发生额，资产、负债和本金余额为各自统计日存量。",
    "成立前没有经营事实；成立当月发生额仅覆盖实际经营期间，人数仍为期末人数，不能按天数折算人数。",
    "收入按交付、验收或服务期间确认，发票按开具和结算期间记录；二者跨月不自动构成记录错误。",
    "营业收入、主营业务收入及发票金额使用不含增值税价款口径；本数据未提供独立的增值税计缴明细，发票金额不表示含税实付金额。",
    "负利润及负债高于资产可能是真实经营事实；参保人数与在册人数口径不同，差异本身不证明违规。",
    "工商状态、经营数据、人员任职和事件结果各表达不同事实；记录缺失不能自动证明业务未发生。",
)


def build_schema_sql() -> str:
    """Return non-destructive table DDL; the caller chooses a new database."""
    statements = []
    for table in TABLES:
        entries = [f"  `{name}` {definition}" for name, definition in COLUMN_DEFINITIONS[table]]
        key = ", ".join(f"`{field}`" for field in PRIMARY_KEYS[table])
        entries.append(f"  PRIMARY KEY ({key})")
        if table == "enterprise_basic":
            entries.append("  UNIQUE KEY `uq_v30_credit_code` (`synthetic_credit_code`)")
            entries.append("  CONSTRAINT `chk_v30_basic_dates` CHECK (`established_date` <= `as_of_date` AND `registration_status_effective_date` <= `as_of_date`)")
        else:
            entries.append(
                f"  CONSTRAINT `fk_v30_{table.removeprefix('enterprise_')}_enterprise` "
                "FOREIGN KEY (`enterprise_id`) REFERENCES `enterprise_basic` (`enterprise_id`)"
            )
        if table == "enterprise_person":
            entries.extend((
                "  UNIQUE KEY `uq_v30_person_enterprise` (`person_id`, `enterprise_id`)",
                "  KEY `idx_v30_person_identity` (`enterprise_id`, `person_entity_id`, `assignment_id`)",
                "  CONSTRAINT `chk_v30_person_flag` CHECK (`is_current` IS NULL OR `is_current` IN (0, 1))",
                "  CONSTRAINT `chk_v30_person_dates` CHECK (`end_date` IS NULL OR `start_date` <= `end_date`)",
            ))
        elif table == "enterprise_operation_profile":
            entries.append("  KEY `idx_v30_profile_period` (`enterprise_id`, `as_of_date`)")
            entries.append("  CONSTRAINT `chk_v30_profile_flag` CHECK (`has_wechat_account` IS NULL OR `has_wechat_account` IN (0, 1))")
        elif table == "enterprise_operation_monthly":
            entries.append("  KEY `idx_v30_monthly_period` (`enterprise_id`, `stat_month`)")
            entries.append("  CONSTRAINT `chk_v30_monthly_dates` CHECK (`period_start` <= `period_end` AND DAY(`stat_month`) = 1 AND YEAR(`period_start`) = YEAR(`stat_month`) AND MONTH(`period_start`) = MONTH(`stat_month`) AND YEAR(`period_end`) = YEAR(`stat_month`) AND MONTH(`period_end`) = MONTH(`stat_month`))")
            entries.append("  CONSTRAINT `chk_v30_monthly_counts` CHECK ((`employee_count` IS NULL OR `employee_count` >= 0) AND (`social_security_count` IS NULL OR `social_security_count` >= 0))")
        elif table in {"enterprise_operation_event", "enterprise_negative_event"}:
            entries.append("  KEY `idx_v30_event_business` (`enterprise_id`, `business_event_ref`, `state_effective_at`)")
            entries.append("  KEY `idx_v30_event_object` (`enterprise_id`, `business_object_ref`, `event_date`)")
        if table == "enterprise_negative_event":
            entries.extend((
                "  KEY `idx_v30_negative_person` (`related_person_id`, `enterprise_id`)",
                "  CONSTRAINT `fk_v30_negative_person` FOREIGN KEY (`related_person_id`, `enterprise_id`) "
                "REFERENCES `enterprise_person` (`person_id`, `enterprise_id`)",
            ))
        if table == "enterprise_data_coverage":
            entries.append("  KEY `idx_v30_coverage_domain` (`enterprise_id`, `table_name`, `coverage_start`, `coverage_end`)")
            entries.append("  CONSTRAINT `chk_v30_coverage_status` CHECK (`coverage_status` IN ('完整', '部分', '未采集', '不适用'))")
            entries.append("  CONSTRAINT `chk_v30_coverage_dates` CHECK (`coverage_start` <= `coverage_end`)")
        elif table != "enterprise_basic":
            suffix = table.removeprefix("enterprise_")
            entries.append(f"  CONSTRAINT `chk_v30_{suffix}_revision` CHECK (`report_revision` >= 1)")
            entries.append(f"  CONSTRAINT `chk_v30_{suffix}_report_status` CHECK (`report_status` IN ('正式', '撤回'))")
            entries.append(f"  CONSTRAINT `chk_v30_{suffix}_report_time` CHECK (`reported_at` <= `ingested_at`)")
            if "amount_unit" in TABLE_COLUMNS[table]:
                entries.append(f"  CONSTRAINT `chk_v30_{suffix}_amount_unit` CHECK (`amount_unit` IN ('元', '万元'))")
        statements.append(
            f"CREATE TABLE `{table}` (\n" + ",\n".join(entries)
            + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;"
        )
    return "\n\n".join(statements) + "\n"


def build_database_sql() -> str:
    """Separate user-operated database creation; no replacement or old-library writes."""
    return (
        "-- 仅由用户执行；先创建独立空库，再在该库执行 database_schema.sql。\n"
        "CREATE DATABASE `synthetic_enterprise_v30_120` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;\n"
    )


def public_dictionary() -> dict:
    """Return public schema facts only; never include hidden answer labels."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tables": {
            table: {
                "description": TABLE_DESCRIPTIONS[table],
                "primary_key": list(PRIMARY_KEYS[table]),
                "columns": [
                    {"name": name, "definition": definition, "description": FIELD_MEANINGS[name]}
                    for name, definition in COLUMN_DEFINITIONS[table]
                ],
            }
            for table in TABLES
        },
        "business_rules": list(PUBLIC_RULES),
    }
