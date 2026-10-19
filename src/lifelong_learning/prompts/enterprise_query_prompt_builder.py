from __future__ import annotations

import json
from pathlib import Path


GUIDED_INTERFACE_PROFILE = "guided_v2"
RESEARCH_INTERFACE_PROFILE = "research_v3"

PROTOCOL_VERSION_BY_PROFILE = {
    GUIDED_INTERFACE_PROFILE: (
        "enterprise_query_protocol_v2"
    ),
    RESEARCH_INTERFACE_PROFILE: (
        "enterprise_query_protocol_v3_research"
    ),
}


GUIDED_TASK_REQUIREMENT = """你需要通过工具查询企业数据库，回答中文用户问题。初始消息提供数据库名称和表目录。

每轮只提交一个动作。

一、按需查看表结构
如果需要了解字段、类型、枚举或主外键关系，使用：

Action: Operation
```sql
DESCRIBE enterprise_person;
```

系统从冻结Schema返回精简字段说明，不返回企业记录。
结构查看最多{schema_lookup_budget}次，错误请求也计数。
这些请求单独计数，不占用业务操作轮数。
已经提供过的结构可以从当前对话历史中读取。

二、执行业务查询
使用：

Action: Operation
```sql
SELECT ...;
```

业务SQL只能使用单条SELECT或WITH查询。
禁止写入、建表、删表、管理操作及多条SQL。
严格遵守结果字段契约：
1. 最终业务SQL只能返回用户明确要求的字段，不得遗漏，也不得添加企业名称、排序键或其他辅助字段；
2. 返回列顺序必须与用户要求的顺序一致；
3. 排序、筛选和关联使用的字段不必出现在SELECT结果中，除非用户明确要求返回；
4. 日期或月份字段默认直接返回MySQL DATE值，由系统统一规范化为YYYY-MM-DD；除非用户明确要求其他格式，不得使用DATE_FORMAT、CAST等方式缩短或改变日期格式。

遵守问题要求的条件、时间、单位和排序。
系统返回实际查询结果或错误信息。

三、结束任务
使用：

Action: Answer
Final Answer: 查询已经完成

业务查询和结束动作合计最多{query_round_budget}轮。
如果已经掌握必要字段，可以直接查询，不强制先查看结构。
SQL失败后，可以在剩余业务轮数内修正。

评价采用B1规则：只比较最后一次成功业务查询的结果。
结构查看结果和最终自然语言说明不参与正确性评分。
"""


RESEARCH_TASK_REQUIREMENT = """你需要通过SQL查询企业数据库回答用户问题。数据库为只读。

如果需要，可以查看数据库表结构。查看表结构时提交：

Action: Operation
```sql
DESCRIBE <table_name>;
```

结构查看最多{schema_lookup_budget}次，并与任务操作轮次分开计数。

每轮只能选择以下一种动作：

1. 执行操作

Action: Operation
```sql
<一条SQL语句>
```

每次只能提交一条SQL。系统会执行该SQL，并返回查询结果或错误信息。

2. 提交答案

Action: Answer
Final Answer: <你的回答>

请在最多{query_round_budget}轮任务操作内完成任务。最终结果必须满足用户提出的查询要求。
"""


TASK_REQUIREMENT_BY_PROFILE = {
    GUIDED_INTERFACE_PROFILE: GUIDED_TASK_REQUIREMENT,
    RESEARCH_INTERFACE_PROFILE: RESEARCH_TASK_REQUIREMENT,
}


STANDARD_SUFFIX = """

现在请处理下面的查询任务。"""


PSU_SUFFIX = """

{previous_sample_utilization_target_position}

现在请处理下面的查询任务。"""


def get_enterprise_query_protocol_version(
    interface_profile: str,
) -> str:
    try:
        return PROTOCOL_VERSION_BY_PROFILE[
            interface_profile
        ]
    except KeyError as error:
        raise ValueError(
            "Unsupported enterprise query interface "
            f"profile: {interface_profile}"
        ) from error


def write_enterprise_query_prompt(
    output_path: str | Path,
    *,
    use_previous_samples: bool,
    interface_profile: str = GUIDED_INTERFACE_PROFILE,
) -> Path:
    try:
        task_requirement = TASK_REQUIREMENT_BY_PROFILE[
            interface_profile
        ]
    except KeyError as error:
        raise ValueError(
            "Unsupported enterprise query interface "
            f"profile: {interface_profile}"
        ) from error

    suffix = (
        PSU_SUFFIX
        if use_previous_samples
        else STANDARD_SUFFIX
    )

    prompt = {
        "metadata": {
            "interface_profile": interface_profile,
            "interaction_protocol_version": (
                get_enterprise_query_protocol_version(
                    interface_profile
                )
            ),
        },
        "value": {
            "0": {
                "role": "user",
                "content": (
                    f"{task_requirement}{suffix}"
                ),
            },
            "1": {
                "role": "agent",
                "content": "好的。",
            },
        },
    }

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            prompt,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    return path