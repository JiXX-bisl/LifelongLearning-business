from __future__ import annotations

import json
from pathlib import Path


# 复现参考：
# 3rd/LifelongAgentBench/src/factories/chat_history_item/offline/
# task_requirement.py
DBBENCH_TASK_REQUIREMENT = """I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.

To do operation, the format should be like this:
Action: Operation
```sql
SELECT * FROM table WHERE condition;
```
You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.
I will use "Action: Operation" literally to match your SQL.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If the SQL is not executed successfully, the response will be the error message.
Otherwise, the response will be the raw MySQL response.
For SELECT queries, the response will be the result of the query, such as [(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...], where each tuple represents a row and the elements are the values of the columns in the row.
For SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully.

If you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:
Action: Answer
Final Answer: [(1, 'John Doe', 'HR'), (2, 'Jane Smith', 'IT'), ...]
DO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If the question is about querying the database, then after done operation, your answer field should be the result of the query.
We note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
Once you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not."""


STANDARD_SUFFIX = """

Now, I will give you the question that you need to solve."""


PSU_SUFFIX = """

{previous_sample_utilization_target_position}

Now, I will give you the question that you need to solve."""

def write_dbbench_prompt(
    output_path: str | Path,
    use_previous_samples: bool,
) -> Path:
    """生成根目录框架使用的 DBBench 初始对话文件。"""
    suffix = PSU_SUFFIX if use_previous_samples else STANDARD_SUFFIX

    chat_history_item_dict = {
        "value": {
            "0": {
                "role": "user",
                "content": f"{DBBENCH_TASK_REQUIREMENT}{suffix}",
            },
            "1": {
                "role": "agent",
                "content": "OK.",
            },
        }
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            chat_history_item_dict,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path