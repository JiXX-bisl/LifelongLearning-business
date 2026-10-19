from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from src.project_paths import resolve_project_path

from pydantic import BaseModel

DBValue = int | float | str


class AnswerType(StrEnum):
    """Two evaluation ways for DBBench"""
    DIRECT = "direct"
    MD5 = "md5"

class AnswerInfo(BaseModel):
    """Standard answer and according SQL"""
    direct: list[list[DBValue]] | None
    md5: str | None
    sql: str

    def get_answer_type(self) -> AnswerType:
        if self.direct is not None:
            return AnswerType.DIRECT

        return AnswerType.MD5

class ColumnInfo(BaseModel):
    """A column of database"""
    name: str
    type: str

class TableInfo(BaseModel):
    """The database table structure and initial data used in a Case."""

    name: str
    column_info_list: list[ColumnInfo]
    row_list: list[list[DBValue]]

class DBBenchEntry(BaseModel):
    """A full DBBench Case"""
    instruction: str
    answer_info: AnswerInfo
    table_info: TableInfo
    skill_list: list[str]
    sql_instruction_row_list_entry_hash: int

    @property
    def database_name(self) -> str:
        return self.table_info.name
    
    def build_task_instruction(self) -> str:
        """Construct the specific task description that is ultimately sent to the Agent"""
        column_names = ", ".join(
            column.name
            for column in self.table_info.column_info_list
        )

        table_description = (
            f"The name of this table is {self.table_info.name}, "
            f"and the headers of this table are {column_names}."
        )

        return f"{self.instruction}\n{table_description}"
    
def load_dbbench_dataset(
        data_file_path: str | Path,
) -> dict[str, DBBenchEntry]:
    """read entry_dict.json, transfer into DBBenchEnty"""
    data_file_path = resolve_project_path(data_file_path)
    with data_file_path.open("r", encoding="utf-8") as file:
        raw_dataset = json.load(file)

    return {
        str(sample_index): DBBenchEntry.model_validate(entry)
        for sample_index, entry in raw_dataset.items()
    }