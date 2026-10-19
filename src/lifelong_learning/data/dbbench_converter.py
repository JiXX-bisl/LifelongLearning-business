from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DATASET_ID = "csyq/LifelongAgentBench"
DATASET_REVISION = "75054b60177d4dcddb93b984413ff799b0a1fdbc"
DBBENCH_PARQUET_FILE = "db_bench/train-00000-of-00001.parquet"


def convert_dbbench_dataset(output_path: str | Path) -> Path:
    """下载 DBBench Parquet，并转换为entry_dict.json。"""

    parquet_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=DBBENCH_PARQUET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )

    table = pq.read_table(parquet_path)
    entry_dict: dict[str, dict[str, Any]] = {}

    for row in table.to_pylist():
        sample_index = str(row["sample_index"])

        entry_dict[sample_index] = {
            "instruction": row["instruction"],
            "answer_info": ast.literal_eval(row["answer_info"]),
            "table_info": ast.literal_eval(row["table_info"]),
            "skill_list": ast.literal_eval(row["skill_list"]),
            "sql_instruction_row_list_entry_hash": row[
                "sql_instruction_row_list_entry_hash"
            ],
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(entry_dict, file, ensure_ascii=False, indent=2)

    return output_path