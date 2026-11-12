"""Real frozen-data interface smoke, explicitly not model-effect validation."""
import hashlib
import json
from pathlib import Path

import pytest

from src.psu_memory.pool.classification import _sql_group
from src.sql_learning.comparison import ComparisonPolicy, compare_results
from src.sql_learning.enterprise import prepare_enterprise
from src.sql_learning.query import Scene, SQLiteExecutor, SqlQueryService


def test_frozen_enterprise_values_and_four_action_groups(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/sql/synthetic_enterprise_v30_120_v1.0.2/tables.json"
    if not source.is_file():
        pytest.skip("install the frozen enterprise input bundle to run this integration check")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    bundle = prepare_enterprise(source, tmp_path / "independent-bundle")
    public = json.loads((bundle / "public.json").read_text())
    private = json.loads((bundle / "private.json").read_text())
    references = json.loads((bundle / "private_sql.json").read_text())
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert provenance["full_benchmark"] is False
    assert provenance["sqlite_value_roundtrip_verified"] is True
    assert provenance["source_sha256"] == source_sha
    assert provenance["row_counts"]["enterprise_basic"] == 120
    assert len(provenance["row_counts"]) == 7
    assert set(_sql_group(sql)[0] for sql in references.values()) == {"detail", "aggregate", "compare", "rank"}
    assert all(set(case) == {"id", "question"} for case in public["cases"])
    assert all(set(expected) == {"columns", "rows"} for expected in private.values())
    assert all(sql not in json.dumps(public, ensure_ascii=False) for sql in references.values())

    class Replay:
        def __init__(self, sql):
            self.sql = sql
            self.calls = 0

        def generate(self, chat_history, system_prompt):
            self.calls += 1
            assert self.sql not in system_prompt  # Reference never enters the generation prompt.
            if self.calls == 1:
                return "Action: Operation\n```sql\n" + self.sql + "\n```"
            operation = json.loads(chat_history.messages[-1].content.split("\n", 1)[1])
            return "Action: Answer\nResult Query ID: " + operation["query_id"] + "\nFinal Answer:\n已查询。"

    for case in public["cases"]:
        with SqlQueryService(
            Scene(SQLiteExecutor(bundle / "database.sqlite3"), public["schema"], public["namespace"],
                  public["data_version"], public["default_rules"]), Replay(references[case["id"]]),
        ) as service:
            turn = service.query(case["question"])
            actual = {"columns": turn.columns, "rows": turn.rows, "is_complete": turn.is_complete}
            assert turn.execution_success and turn.status == "completed"
            assert compare_results(actual, private[case["id"]], policy=ComparisonPolicy(ordered=True)).status == "PASS"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    with pytest.raises(FileExistsError):
        prepare_enterprise(source, bundle)
