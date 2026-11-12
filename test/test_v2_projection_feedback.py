"""Projected result evaluation must reach the real history/pool/index chain."""
import json
import sqlite3
from contextlib import closing

import pytest

from src.psu_memory.models import RetrievalRequest
from src.sql_learning.comparison import ComparisonPolicy, compare_results
from src.sql_learning.query import SQLiteExecutor
from test_v2_feedback import chain, record


def _execute_turn(turn_id, question, sql):
    """Use actual SQLite output, including all auxiliary columns, as the source."""
    with closing(sqlite3.connect(":memory:")) as connection:
        executor = SQLiteExecutor(connection)
        try:
            actual = executor.execute_query(sql)
        finally:
            executor.close()
    return record(turn_id, question=question, sql=sql, columns=actual["columns"],
                  rows=actual["rows"], is_complete=actual["is_complete"])


def _retrieve(retriever, turn):
    return retriever.retrieve(RetrievalRequest(
        task_id="later_" + turn["turn_id"], question=turn["question"],
        namespace=turn["namespace"], data_version=turn["data_version"]))


@pytest.mark.parametrize("turn_id,question,sql,expected", [
    ("texas_area", "What is the area of Texas?",
     "SELECT 'texas' AS state_name, 266807.0 AS area",
     {"columns": ["area"], "rows": [[266807.0]]}),
    ("phoenix_city", "What is the largest city in Arizona?",
     "SELECT 'phoenix' AS city_name, 789704 AS population",
     {"columns": ["city_name"], "rows": [["phoenix"]]}),
])
def test_extra_columns_pass_then_enter_pool_and_are_retrieved(
        chain, turn_id, question, sql, expected):
    feedback, retriever = chain
    turn = _execute_turn(turn_id, question, sql)
    comparison = compare_results(turn, expected)
    assert comparison.status == "PASS"
    assert comparison.version == "row_projection_v2"

    assert not feedback.submit(turn)["admitted"]
    assert not _retrieve(retriever, turn).items
    assert feedback.evaluate(turn_id, comparison.status,
                             policy_version=comparison.version)["admitted"]
    result = _retrieve(retriever, turn)
    assert len(result.items) == 1, result.to_dict()
    assert result.items[0].item_id == turn_id
    assert sql in result.items[0].content
    assert retriever.index.inspect(namespace=turn["namespace"],
                                   data_version=turn["data_version"])["complete"]

    # Projection is evaluator-side only. Stored evidence must still bind the
    # full actually executed SQL/result, not a synthetic projected operation.
    with closing(sqlite3.connect(feedback.path)) as db:
        historical = json.loads(db.execute(
            "SELECT record_json FROM history_records WHERE execution_id=?", (turn_id,)
        ).fetchone()[0])
        saved_policy = db.execute(
            "SELECT verdict, policy_version FROM sql_learning_evaluations WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
    operation = json.loads(historical["task"]["evidence_text"])[0]
    assert operation["sql"] == sql
    assert operation["columns"] == turn["columns"]
    assert operation["rows"] == turn["rows"]
    assert saved_policy == ("PASS", comparison.version)


def test_projected_columns_cannot_mix_values_from_different_rows(chain):
    feedback, retriever = chain
    turn = _execute_turn(
        "wrong_association", "Return revenue for A and B.",
        "SELECT 'A' AS company, 200 AS revenue, 'extra' AS note "
        "UNION ALL SELECT 'B', 100, 'extra'")
    expected = {"columns": ["company", "revenue"], "rows": [["A", 100], ["B", 200]]}
    comparison = compare_results(turn, expected)
    assert comparison.status == "FAIL"
    feedback.submit(turn)
    assert not feedback.evaluate(turn["turn_id"], comparison.status,
                                 policy_version=comparison.version)["admitted"]
    assert not _retrieve(retriever, turn).items
    assert not feedback.pool.load_active_members(namespace=turn["namespace"],
                                                 data_version=turn["data_version"])


@pytest.mark.parametrize("numeric_strings,verdict", [(False, "FAIL"), (True, "PASS")])
def test_projection_keeps_numeric_conversion_explicit(chain, numeric_strings, verdict):
    feedback, retriever = chain
    turn = _execute_turn("amount_text", "Return the registered capital of A.",
                         "SELECT 'A' AS company, '980000.00' AS amount")
    comparison = compare_results(
        turn, {"columns": ["amount"], "rows": [[980000]]},
        policy=ComparisonPolicy(numeric_strings=numeric_strings))
    assert comparison.status == verdict
    feedback.submit(turn)
    result = feedback.evaluate(turn["turn_id"], comparison.status,
                               policy_version=comparison.version)
    assert result["admitted"] is (verdict == "PASS")
    assert bool(_retrieve(retriever, turn).items) is (verdict == "PASS")


def test_three_condition_runner_preserves_projected_pass_and_original_evidence(tmp_path):
    """Real runner execution proves the new comparison feeds online learning."""
    from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
    from src.psu_memory.retrieval.service import PSURetriever
    from src.sql_learning.experiment import run_experiment
    from test_v2_experiment import ScriptedModel
    from test_v2_feedback import Encoder, Selector

    sql = "SELECT state_name, area FROM state WHERE state_name = 'texas'"

    class ProjectionModel(ScriptedModel):
        def generate(self, chat_history, system_prompt=""):
            if chat_history.messages[-1].content.startswith("工具返回:"):
                return super().generate(chat_history, system_prompt)
            return "Action: Operation\n```sql\n" + sql + "\n```"

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with closing(sqlite3.connect(bundle / "database.sqlite3")) as db, db:
        db.execute("CREATE TABLE state (state_name TEXT, area REAL)")
        db.execute("INSERT INTO state VALUES ('texas', 266807)")
    scope = {"namespace": "projection_experiment", "data_version": "v2"}
    cases = [{"id": "first", "question": "What is the area of Texas?"},
             {"id": "later", "question": "Return the area of Texas."}]
    (bundle / "public.json").write_text(json.dumps({
        **scope, "schema": "CREATE TABLE state (state_name TEXT, area REAL)", "cases": cases}))
    (bundle / "private.json").write_text(json.dumps({
        case["id"]: {"columns": ["area"], "rows": [[266807.0]]} for case in cases}))
    config = RetrievalConfig(embedding_dimension=2, device="cpu")
    output = tmp_path / "run"
    summary = run_experiment(
        bundle, output, model_factory=ProjectionModel,
        retriever_factory=lambda path: PSURetriever.from_local(
            path, retrieval_config=config, encoder=Encoder(config), client_factory=Selector),
        retrieval_config=config, model_settings={"fixture": "projected_result_protocol"},
        validation_kind="protocol_replay")
    saved_summary = json.loads((output / "summary.json").read_text())
    assert summary["config"]["comparison_version"] == "row_projection_v2"
    assert saved_summary["config"]["comparison_version"] == "row_projection_v2"

    for condition in ("no_experience", "fixed", "continual"):
        counts = summary["conditions"][condition]
        assert counts["PASS"] == 2 and counts["FAIL"] == counts["UNKNOWN"] == 0
        assert counts["active_members"] == (2 if condition == "continual" else 0)
        turns = [json.loads((output / condition / f"{i:05d}.json").read_text()) for i in range(2)]
        assert [result["case_id"] for result in turns] == [case["id"] for case in cases]
        for result in turns:
            turn = result["turn"]
            assert result["comparison"]["version"] == "row_projection_v2"
            assert result["comparison"]["column_map"] == [1]
            assert turn["sql"] == sql
            assert turn["columns"] == ["state_name", "area"]
            assert turn["rows"] == [["texas", 266807.0]]
            assert turn["trace"]["operations"][0]["columns"] == turn["columns"]
            assert turn["trace"]["operations"][0]["rows"] == turn["rows"]
            if condition == "continual":
                assert result["update"]["admitted"] is True
            else:
                assert result["update"] is None

        with closing(sqlite3.connect(output / condition / "history.sqlite3")) as db:
            histories = [json.loads(row[0]) for row in db.execute(
                "SELECT record_json FROM history_records")]
        assert len(histories) == (2 if condition == "continual" else 0)
        for historical in histories:
            actual_operation = json.loads(historical["task"]["evidence_text"])[0]
            assert actual_operation["sql"] == sql
            assert actual_operation["columns"] == ["state_name", "area"]
            assert actual_operation["rows"] == [["texas", 266807.0]]
        if condition == "continual":
            assert turns[1]["turn"]["trace"]["retrieval"]["used_experience_ids"] == [turns[0]["turn"]["turn_id"]]
        else:
            retrieval = turns[1]["turn"]["trace"].get("retrieval", {})
            assert retrieval.get("used_experience_ids", []) == []
