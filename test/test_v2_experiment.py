import json
import sqlite3
from contextlib import closing

import pytest

from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.service import PSURetriever
from src.sql_learning.experiment import run_experiment, load_comparison_policy
from src.sql_learning.comparison import compare_results
from test_v2_feedback import Encoder, Selector


class ScriptedModel:
    """Protocol fixture only; never used to measure LLM accuracy."""
    def generate(self, chat_history, system_prompt=""):
        message = chat_history.messages[-1].content
        if message.startswith("工具返回:"):
            operation = json.loads(message.split("\n", 1)[1])
            return f"Action: Answer\nResult Query ID: {operation['query_id']}\nFinal Answer:\nDone"
        return "Action: Operation\n```sql\nSELECT COUNT(*) FROM album\n```"


def test_evaluator_side_policy_keeps_rank_order_and_column_mapping(tmp_path):
    (tmp_path / "comparison_policy.json").write_text(json.dumps({"ordered": True, "column_map": [1, 0]}))
    policy = load_comparison_policy(tmp_path)
    assert policy.column_map == (1, 0)
    expected = {"columns": ["name", "rank"], "rows": [["A", 1], ["B", 2]]}
    actual = {"columns": ["rank", "name"], "rows": [[2, "B"], [1, "A"]]}
    assert compare_results(actual, expected, policy=policy).status == "FAIL"


def test_three_conditions_are_isolated_and_only_continual_updates(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with closing(sqlite3.connect(bundle / "database.sqlite3")) as db, db:
        db.execute("CREATE TABLE album (title TEXT)")
        db.executemany("INSERT INTO album VALUES (?)", [("A",), ("B",)])
    cases = [{"id": str(i), "question": "How many albums?"} for i in range(3)]
    (bundle / "public.json").write_text(json.dumps({
        "namespace": "album", "data_version": "v2", "schema": "CREATE TABLE album(title TEXT)", "cases": cases}))
    (bundle / "private.json").write_text(json.dumps({
        "0": {"columns": ["count"], "rows": [[2]]},
        "1": {"columns": ["count"], "rows": [[2]]},
        "2": {"columns": ["count"], "rows": [[999]]}}))
    config = RetrievalConfig(embedding_dimension=2, device="cpu")
    kwargs = dict(model_factory=ScriptedModel,
                  retriever_factory=lambda path: PSURetriever.from_local(
                      path, retrieval_config=config, encoder=Encoder(config), client_factory=Selector),
                  retrieval_config=config, model_settings={"fixture": "scripted"},
                  validation_kind="protocol_replay")
    summary = run_experiment(bundle, tmp_path / "run", **kwargs)
    assert summary["config"]["initial_memory"] == "empty"
    assert summary["config"]["case_order"] == ["0", "1", "2"]
    for condition, result in summary["conditions"].items():
        assert result["PASS"] == 2
        assert result["FAIL"] == 1
        assert result["active_members"] == (2 if condition == "continual" else 0)
    second = json.loads((tmp_path / "run/continual/00001.json").read_text())
    assert "历史 SQL 方法参考" in json.dumps(second, ensure_ascii=False)
    for condition in ("fixed", "no_experience"):
        second = json.loads((tmp_path / f"run/{condition}/00001.json").read_text())
        assert "历史 SQL 方法参考" not in json.dumps(second, ensure_ascii=False)
    with pytest.raises(FileExistsError):
        run_experiment(bundle, tmp_path / "run", **kwargs)
    public = json.loads((bundle / "public.json").read_text())
    public["cases"][0]["reference_sql"] = "SELECT 2"
    (bundle / "public.json").write_text(json.dumps(public))
    with pytest.raises(ValueError, match="gold"):
        run_experiment(bundle, tmp_path / "leaked", **kwargs)


def test_nonempty_seed_is_identical_and_fixed_remains_frozen(tmp_path):
    """Real seeded stores isolate memory policy; scripted generation is not an accuracy benchmark."""
    import hashlib
    from types import SimpleNamespace

    from src.psu_memory.integration.bootstrap import initialize_empty_store
    from src.psu_memory.pool.pool_models import PoolConfig
    from src.sql_learning.feedback import TurnFeedback
    from test_v2_feedback import record

    scope = {"namespace": "album", "data_version": "v2"}
    config = RetrievalConfig(embedding_dimension=2, device="cpu")
    seed_path = tmp_path / "seed.sqlite3"
    initialize_empty_store(seed_path, **scope, retrieval_config=config,
                           pool_config=PoolConfig(), record_runs=False)
    seed_retriever = PSURetriever.from_local(
        seed_path, retrieval_config=config, encoder=Encoder(config), client_factory=Selector)
    try:
        feedback = TurnFeedback(seed_retriever, SimpleNamespace(**scope))
        seed = record("seed_album_count", **scope, rows=[[2]], answer="2")
        feedback.submit(seed)
        assert feedback.evaluate(seed["turn_id"], "PASS")["admitted"]
        assert seed_retriever.index.inspect(**scope)["complete"]
    finally:
        seed_retriever.close()
    frozen_seed = seed_path.read_bytes()

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    with closing(sqlite3.connect(bundle / "database.sqlite3")) as db, db:
        db.execute("CREATE TABLE album (title TEXT)")
        db.executemany("INSERT INTO album VALUES (?)", [("A",), ("B",)])
    case_order = ["first", "second", "failure"]
    (bundle / "public.json").write_text(json.dumps({
        **scope, "schema": "CREATE TABLE album(title TEXT)",
        "cases": [{"id": key, "question": "How many albums?"} for key in case_order]}))
    (bundle / "private.json").write_text(json.dumps({
        key: {"columns": ["count"], "rows": [[999 if key == "failure" else 2]]}
        for key in case_order}))
    initial_hashes = []

    def make_retriever(path):
        # Capture each work database before model/retriever initialization can write caches.
        initial_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        return PSURetriever.from_local(
            path, retrieval_config=config, encoder=Encoder(config), client_factory=Selector)

    output = tmp_path / "seeded_run"
    summary = run_experiment(
        bundle, output, model_factory=ScriptedModel, retriever_factory=make_retriever,
        retrieval_config=config, model_settings={"fixture": "scripted"}, seed_store=seed_path,
        max_context_chars=3000, validation_kind="protocol_replay")
    assert summary["config"]["initial_memory"] == "provided_snapshot"
    assert summary["config"]["case_order"] == case_order
    assert initial_hashes == [summary["config"]["initial_store_sha256"]] * 3
    assert seed_path.read_bytes() == frozen_seed
    for condition, result in summary["conditions"].items():
        assert result["PASS"] == 2 and result["FAIL"] == 1
        assert result["active_members"] == (3 if condition == "continual" else 1)
        turns = [json.loads((output / condition / f"{i:05d}.json").read_text()) for i in range(3)]
        assert [turn["case_id"] for turn in turns] == case_order
        augmented = "历史 SQL 方法参考" in json.dumps(turns[0]["turn"], ensure_ascii=False)
        assert augmented is (condition != "no_experience")
        if condition != "continual":
            with closing(sqlite3.connect(output / condition / "history.sqlite3")) as db:
                assert db.execute("SELECT execution_id FROM psu_pool_members").fetchall() == [(seed["turn_id"],)]
                assert db.execute("SELECT COUNT(*) FROM history_records").fetchone()[0] == 1
