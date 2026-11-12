"""Generic interface checks run against actual SQLite execution and the shared loop."""
import json
import sqlite3

import pytest

from src.sql_learning.query import QUERY_RULES, Scene, SqlQueryService, SQLiteExecutor
from src.psu_memory.models import RetrievalResult, UsageMetrics


class ScriptModel:
    def __init__(self, statements, *, submitted="actual"):
        self.statements = list(statements)
        self.submitted = submitted
        self.calls = []
        self.closed = False

    def generate(self, chat_history, system_prompt):
        self.calls.append((system_prompt, chat_history.model_dump(mode="json")))
        if self.statements:
            return "Action: Operation\n```sql\n" + self.statements.pop(0) + "\n```"
        operations = [json.loads(message.content.split("\n", 1)[1])
                      for message in chat_history.messages if message.content.startswith("工具返回:")]
        identifier = operations[-1].get("query_id", "null") if operations else "null"
        if self.submitted != "actual":
            identifier = self.submitted
        return f"Action: Answer\nResult Query ID: {identifier}\nFinal Answer:\nResult submitted."

    def close(self):
        self.closed = True


@pytest.fixture
def scene():
    connection = sqlite3.connect(":memory:")
    connection.executescript("CREATE TABLE books(title TEXT, pages INTEGER);"
                             "INSERT INTO books VALUES ('A', 10), ('B', 20), ('B', 20);")
    executor = SQLiteExecutor(connection)
    yield Scene(executor, {"books": "books(title TEXT, pages INTEGER)"}, "library", "fixture-v1")
    executor.close()
    connection.close()


def test_generic_scene_reuses_loop_and_returns_actual_submission(scene):
    model = ScriptModel(["SELECT title, pages FROM books ORDER BY title"])
    service = SqlQueryService(scene, model)
    context = [{"role": "user", "content": "Use the library."}]
    result = service.query("List books", context=context, constraints="Preserve duplicates")
    assert result.status == "completed"
    assert result.execution_success and result.is_complete
    assert result.sql == "SELECT title, pages FROM books ORDER BY title"
    assert result.columns == ["title", "pages"]
    assert result.rows == [["A", 10], ["B", 20], ["B", 20]]
    assert result.namespace == "library" and result.data_version == "fixture-v1"
    assert result.context == context
    assert "Constraints:\nPreserve duplicates" in model.calls[0][1]["messages"][-1]["content"]
    assert "trace" not in result.to_dict()
    assert "query_id" not in result.to_dict()
    assert service._run_query.__func__.__qualname__ == "QueryService._run_query"
    model.statements = ["SELECT COUNT(*) FROM books"]
    second = service.query("Count books")
    assert second.turn_id != result.turn_id
    assert second.context == []
    assert len(second.turn_id) == 32


def test_unknown_query_id_cannot_bind_to_actual_results(scene):
    result = SqlQueryService(scene, ScriptModel(["SELECT 1"], submitted="invented")).query("one")
    assert result.status == "failed"
    assert result.sql == "SELECT 1" and result.execution_success and result.is_complete
    assert result.rows == [[1]]


def test_failed_operation_cannot_be_submitted_but_can_be_retried(scene):
    result = SqlQueryService(scene, ScriptModel(["SELECT absent FROM books"])).query("one")
    assert result.status == "failed" and not result.execution_success
    result = SqlQueryService(scene, ScriptModel(["SELECT absent FROM books", "SELECT 1"])).query(
        "one", include_trace=True)
    assert result.execution_success and result.rows == [[1]]
    assert len(result.trace["operations"]) == 2
    assert result.trace["operations"][0]["success"] is False


def test_partial_result_remains_partial(scene):
    scene.executor.max_total_rows = 1
    result = SqlQueryService(scene, ScriptModel(["SELECT * FROM books"])).query("books")
    assert result.execution_success and not result.is_complete
    assert len(result.rows) == 1


def test_no_submission_is_not_success(scene):
    result = SqlQueryService(scene, ScriptModel([])).query("please clarify")
    assert result.status == "completed"
    assert result.sql is None and not result.execution_success


def test_public_prompt_has_no_enterprise_coupling(scene):
    prompt = scene.prompt()
    for forbidden in ("enterprise_basic", "Schema 3.0", "营收", "expected_rows", "reference_sql"):
        assert forbidden not in prompt
    assert prompt.startswith(QUERY_RULES)
    assert "books(title TEXT, pages INTEGER)" in prompt


def test_executor_read_only_boundary_and_sql_literals(scene):
    executor = scene.executor
    assert executor.execute_query("SELECT '; DELETE DROP' AS literal")["rows"] == [["; DELETE DROP"]]
    assert executor.execute_query("WITH x AS (SELECT 1 AS n) SELECT n FROM x")["rows"] == [[1]]
    for sql in ("DELETE FROM books", "WITH x AS (SELECT 1) DELETE FROM books",
                "SELECT 1; DELETE FROM books", "SELECT load_extension('something')"):
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            executor.execute_query(sql)
    assert executor.execute_query("SELECT COUNT(*) FROM books")["rows"] == [[3]]


def test_file_executor_does_not_create_missing_database(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        SQLiteExecutor(missing)
    assert not missing.exists()


def test_retrieval_uses_turn_id_and_existing_prompt_recheck(scene):
    class Retriever:
        def __init__(self):
            self.calls = []

        def enhance_prompt(self, original_prompt, request, *, retrieval_result=None):
            self.calls.append((request, retrieval_result))
            result = RetrievalResult(reason_code="disabled", usage=UsageMetrics())
            result.diagnostics = {"augmentation": {"usage": {}}}
            return original_prompt, result

    retriever = Retriever()
    result = SqlQueryService(scene, ScriptModel(["SELECT 1"]), retriever=retriever,
                             max_context_chars=100).query("one", constraints="exact", include_trace=True)
    assert len(retriever.calls) == 2
    request, cached = retriever.calls[0]
    assert request.task_id == result.turn_id
    assert request.namespace == scene.namespace and request.data_version == scene.data_version
    assert request.data_advice == "exact" and request.max_context_chars == 100
    assert cached is None and retriever.calls[1][1] is not None
    assert "retrieval" in result.trace


def test_invalid_executor_output_is_failed_not_silently_complete():
    class Broken:
        def execute_query(self, sql):
            return {"columns": ["x"], "rows": [[1, 2]], "is_complete": True}

    result = SqlQueryService(Scene(Broken(), "x", "scope", "v1"), ScriptModel(["SELECT 1"])).query("one")
    assert result.status == "failed" and not result.execution_success


def test_assistant_context_uses_standard_public_role(scene):
    context = [{"role": "assistant", "content": "The table is books."}]
    model = ScriptModel(["SELECT 1"])
    turn = SqlQueryService(scene, model).query("one", context=context)
    assert turn.context == context
    assert model.calls[0][1]["messages"][0]["role"] == "agent"


def test_null_submission_preserves_last_execution_without_allowing_success(scene):
    turn = SqlQueryService(scene, ScriptModel(["SELECT 1"], submitted="null")).query("one")
    assert turn.status == "failed"
    assert turn.sql == "SELECT 1" and turn.rows == [[1]]
    assert turn.execution_success and turn.is_complete


def test_failed_and_limit_turns_retain_last_sql_attempt(scene):
    class FailingModel(ScriptModel):
        def generate(self, chat_history, system_prompt):
            if not self.statements:
                raise RuntimeError("simulated model failure after execution")
            return super().generate(chat_history, system_prompt)

    turn = SqlQueryService(scene, FailingModel(["SELECT 1"])).query("one")
    assert turn.status == "failed" and turn.sql == "SELECT 1" and turn.rows == [[1]]
    turn = SqlQueryService(scene, ScriptModel(["SELECT 2"] * 12)).query("two")
    assert turn.status == "limit_reached" and turn.sql == "SELECT 2" and turn.rows == [[2]]
    turn = SqlQueryService(scene, ScriptModel(["SELECT absent FROM books"])).query("bad column")
    assert turn.status == "failed" and turn.sql == "SELECT absent FROM books"
    assert not turn.execution_success and not turn.is_complete and turn.rows == []


def test_text_schema_describe_returns_public_schema(scene):
    public_schema = "CREATE TABLE books(title TEXT, pages INTEGER);"
    service = SqlQueryService(Scene(scene.executor, public_schema, "library", "v2"),
                              ScriptModel(["DESCRIBE books", "SELECT COUNT(*) FROM books"]))
    turn = service.query("count books", include_trace=True)
    described = turn.trace["operations"][0]
    assert described["success"] and described["schema"] == public_schema
    assert described["schema_scope"] == "public_scene"
    assert turn.status == "completed" and turn.rows == [[3]]


def test_sqlite_query_deadline_and_cleanup(scene):
    executor = scene.executor
    executor.statement_timeout_seconds = 0.005
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        executor.execute_query("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n")
    # Handler and cursor are removed even when execute() aborts before returning.
    assert executor.execute_query("SELECT 1")["rows"] == [[1]]
    with pytest.raises(sqlite3.DatabaseError):
        executor.execute_query("SELECT missing FROM absent")
    assert executor.execute_query("SELECT 2")["rows"] == [[2]]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_executor_result_is_failed(value):
    class Broken:
        def execute_query(self, sql):
            return {"columns": ["x"], "rows": [[value]], "is_complete": True}

    turn = SqlQueryService(Scene(Broken(), "x", "s", "v"), ScriptModel(["SELECT 1"])).query("one")
    assert turn.status == "failed" and not turn.execution_success and not turn.is_complete
    assert turn.sql == "SELECT 1"
    json.dumps(turn.to_dict(), allow_nan=False)
