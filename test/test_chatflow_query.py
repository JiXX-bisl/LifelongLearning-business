import pytest
import json

from pathlib import Path
from unittest.mock import Mock, patch

from src.chatflow_agent.query import QueryService
from src.chatflow_agent.memory import (
    BusinessExperienceStore,
    BusinessMemorySnapshot,
)
from src.chatflow_agent.runtime import parse_learning_config


def count_query_result():
    return {
        "columns": ["enterprise_count"],
        "rows": [[40]],
        "row_count": 1,
        "is_complete": True,
        "incomplete_reason": None,
    }

def configure_standard_learning(service):
    service.learning = parse_learning_config({})
    service.experience_store = Mock()
    service.experience_store.snapshot.return_value = BusinessMemorySnapshot(
        version=0,
        sessions=(),
    )
    service.experience_store.observe.return_value = False

def test_query_stops_after_answer():
    # 仅在测试中跳过初始化，避免预检、真实数据库和模型连接。
    service = QueryService.__new__(QueryService)
    configure_standard_learning(service)
    service.reset()
    service.system_prompt = "测试提示词"
    service.data_context = {}
    service._save_result = Mock()
    database = Mock()
    database.execute_query.return_value = count_query_result()
    service.environment = database

    model = Mock()
    model.generate.side_effect = [
        (
            "Action: Operation\n"
            "```sql\n"
            "SELECT COUNT(*) FROM enterprise_basic;\n"
            "```"
        ),
        (
            "Action: Answer\n"
            "Final Answer:\n"
            "数据库中一共有40家企业。"
        ),
    ]
    service.language_model = model

    try:
        result = service.query("数据库里一共有多少家企业？")

        assert result["status"] == "completed"
        assert result["answer"] == "数据库中一共有40家企业。"
        assert result["rounds"] == 2
        assert result["result_query_id"] is None
        assert not {"evaluation", "evaluation_calls", "repair_count", "attempts"} & result.keys()

        assert len(result["operations"]) == 1
        assert result["operations"][0]["rows"] == [[40]]

        operation = result["operations"][0]
        assert len(operation["query_id"]) == 32
        assert operation["columns"] == ["enterprise_count"]
        assert operation["row_count"] == 1
        assert operation["is_complete"] is True

        # 后续对话中保留同一个查询ID。
        assert any(
            operation["query_id"] in message.content
            for message in service._history.messages
        )

        database.execute_query.assert_called_once_with(
            "SELECT COUNT(*) FROM enterprise_basic;"
        )
        assert model.generate.call_count == 2
    finally:
        service.close()

    database.close.assert_called_once()

def test_followup_context_and_reset():
    service = QueryService.__new__(QueryService)
    configure_standard_learning(service)    
    service.reset()
    service.system_prompt = "测试提示词"
    service.data_context = {}
    service._save_result = Mock()
    database = Mock()
    database.execute_query.return_value = count_query_result()
    service.environment = database

    first_question = "数据库里一共有多少家企业？"
    followup = "这个数量指的是企业还是人员？"

    responses = iter([
        (
            "Action: Operation\n"
            "```sql\n"
            "SELECT COUNT(*) FROM enterprise_basic;\n"
            "```"
        ),
        (
            "Action: Answer\n"
            "Final Answer:\n"
            "工商基本信息表中共有40家企业。"
        ),
        (
            "Action: Answer\n"
            "Final Answer:\n"
            "这里的40指企业数量，不是人员数量。"
        ),
        (
            "Action: Answer\n"
            "Final Answer:\n"
            "请问你指的是哪个数量？"
        ),
    ])

    # 在每次模型调用发生时，记录它实际收到的上下文。
    seen_contexts = []

    def respond(chat_history, system_prompt):
        seen_contexts.append([
            message.content
            for message in chat_history.messages
        ])
        return next(responses)

    model = Mock()
    model.generate.side_effect = respond
    service.language_model = model

    try:
        first = service.query(first_question)
        second = service.query(followup)

        assert first["status"] == "completed"
        assert second["status"] == "completed"

        # 前两次模型调用完成第一问，第三次调用处理追问。
        followup_context = seen_contexts[2]
        assert first_question in followup_context
        assert followup_context[-1] == followup
        assert any(
            '"rows": [[40]]' in content
            for content in followup_context
        )

        # 清空对话后，相同的追问不应再获得之前的上下文。
        service.reset()
        third = service.query(followup)

        assert third["status"] == "completed"
        assert seen_contexts[3] == [followup]

        # reset只清空对话，不关闭数据库。
        database.close.assert_not_called()
    finally:
        service.close()

    database.close.assert_called_once()

def test_cli_single_question(capsys):
    from scripts.run_business_query import main

    with patch(
        "scripts.run_business_query.QueryService"
    ) as service_class:
        service = service_class.return_value.__enter__.return_value
        service.query.return_value = {
            "status": "completed",
            "answer": "共有40家企业。",
            "operations": [{"sql": "SELECT COUNT(*)"}],
        }

        exit_code = main([
            "--mode", "psu",
            "--experiment-id", "q05_cli_test",
            "--question", "有多少家企业？",
        ])

        assert service_class.call_args.kwargs == {
            "mode": "psu",
            "experiment_id": "q05_cli_test",
        }

        assert exit_code == 0
        service.query.assert_called_once_with("有多少家企业？")
        service_class.return_value.__exit__.assert_called_once()

    # 用户只看到回复，不看到SQL。
    assert capsys.readouterr().out == "共有40家企业。\n"


def test_cli_interactive_commands(capsys):
    from scripts.run_business_query import main

    with patch(
        "scripts.run_business_query.QueryService"
    ) as service_class, patch(
        "builtins.input",
        side_effect=[
            "",
            "有多少家企业？",
            "这个数量指什么？",
            "/new",
            "/exit",
        ],
    ):
        service = service_class.return_value.__enter__.return_value
        service.query.return_value = {
            "status": "completed",
            "answer": "测试回复。",
        }

        # 先执行，才会产生QueryService的调用记录。
        exit_code = main(["--interactive"])

        # 再检查实际传入的参数。
        assert service_class.call_args.kwargs == {
            "mode": None,
            "experiment_id": None,
        }

        assert exit_code == 0
        questions = [
            call.args[0]
            for call in service.query.call_args_list
        ]
        assert questions == [
            "有多少家企业？",
            "这个数量指什么？",
        ]
        service.reset.assert_called_once()
        service_class.return_value.__exit__.assert_called_once()

    assert "已开始新对话。" in capsys.readouterr().out


def test_cli_runtime_failure(capsys):
    from scripts.run_business_query import main
    from src.chatflow_agent.runtime import BusinessRuntimeError

    with patch(
        "scripts.run_business_query.QueryService",
        side_effect=BusinessRuntimeError("数据库连接失败。"),
    ):
        exit_code = main(["--question", "有多少家企业？"])

    assert exit_code == 1
    error_output = capsys.readouterr().err
    assert "数据库连接失败" in error_output
    assert "Traceback" not in error_output

def make_query_service(responses):
    service = QueryService.__new__(QueryService)
    configure_standard_learning(service)    
    service.reset()
    service.system_prompt = "测试提示词"
    service.data_context = {}
    service._save_result = Mock()

    service.environment = Mock()
    service.environment.execute_query.return_value = count_query_result()

    service.language_model = Mock()
    service.language_model.generate.side_effect = responses
    return service


def test_query_returns_answer_without_internal_validation():
    service = make_query_service([
        (
            "Action: Operation\n"
            "```sql\n"
            "SELECT COUNT(*) FROM enterprise_basic;\n"
            "```"
        ),
        "Action: Answer\nFinal Answer:\n共有41家企业。",
    ])

    try:
        result = service.query("有多少家企业？")

        # 查询入口返回模型提交的回复，正确性由外部SQL验证决定。
        assert result["status"] == "completed"
        assert result["answer"] == "共有41家企业。"
        assert service.language_model.generate.call_count == 2
        assert not {"evaluation", "evaluation_calls", "repair_count", "attempts"} & result.keys()
        assert result["answer"] in service._history.messages[-1].content
    finally:
        service.close()


def test_clarification_ends_without_additional_model_call():
    clarification = "请问你要查询哪家企业？"
    service = make_query_service([
        "Action: Answer\nFinal Answer:\n" + clarification,
    ])

    try:
        result = service.query("它的法人是谁？")

        assert result["answer"] == clarification
        assert result["operations"] == []
        assert result["result_query_id"] is None
        assert service.language_model.generate.call_count == 1
    finally:
        service.close()


def test_query_preserves_explicit_result_binding():
    service = make_query_service([])

    def respond(chat_history, system_prompt):
        if len(chat_history.messages) == 1:
            return "Action: Operation\n```sql\nSELECT COUNT(*) FROM enterprise_basic;\n```"
        observation = json.loads(chat_history.messages[-1].content.split("\n", 1)[1])
        return (
            "Action: Answer\nResult Query ID: " + observation["query_id"]
            + "\nFinal Answer:\n共有40家企业。"
        )

    service.language_model.generate.side_effect = respond
    try:
        result = service.query("有多少家企业？")
        assert result["result_query_id"] == result["operations"][0]["query_id"]
        assert result["answer"] == "共有40家企业。"
        assert result["result_query_id"] in service._history.messages[-1].content
        assert service.language_model.generate.call_count == 2
    finally:
        service.close()


def test_validation_query_keeps_business_record_in_memory(tmp_path):
    from src.psu_memory.integration.local_query import LocalValidationQueryService

    service = LocalValidationQueryService.__new__(LocalValidationQueryService)
    configure_standard_learning(service)
    service.reset()
    service.retriever = None
    service.case_id, service.run_id, service.condition = "case1", "run1", "baseline"
    service.config = {"output_dir": str(tmp_path), "data": {"baseline_id": "test-baseline"}}
    service.data_context = {"reference_date": "2026-08-31", "schema_version": "3.0"}
    service.system_prompt = "测试提示词"
    service.environment = Mock()
    service.language_model = Mock()
    service.language_model.generate.return_value = (
        "Action: Answer\nResult Query ID: null\nFinal Answer:\n请提供企业名称。"
    )
    try:
        result = service.query("查一下这家企业。")
        assert result["turn_index"] == 1
        assert result["session_id"] == service.session_id
        assert result["baseline_id"] == "test-baseline"
        assert result["validation_case_id"] == "case1"
        assert result["validation_condition"] == "baseline"
        assert "record_path" not in result
        assert "messages" in result
        assert "psu" not in result
        assert list(tmp_path.iterdir()) == []
    finally:
        service.close()


def test_round_limit_stops_query():
    service = make_query_service(["格式不正确"] * 12)

    try:
        result = service.query("有多少家企业？")

        assert result["status"] == "limit_reached"
        assert service.language_model.generate.call_count == 12
        assert service._history.messages == []
    finally:
        service.close()

def make_storage_service(output_dir):
    service = QueryService.__new__(QueryService)
    configure_standard_learning(service)

    service.reset()
    service.config = {
        "output_dir": str(output_dir),
        "data": {"baseline_id": "test-baseline"},
        "model": {"model_name": "test-model"},
        # 测试用哨兵值：确认不会把整个配置写入记录。
        "private_test_value": "do-not-save-this-secret",
    }
    service.data_context = {"reference_date": "2026-08-31"}
    return service


def test_save_records_and_start_new_session(tmp_path):
    service = make_storage_service(tmp_path)

    def make_result():
        return {
            "question": "有多少家企业？",
            "status": "completed",
            "answer": "共有40家企业。",
            "operations": [
                {
                    "kind": "sql",
                    "sql": "SELECT COUNT(*) FROM enterprise_basic;",
                    "success": True,
                    "rows": [[40]],
                }
            ],
            "messages": [],
        }

    first = make_result()
    service._save_result(first)
    first_path = Path(first["record_path"])
    first_text = first_path.read_text(encoding="utf-8")
    saved = json.loads(first_text)

    assert saved["answer"] == first["answer"]
    assert saved["operations"][0]["rows"] == [[40]]
    assert not {"evaluation", "evaluation_calls", "repair_count", "attempts"} & saved.keys()
    assert saved["baseline_id"] == "test-baseline"
    assert "do-not-save-this-secret" not in first_text
    assert first_path.name == "0001.json"

    second = make_result()
    service._save_result(second)

    assert second["session_id"] == first["session_id"]
    assert second["turn_index"] == 2
    assert Path(second["record_path"]).name == "0002.json"

    service.reset()
    third = make_result()
    service._save_result(third)

    assert third["session_id"] != first["session_id"]
    assert third["turn_index"] == 1
    assert Path(third["record_path"]).parent != first_path.parent

    # 新提问和新会话都没有覆盖第一次的记录。
    assert first_path.read_text(encoding="utf-8") == first_text


def test_save_failure_is_reported(tmp_path):
    from src.chatflow_agent.runtime import BusinessRuntimeError

    # 用文件占据目录位置，稳定模拟无法创建输出目录。
    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("occupied", encoding="utf-8")

    service = make_storage_service(blocked_path)
    result = {"question": "测试问题", "answer": "测试回复"}

    with pytest.raises(
        BusinessRuntimeError,
        match="执行记录保存失败",
    ):
        service._save_result(result)

    assert service._turn_index == 0
    assert "record_path" not in result


def test_database_failure_preserves_trace_and_stops():
    import mysql.connector

    service = make_query_service([
        (
            "Action: Operation\n"
            "```sql\n"
            "SELECT COUNT(*) FROM enterprise_basic;\n"
            "```"
        ),
        (
            "Action: Operation\n"
            "```sql\n"
            "SELECT enterprise_id FROM enterprise_basic;\n"
            "```"
        ),
    ])

    service.environment.execute_query.side_effect = [
        count_query_result(),
        mysql.connector.OperationalError(
            msg="模拟异常中的敏感信息",
            errno=2013,
        ),
    ]

    try:
        result = service.query("查询企业数量和企业ID。")

        assert result["status"] == "failed"
        assert result["rounds"] == 2
        assert service.language_model.generate.call_count == 2

        # 已成功的查询和随后失败的查询都得到保留。
        assert len(result["operations"]) == 2
        first, failed = result["operations"]
        assert first["success"] is True
        assert failed["success"] is False
        assert failed["fatal"] is True
        assert failed["is_complete"] is None
        assert first["query_id"] != failed["query_id"]

        assert "模拟异常中的敏感信息" not in json.dumps(
            result,
            ensure_ascii=False,
        )
        service._save_result.assert_called_once_with(result=result)
        assert service._history.messages == []
    finally:
        service.close()

@pytest.mark.parametrize(
    "scope_line, expected_scope",
    [
        (
            "",
            {
                "enterprise": None,
                "time_range": None,
                "source": "not_provided",
            },
        ),
        (
            'Scope: {"enterprise": "全部企业", "time_range": null}\n',
            {
                "enterprise": "全部企业",
                "time_range": None,
                "source": "agent_declared",
            },
        ),
    ],
)
def test_scope_reaches_history_and_file(
    tmp_path,
    scope_line,
    expected_scope,
):
    # 使用真实保存方法，但模型和数据库仍是替身。
    service = make_storage_service(tmp_path)
    service.system_prompt = "测试提示词"

    service.environment = Mock()
    service.environment.execute_query.return_value = count_query_result()

    service.language_model = Mock()
    service.language_model.generate.side_effect = [
        (
            "Action: Operation\n"
            + scope_line
            + "```sql\n"
            "SELECT COUNT(*) FROM enterprise_basic;\n"
            "```"
        ),
        "Action: Answer\nFinal Answer:\n共有40家企业。",
    ]

    try:
        result = service.query("有多少家企业？")

        operation = result["operations"][0]
        assert operation["scope"] == expected_scope

        tool_message = next(
            message.content
            for message in service._history.messages
            if message.content.startswith("工具返回")
        )
        remembered_operation = json.loads(
            tool_message.split("\n", 1)[1]
        )
        assert remembered_operation["scope"] == expected_scope
        assert (
            remembered_operation["query_id"]
            == operation["query_id"]
        )

        saved = json.loads(
            Path(result["record_path"]).read_text(encoding="utf-8")
        )
        assert saved["operations"][0] == operation
    finally:
        service.close()


def test_scope_rejects_invalid_value_type():
    from src.chatflow_agent.protocol import parse_business_response
    from src.lifelong_learning.core.models import AgentAction

    parsed = parse_business_response(
        "Action: Operation\n"
        'Scope: {"enterprise": 42, "time_range": null}\n'
        "```sql\n"
        "SELECT 1;\n"
        "```"
    )

    assert parsed.action == AgentAction.INVALID
    assert "enterprise" in parsed.finish_reason

@pytest.mark.parametrize("mode", ["psu", "gsc"])
def test_learning_query_uses_fixed_snapshot_without_auto_admission(
    tmp_path, mode
):
    from src.lifelong_learning.core.models import (
        ChatHistory, EvaluationOutcome, EvaluationRecord,
        Role, SampleStatus, Session, TaskName,
    )

    service = make_storage_service(tmp_path)
    service.learning = parse_learning_config({"mode": mode})
    service.experience_store = BusinessExperienceStore(
        tmp_path,
        baseline_id="test-baseline",
        learning=service.learning,
    )
    service.config["output_dir"] = str(
        service.experience_store.directory
    )
    service.system_prompt = "测试提示词"

    service.environment = Mock()
    service.environment.execute_query.return_value = count_query_result()

    operation = (
        "Action: Operation\n```sql\n"
        "SELECT COUNT(*) FROM enterprise_basic;\n```"
    )
    correct_answer = "Action: Answer\nFinal Answer:\n共有40家企业。"
    seed_question = "数据库里一共有多少家企业？"
    seed_id = "externally-verified-legacy-seed"
    seed_history = ChatHistory()
    seed_history.append(Role.USER, "历史案例仅供参考查询方法。")
    seed_history.append(Role.AGENT, "明白。")
    seed_history.append(Role.USER, seed_question)
    seed_history.append(Role.AGENT, operation)
    seed_history.append(Role.USER, '工具返回：{"rows": [[40]]}')
    seed_history.append(Role.AGENT, correct_answer)
    # 固定一条已有且经过外部核验的legacy快照；不由本次回复生成成功标记。
    service.experience_store._sessions = [Session(
        task_name=TaskName.BUSINESS_QUERY,
        sample_index=seed_id,
        sample_status=SampleStatus.COMPLETED,
        chat_history=seed_history,
        evaluation_record=EvaluationRecord(outcome=EvaluationOutcome.CORRECT),
    )]

    # GSC每轮生成基础候选和经验候选，两个候选提交相同动作。
    candidate_count = 2 if mode == "gsc" else 1
    task_responses = [
        response
        for response in (operation, correct_answer)
        for _ in range(candidate_count)
    ]
    responses = iter(task_responses * 2)
    seen_versions = []

    def respond(chat_history, system_prompt):
        seen_versions.append(service.experience_store.version)
        return next(responses)

    service.language_model = Mock()
    service.language_model.generate.side_effect = respond

    try:
        with patch.object(
            service.experience_store,
            "snapshot",
            wraps=service.experience_store.snapshot,
        ) as snapshot_method:
            first = service.query("先统计一下企业总数。")
            snapshot_method.assert_called_once()
            service.reset()
            snapshot_method.reset_mock()
            second = service.query("请重新统计企业总数。")
            snapshot_method.assert_called_once()

        for result in (first, second):
            assert result["memory_version"] == 1
            assert result["used_experience_ids"] == [seed_id]
            assert "evaluation" not in result
        assert service.experience_store.version == 1
        assert not service.experience_store.memory_path.exists()
        assert seen_versions == [1] * (4 * candidate_count)

        # 两个任务各执行一次SQL，不因生成多个候选而重复执行。
        assert service.environment.execute_query.call_count == 2

        # 第二个任务实际收到历史经验。
        # GSC第二个任务的第一个调用是基础候选，不带经验；
        # 下一个调用才是经验组候选。
        experience_call_index = 2 * candidate_count + (1 if mode == "gsc" else 0)
        second_input = (
            service.language_model.generate.call_args_list[
                experience_call_index
            ].kwargs["chat_history"]
        )
        assert any(
            seed_question in message.content
            for message in second_input.messages
        )

        # 但历史经验没有混入实际轨迹或后续对话。
        assert seed_question not in json.dumps(
            second["messages"], ensure_ascii=False
        )
        assert all(
            seed_question not in message.content
            for message in service._history.messages
        )

        saved = json.loads(
            Path(second["record_path"]).read_text(encoding="utf-8")
        )
        assert saved == second

        if mode == "gsc":
            assert all(
                len(record["candidate_list"]) == 2
                for record in first["gsc_inference_records"]
            )

            records = second["gsc_inference_records"]
            assert len(records) == 2
            assert [
                record["round_index"] for record in records
            ] == [1, 2]

            for record in records:
                assert len(record["candidate_list"]) == 2
                assert record["logical_inference_call_count"] == 2
                assert record["group_prompt_list"][0][
                    "experience_sample_index_list"
                ] == [seed_id]
    finally:
        service.close()

def make_telemetry_model(responses):
    """模拟已具备统计回调的模型，不发起网络请求。"""
    model = Mock()
    response_iterator = iter(responses)
    model.observer_owners = []

    def respond(chat_history, system_prompt):
        response = next(response_iterator)
        failed = isinstance(response, Exception)

        observer = model.call_observer
        model.observer_owners.append(observer.__self__)

        observer(
            success=not failed,
            request_count=1,
            retry_count=0,
            elapsed_seconds=0.125,
            usage=(
                None
                if failed
                else {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            ),
            error_type=type(response).__name__ if failed else None,
        )

        if failed:
            raise response
        return response

    model.generate.side_effect = respond
    return model

def test_query_telemetry_is_task_scoped_and_saved(tmp_path):
    service = make_storage_service(tmp_path)
    service.system_prompt = "测试提示词"
    service.environment = Mock()
    service.environment.execute_query.return_value = count_query_result()

    operation = (
        "Action: Operation\n```sql\n"
        "SELECT COUNT(*) FROM enterprise_basic;\n```"
    )
    service.language_model = make_telemetry_model([
        operation,
        "Action: Answer\nFinal Answer:\n共有40家企业。",
        "Action: Answer\nFinal Answer:\n这里指企业数量。",
    ])

    try:
        progress_messages = []
        first = service.query(
            "数据库里有多少家企业？",
            on_progress=progress_messages.append,
        )

        assert progress_messages == [
            "正在分析问题，第1轮...",
            "正在查询数据...",
            "正在分析问题，第2轮...",
            "正在保存本次记录...",
            "本次处理结束。",
        ]
        assert first["status"] == "completed"
        assert first["metrics"]["model_call_count"] == 2
        assert first["metrics"]["request_count"] == 2
        assert first["metrics"]["retry_count"] == 0
        assert first["metrics"]["model_elapsed_seconds"] == 0.25
        assert first["metrics"]["usage"]["total_tokens"] == 24
        assert [
            call["phase"] for call in first["model_calls"]
        ] == [
            "decision",
            "decision",
        ]
        assert service.language_model.call_observer is None

        second = service.query("这个数量指企业还是人员？")

        # 第二问的查询模型仍能看到第一问留下的实际查询证据。
        followup_input = (
            service.language_model.generate.call_args_list[-1]
            .kwargs["chat_history"]
        )
        previous_query_id = first["operations"][0]["query_id"]
        assert any(
            previous_query_id in message.content
            for message in followup_input.messages
        )
        assert second["status"] == "completed"
        assert second["metrics"]["model_call_count"] == 1
        assert second["metrics"]["request_count"] == 1
        assert second["metrics"]["usage"]["total_tokens"] == 12
        assert [
            call["call_index"] for call in second["model_calls"]
        ] == [1]
        assert [
            call["phase"] for call in second["model_calls"]
        ] == ["decision"]
        assert service.language_model.call_observer is None

        # 第一问的全部调用使用同一个容器。
        owners = service.language_model.observer_owners
        assert owners[0] is owners[1]

        # 第二问使用新容器，不能累计第一问的消耗。
        assert owners[2] is not owners[0]

        # 第二问执行后，第一问结果不被修改。
        assert first["metrics"]["model_call_count"] == 2

        for result in (first, second):
            saved = json.loads(
                Path(result["record_path"]).read_text(encoding="utf-8")
            )
            assert saved == result
    finally:
        service.close()

def test_model_failure_saves_trace_and_telemetry(tmp_path):
    service = make_storage_service(tmp_path)
    service.learning = parse_learning_config({"mode": "psu"})
    service.experience_store = BusinessExperienceStore(
        tmp_path,
        baseline_id="test-baseline",
        learning=service.learning,
    )
    service.config["output_dir"] = str(
        service.experience_store.directory
    )
    service.system_prompt = "测试提示词"
    service.environment = Mock()
    service.environment.execute_query.return_value = count_query_result()

    service.language_model = make_telemetry_model([
        (
            "Action: Operation\n```sql\n"
            "SELECT COUNT(*) FROM enterprise_basic;\n```"
        ),
        RuntimeError("sensitive-error-body"),
    ])

    try:
        result = service.query("数据库里有多少家企业？")

        assert result["status"] == "failed"
        assert not {"evaluation", "evaluation_calls", "repair_count", "attempts"} & result.keys()

        assert len(result["operations"]) == 1
        assert result["operations"][0]["success"] is True
        assert result["operations"][0]["rows"] == [[40]]

        assert result["metrics"]["model_call_count"] == 2
        assert result["metrics"]["failed_model_call_count"] == 1
        assert result["metrics"]["request_count"] == 2
        assert result["metrics"]["usage_complete"] is False
        assert result["model_calls"][-1]["error_type"] == "RuntimeError"

        assert service.language_model.call_observer is None
        assert service._history.messages == []
        assert service.experience_store.version == 0
        assert not service.experience_store.memory_path.exists()

        saved_text = Path(result["record_path"]).read_text(
            encoding="utf-8"
        )
        assert json.loads(saved_text) == result
        assert "sensitive-error-body" not in saved_text
    finally:
        service.close()

@pytest.mark.parametrize(
    "reference_date, expected_start, expected_end",
    [
        ("2026-08-31", "2026-06-01", "2026-08-31"),
        ("2026-08-15", "2026-05-01", "2026-07-31"),
        ("2026-01-15", "2025-10-01", "2025-12-31"),
        ("2024-02-29", "2023-12-01", "2024-02-29"),
    ],
)

def test_default_revenue_period(
    reference_date, expected_start, expected_end
):
    from src.chatflow_agent.protocol import default_revenue_period

    assert default_revenue_period(reference_date) == {
        "start_date": expected_start,
        "end_date": expected_end,
    }


def test_business_prompt_contains_default_revenue_period():
    from src.chatflow_agent.protocol import build_business_prompt

    prompt = build_business_prompt(
        database="test_database",
        mysql_version="8.0",
        reference_date="2026-08-31",
    )

    assert "2026-06-01至2026-08-31" in prompt
    assert "Action: Operation" in prompt
    assert "Action: Answer" in prompt


def test_business_prompt_includes_provided_schemas():
    from src.chatflow_agent.protocol import build_business_prompt

    schemas = {
        "enterprise_basic": (
            "enterprise_basic：工商基本信息\n"
            "- enterprise_id: varchar(32)；企业ID\n"
            "- enterprise_name: varchar(120)；企业名称"
        ),
        "enterprise_operation_monthly": (
            "enterprise_operation_monthly：月度经营数据\n"
            "- enterprise_id: varchar(32)；企业ID\n"
            "- stat_month: date；统计月份\n"
            "- operating_revenue: decimal(18,2)；营业收入，元"
        ),
    }
    original = dict(schemas)

    prompt = build_business_prompt(
        database="test_database",
        mysql_version="8.0",
        reference_date="2026-08-31",
        schemas=schemas,
    )

    for schema_text in schemas.values():
        assert prompt.count(schema_text) == 1

    assert "2026-06-01至2026-08-31" in prompt
    assert schemas == original
