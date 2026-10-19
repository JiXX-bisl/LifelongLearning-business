from unittest.mock import Mock

import pytest

from src.chatflow_agent.memory import BusinessMemorySnapshot
from src.chatflow_agent.strategy import BusinessQueryStrategy
from src.lifelong_learning.core.models import (
    ChatHistory,
    EvaluationOutcome,
    EvaluationRecord,
    Role,
    SampleStatus,
    Session,
    TaskName,
)
from src.lifelong_learning.language_models.base import LanguageModel
from src.chatflow_agent.strategy import normalize_business_action
from src.lifelong_learning.agents.gsc_candidate_generator import GSCCandidate
from src.lifelong_learning.agents.gsc_voting import vote_gsc_candidates
from src.lifelong_learning.core.models import ChatMessage
from src.lifelong_learning.agents.base import Agent
from src.lifelong_learning.agents.gsc_decision import GSCDecisionEngine
from src.lifelong_learning.memory.group_self_consistency import (
    GroupSelfConsistencyMemory,
)

def make_session(index):
    history = ChatHistory()
    history.append(Role.USER, "离线案例说明。")
    history.append(Role.AGENT, "明白。")
    history.append(Role.USER, f"历史问题{index}")
    history.append(
        Role.AGENT,
        f"Action: Answer\nFinal Answer:\n历史回答{index}",
    )

    return Session(
        task_name=TaskName.BUSINESS_QUERY,
        sample_index=f"case-{index}",
        sample_status=SampleStatus.COMPLETED,
        chat_history=history,
        evaluation_record=EvaluationRecord(
            outcome=EvaluationOutcome.CORRECT,
        ),
    )


@pytest.mark.parametrize("mode", ["standard", "psu"])
def test_strategy_uses_experience_without_changing_history(mode):
    model = Mock(spec=LanguageModel)
    model.generate.return_value = (
        "Action: Answer\nFinal Answer:\n这是当前回复。"
    )

    snapshot = BusinessMemorySnapshot(
        version=5,
        sessions=tuple(make_session(index) for index in range(1, 6)),
    )
    strategy = BusinessQueryStrategy(
        language_model=model,
        system_prompt="业务查询规则。",
        learning={"mode": mode},
        snapshot=snapshot,
    )

    history = ChatHistory()
    history.append(Role.USER, "本次问题")
    original = history.model_dump(mode="json")

    for round_index in (1, 2):
        response = strategy.decide(
            history,
            task_id="offline-task",
            round_index=round_index,
        )
        assert response == model.generate.return_value
        assert history.model_dump(mode="json") == original

        sent_history = model.generate.call_args.kwargs["chat_history"]
        sent_text = "\n".join(
            message.content for message in sent_history.messages
        )
        assert sent_history.messages[-1].content == "本次问题"

        if mode == "standard":
            assert strategy.memory_version == 0
            assert strategy.used_experience_ids == []
            assert sent_history.model_dump(mode="json") == original
        else:
            assert strategy.memory_version == 5
            assert strategy.used_experience_ids == [
                "case-2", "case-3", "case-4", "case-5",
            ]
            assert "历史问题1" not in sent_text
            for index in range(2, 6):
                assert sent_text.count(f"历史问题{index}") == 1

    assert model.generate.call_count == 2

@pytest.mark.parametrize(
    "responses, expected_selected, expected_invalid",
    [
        (
            [
                (
                    "Action: Operation\n"
                    'Scope: {"enterprise": "全部企业", "time_range": null}\n'
                    "```sql\n"
                    "SELECT COUNT(*) FROM enterprise_basic;\n"
                    "```"
                ),
                (
                    "Action: Operation\n"
                    "```sql\n"
                    "SELECT COUNT(*) FROM enterprise_basic;\n"
                    "```"
                ),
                (
                    "Action: Operation\n"
                    'Scope: {"enterprise": 42, "time_range": null}\n'
                    "```sql\n"
                    "SELECT COUNT(*) FROM enterprise_basic;\n"
                    "```"
                ),
            ],
            0,
            [2],
        ),
        (
            [
                "Action: Answer\nFinal Answer:\n企业数量如下：\n40家。",
                "Action: Answer\nFinal Answer:\n企业数量如下：\n41家。",
                "Action: Answer\nFinal Answer:\n企业数量如下：\n41家。",
            ],
            1,
            [],
        ),
    ],
)
def test_business_gsc_voting(
    responses,
    expected_selected,
    expected_invalid,
):
    candidates = [
        GSCCandidate(
            candidate_index=index,
            experience_group_index=None if index == 0 else index - 1,
            experience_sample_index_list=[],
            agent_response=ChatMessage(
                role=Role.AGENT,
                content=response,
            ),
        )
        for index, response in enumerate(responses)
    ]

    vote = vote_gsc_candidates(
        candidates,
        action_normalizer=normalize_business_action,
    )

    assert vote.maximum_vote_count == 2
    assert vote.selected_candidate_index == expected_selected
    assert vote.invalid_candidate_index_list == expected_invalid

    # 投票只计算比较键，不改写原始候选。
    assert [
        candidate.agent_response.content
        for candidate in candidates
    ] == responses

    if expected_invalid:
        assert len(vote.action_to_candidate_index_list) == 1
    else:
        # 多行答案中40与41不同，必须形成两个动作。
        assert len(vote.action_to_candidate_index_list) == 2

def test_business_gsc_engine_selects_original_candidate(capsys):
    memory = GroupSelfConsistencyMemory(
        group_count=2,
        sample_count_per_group=2,
    )
    for index in range(1, 5):
        assert memory.observe(make_session(index))

    # 三个不同动作形成平票，仅测试选择流程，不执行这些SQL。
    responses = [
        (
            "Action: Operation\n"
            'Scope: {"enterprise": null, "time_range": null}\n'
            f"```sql\nSELECT {number};\n```"
        )
        for number in (1, 2, 3)
    ]

    candidate_agent = Mock(spec=Agent)
    candidate_agent.inference.side_effect = [
        ChatMessage(role=Role.AGENT, content=response)
        for response in responses
    ]

    selector_agent = Mock(spec=Agent)
    selector_agent.inference.return_value = ChatMessage(
        role=Role.AGENT,
        content="Selection: B",
    )

    normalizer = Mock(wraps=normalize_business_action)

    engine = GSCDecisionEngine(
        agent=candidate_agent,
        memory=memory,
        action_normalizer=normalizer,
        selector_agent=selector_agent,
        verbose=False,
    )

    history = ChatHistory()
    history.append(Role.USER, "业务查询规则。")
    history.append(Role.AGENT, "明白。")
    history.append(Role.USER, "请决定下一步操作。")
    original = history.model_dump(mode="json")

    response, record = engine.decide(
        sample_index="offline-task",
        round_index=1,
        current_chat_history=history,
    )

    assert [
        group.experience_sample_index_list
        for group in record.group_prompt_list
    ] == [
        ["case-4", "case-3"],
        ["case-2", "case-1"],
    ]

    assert candidate_agent.inference.call_count == 3
    assert normalizer.call_count == 3
    selector_agent.inference.assert_called_once()

    # 沿用原规则：平票先移除基础动作。
    # 剩下候选1和2，Selection: B选中候选2。
    assert record.selection_result.removed_base_action is True
    assert record.selection_result.selected_candidate_index == 2
    assert record.selection_result.selected_reason == "selected_by_llm"
    assert record.logical_inference_call_count == 4

    # 返回原始候选，Scope没有在归一化过程中丢失。
    assert response.content == responses[2]
    assert history.model_dump(mode="json") == original
    assert capsys.readouterr().out == ""