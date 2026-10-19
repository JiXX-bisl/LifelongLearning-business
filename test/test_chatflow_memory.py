import pytest

from src.chatflow_agent.memory import BusinessExperienceStore


def make_result(index, *, verdict="pass", status="completed"):
    """仅用于存储测试的模拟业务结果，不调用模型。"""
    question = "你只能查询还是也能修改数据？"
    answer = "我只支持只读查询，不会修改数据库。"

    return {
        "session_id": "offline-demo",
        "turn_index": index,
        "record_path": f"offline-only/{index:04d}.json",
        "baseline_id": "baseline-v1",
        "reference_date": "2026-08-31",
        "question": question,
        "answer": answer,
        "status": status,
        "operations": [],
        "evaluation": {
            "verdict": verdict,
            "reason": "离线测试指定的评价结果。",
        },
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "agent",
                "content": "Action: Answer\nFinal Answer:\n" + answer,
            },
        ],
    }


@pytest.mark.parametrize("mode", ["psu", "gsc"])
def test_memory_lifecycle(tmp_path, mode):
    def open_store(
        selected_mode=mode,
        experiment_id="q05_test",
        baseline_id="baseline-v1",
    ):
        return BusinessExperienceStore(
            tmp_path,
            baseline_id=baseline_id,
            learning={
                "mode": selected_mode,
                "experiment_id": experiment_id,
            },
        )

    store = open_store()
    assert store.version == 0
    assert not store.memory_path.exists()

    # 第一个合格任务入库，取得任务快照。
    assert store.observe(make_result(1), context_message_count=0)
    snapshot = store.snapshot()
    assert snapshot.version == 1
    assert snapshot.sessions[0].sample_index == "offline-demo:0001"

    # 第二个任务入库，不改变已经取得的快照。
    assert store.observe(make_result(2), context_message_count=0)
    assert store.version == 2
    assert snapshot.version == 1
    assert len(snapshot.sessions) == 1

    # 同一经验ID不重复收录。
    assert not store.observe(make_result(2), context_message_count=0)
    assert store.version == 2

    # 不合格结果不入库。
    for status, verdict in [
        ("completed", "revise"),
        ("completed", "unknown"),
        ("failed", "pass"),
    ]:
        assert not store.observe(
            make_result(3, status=status, verdict=verdict),
            context_message_count=0,
        )
    assert store.version == 2

    # 重新创建对象，模拟程序重启后加载。
    reopened = open_store()
    assert reopened.version == 2
    assert [
        session.model_dump(mode="json")
        for session in reopened.snapshot().sessions
    ] == [
        session.model_dump(mode="json")
        for session in store.snapshot().sessions
    ]

    # 不同实验、基线、模式均从各自空间读取。
    assert open_store(experiment_id="another_run").version == 0
    assert open_store(baseline_id="baseline-v2").version == 0
    other_mode = "gsc" if mode == "psu" else "psu"
    assert open_store(selected_mode=other_mode).version == 0

    # Standard不收录、不生成经验文件。
    standard = open_store(selected_mode="standard")
    assert not standard.observe(make_result(1), context_message_count=0)
    assert standard.snapshot().sessions == ()
    assert not standard.memory_path.exists()