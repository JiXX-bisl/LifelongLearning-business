import json
from copy import deepcopy

from src.chatflow_agent.feedback import save_feedback


def test_feedback_update_does_not_modify_task_or_memory(tmp_path):
    mode_dir = tmp_path / "test-baseline" / "q07_test" / "psu"
    record_path = mode_dir / "sessions" / "session-test" / "0001.json"
    record_path.parent.mkdir(parents=True)

    result = {
        "session_id": "session-test",
        "turn_index": 1,
        "baseline_id": "test-baseline",
        "experiment_id": "q07_test",
        "mode": "psu",
        "record_path": str(record_path),
        "answer": "共有40家企业。",
        "evaluation": {"verdict": "pass"},
    }
    record_path.write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )

    memory_path = mode_dir / "memory.json"
    memory_path.write_text("[]", encoding="utf-8")

    original_result = deepcopy(result)
    original_record = record_path.read_bytes()
    original_memory = memory_path.read_bytes()

    first_path = save_feedback(result, rating="helpful")
    assert json.loads(first_path.read_text())["rating"] == "helpful"

    second_path = save_feedback(
        result,
        rating="problem",
        comment="  希望补充统计口径。  ",
    )

    assert second_path == first_path
    assert first_path == (
        mode_dir / "feedback" / "session-test" / "0001.json"
    )

    saved = json.loads(second_path.read_text(encoding="utf-8"))
    assert saved["rating"] == "problem"
    assert saved["comment"] == "希望补充统计口径。"
    assert saved["source_record_path"] == str(record_path)

    assert result == original_result
    assert record_path.read_bytes() == original_record
    assert memory_path.read_bytes() == original_memory