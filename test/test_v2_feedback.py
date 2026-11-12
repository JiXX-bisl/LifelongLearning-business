"""Full persistent feedback chain; fake models only, real SQL stores/index/retrieval."""
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.psu_memory.integration.bootstrap import initialize_empty_store
from src.psu_memory.integration.direct_update import DirectUpdateError
from src.psu_memory.models import RetrievalRequest
from src.psu_memory.pool.pool_models import PoolConfig, ModelReply, DifficultyUsage
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.service import PSURetriever
from src.sql_learning.feedback import TurnFeedback


class Encoder:
    def __init__(self, config):
        self.config = config

    def count_tokens(self, text, *, usage):
        return len(text)

    def encode(self, texts, *, usage):
        return [(1.0, 0.0) for _ in texts]


class Selector:
    signature = "deterministic-test-only"

    def generate(self, *, system_prompt, user_text):
        return ModelReply(json.dumps({"selected_id": "c1", "reason": "test"}), DifficultyUsage())

    def close(self):
        pass


@pytest.fixture
def chain(tmp_path):
    config = RetrievalConfig(embedding_dimension=2, device="cpu")
    path = tmp_path / "working.sqlite3"
    initialize_empty_store(path, namespace="different-domain", data_version="v2",
                           retrieval_config=config, pool_config=PoolConfig(), record_runs=False)
    retriever = PSURetriever.from_local(path, retrieval_config=config,
                                      encoder=Encoder(config), client_factory=Selector)
    scene = SimpleNamespace(namespace="different-domain", data_version="v2")
    feedback = TurnFeedback(retriever, scene)
    yield feedback, retriever
    retriever.close()


def record(turn_id="turn_a", **overrides):
    return {"turn_id": turn_id, "namespace": "different-domain", "data_version": "v2",
            "question": "How many albums?", "context": [], "constraints": "",
            "status": "completed", "sql": "SELECT COUNT(*) FROM album", "columns": ["count"],
            "rows": [[3]], "is_complete": True, "execution_success": True,
            "answer": "3", **overrides}


def retrieve(retriever):
    return retriever.retrieve(RetrievalRequest(task_id="later", question="How many albums?",
                                             namespace="different-domain", data_version="v2"))


def dump(path):
    with closing(sqlite3.connect(path)) as connection:
        return tuple(connection.iterdump())


def test_unknown_fail_pass_late_feedback_and_revocation(chain):
    feedback, retriever = chain
    assert not feedback.submit(record())["admitted"]
    assert not retrieve(retriever).items
    assert not feedback.evaluate("turn_a", "FAIL")["admitted"]
    assert not retrieve(retriever).items
    assert feedback.evaluate("turn_a", "PASS")["admitted"]
    result = retrieve(retriever)
    assert len(result.items) == 1, result.to_dict()
    assert result.items[0].item_id == "turn_a"
    assert "COUNT(*)" in result.items[0].content
    before = dump(feedback.path)
    assert feedback.evaluate("turn_a", "PASS")["admitted"]
    assert dump(feedback.path) == before
    assert not feedback.evaluate("turn_a", "UNKNOWN")["admitted"]
    assert not retrieve(retriever).items
    assert feedback.evaluate("turn_a", "PASS")["admitted"]
    assert not feedback.evaluate("turn_a", "FAIL")["admitted"]
    assert not retrieve(retriever).items


def test_failed_index_publication_rolls_back_and_retries(chain):
    feedback, retriever = chain
    feedback.submit(record())
    before = dump(feedback.path)
    with patch("src.sql_learning.feedback.OfflineIndexBuilder.build", side_effect=RuntimeError("unavailable")):
        with pytest.raises(DirectUpdateError):
            feedback.evaluate("turn_a", "PASS")
    assert dump(feedback.path) == before
    assert not retrieve(retriever).items
    assert feedback.evaluate("turn_a", "PASS")["admitted"]


def test_annotation_failure_rolls_back_and_retries(chain):
    feedback, retriever = chain
    feedback.submit(record())
    before = dump(feedback.path)
    with patch.object(feedback.pool, "save_annotation", side_effect=RuntimeError("write failure")):
        with pytest.raises(DirectUpdateError):
            feedback.evaluate("turn_a", "PASS")
    assert dump(feedback.path) == before
    assert feedback.evaluate("turn_a", "PASS")["admitted"]


def test_turn_binding_tampering_and_scope(chain):
    feedback, retriever = chain
    feedback.submit(record())
    feedback.evaluate("turn_a", "PASS")
    before = dump(feedback.path)
    with pytest.raises(DirectUpdateError):
        feedback.submit(record(rows=[[4]]))
    assert dump(feedback.path) == before
    with pytest.raises(ValueError, match="scope"):
        feedback.submit(record("other", namespace="different"))
    path = next(feedback.records.glob("*.json"))
    path.write_text("{}")
    with pytest.raises(DirectUpdateError):
        feedback.evaluate("turn_a", "PASS")
    assert not retrieve(retriever).items


@pytest.mark.parametrize("overrides", [{"is_complete": False}, {"status": "failed"},
                                      {"execution_success": False},
                                      {"sql": None, "execution_success": False, "columns": [], "rows": []}])
def test_incomplete_or_failed_cannot_pass(chain, overrides):
    feedback, retriever = chain
    feedback.submit(record(**overrides))
    with pytest.raises(DirectUpdateError):
        feedback.evaluate("turn_a", "PASS")
    assert not retrieve(retriever).items


def test_restart_accepts_late_evaluation(chain):
    feedback, retriever = chain
    feedback.submit(record())
    resumed = TurnFeedback(retriever, feedback.scene)
    assert resumed.evaluate("turn_a", "PASS")["admitted"]
    assert retrieve(retriever).items
