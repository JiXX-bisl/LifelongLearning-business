"""Difficulty is optional throughout maintenance, selection and indexed retrieval."""
from dataclasses import replace
import json
import shutil
import sqlite3
from types import SimpleNamespace

import pytest

from src.psu_memory.history.history_adapter import history_from_query_result
from src.psu_memory.integration.direct_update import DirectCandidateCollector
from src.psu_memory.models import RetrievalRequest
from src.psu_memory.pool.classification import classify
from src.psu_memory.pool.maintenance import run_pool_maintenance
from src.psu_memory.pool.pool_models import (
    DifficultyAssessment, DifficultyResult, DifficultyUsage, GroupingInput,
    PoolCandidate, PoolConfig, PoolScope, PoolSnapshot,
)
from src.psu_memory.pool.pool_store import SQLitePoolStore, POOL_SCHEMA_VERSION
from src.psu_memory.pool.selection import plan_group_update, select_fifo_members
from src.psu_memory.retrieval.embedding_service import EmbeddingService
from src.psu_memory.retrieval.index_store import SQLiteRetrievalIndex
from src.psu_memory.retrieval.indexing import OfflineIndexBuilder
from src.psu_memory.retrieval.prefilter import SemanticPrefilter
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.vector_cache import SQLiteVectorCache


SCOPE = {"namespace": "v2-tests", "data_version": "schema-a"}


class FakeEncoder:
    """Deterministic retrieval mechanics fixture; not an embedding quality benchmark."""
    def __init__(self, config):
        self.config = config

    def count_tokens(self, text, *, usage):
        return len(text)

    def encode(self, texts, *, usage):
        return [(1.0, 0.0) for _ in texts]


@pytest.fixture
def stored(tmp_path):
    path = tmp_path / "working.sqlite3"
    collector = DirectCandidateCollector(path, **SCOPE)
    case = {"question": "列出顾客姓名", "history": []}
    record = {"question": case["question"], "baseline_id": SCOPE["data_version"],
              "session_id": "session-a", "turn_index": 1, "status": "completed",
              "answer": "Ada", "operations": [{"kind": "sql", "query_id": "q1",
              "sql": "SELECT name FROM customers", "success": True,
              "is_complete": True, "columns": ["name"], "rows": [["Ada"]]}]}
    source = tmp_path / "result.json"
    source.write_text(json.dumps(record))
    saved = collector.apply(case, record, source, "q1")
    pool = SQLitePoolStore(path, record_runs=False)
    pool.initialize()
    config = RetrievalConfig(device="cpu", embedding_dimension=2)
    cache = SQLiteVectorCache(pool)
    index = SQLiteRetrievalIndex(cache, config)
    index.initialize()
    embeddings = EmbeddingService(config, cache, encoder=FakeEncoder(config))
    return SimpleNamespace(path=path, pool=pool, quality=collector.quality,
                           history=collector.history, case=case, record=record, source=source,
                           execution_id=saved["execution_id"], index=index, embeddings=embeddings)


def maintain(stored, **kwargs):
    return run_pool_maintenance(quality_store=stored.quality, pool_store=stored.pool,
                                **SCOPE, strict=True, refresh_quality=False, **kwargs)


def test_unlabeled_pass_is_stored_active_indexed_and_retrieved(stored):
    report = maintain(stored)
    assert report["error_count"] == 0
    assert report["usage"]["model_call_count"] == 0
    assert report["sync"]["selection_version"] == "quality_without_difficulty_v2"
    members = stored.pool.load_active_members(**SCOPE)
    assert len(members) == 1
    assert members[0]["candidate"].difficulty is None
    with stored.pool.connection() as db:
        assert db.execute("SELECT cache_key, assessment_sha256 FROM psu_pool_annotations").fetchone() == (None, None)
        assert db.execute("SELECT COUNT(*) FROM psu_pool_difficulty_versions").fetchone()[0] == 0
    assert OfflineIndexBuilder(stored.index, stored.embeddings).build(**SCOPE)["index_complete"]
    found = SemanticPrefilter(stored.index, stored.embeddings).prefilter(
        RetrievalRequest(task_id="later-turn", question="列出顾客姓名", **SCOPE))
    assert found.status == "ok"
    assert found.candidates[0]["member"]["candidate"].execution_id == stored.execution_id
    again = maintain(stored)
    assert again["preparation"]["changed_annotations"] == 0
    assert stored.index.inspect(**SCOPE)["complete"]


def test_failed_optional_label_does_not_block_pass(stored):
    def fail(*args, **kwargs):
        raise RuntimeError("labeler unavailable")
    report = maintain(stored, assessor=SimpleNamespace(cache=stored.pool, assess=fail))
    assert report["error_count"] == 0
    assert report["difficulty_errors"][0]["error_type"] == "RuntimeError"
    assert len(stored.pool.load_active_members(**SCOPE)) == 1


def test_unknown_quality_is_never_admitted(stored):
    case = {"question": "未核验问题", "history": []}
    record = {**stored.record, "question": case["question"], "session_id": "unknown"}
    path = stored.path.parent / "unknown.json"
    path.write_text(json.dumps(record))
    unknown = history_from_query_result(case, record, path, "q1", **SCOPE)
    stored.history.import_batch([unknown], **SCOPE, source_root=str(path.parent),
                                scanned_files=1, errors=[], started_at=0)
    report = run_pool_maintenance(quality_store=stored.quality, pool_store=stored.pool, **SCOPE)
    assert report["preparation"]["prepared_count"] == 1
    assert [m["identity"]["execution_id"] for m in stored.pool.load_active_members(**SCOPE)] == [stored.execution_id]


def candidate(key, *, difficulty=None, length=10, validation=3, complete=4):
    return PoolCandidate(execution_id=key, content_version="v1", **SCOPE,
                         admission_state="eligible", group_id="detail", difficulty=difficulty,
                         validation_level=validation, completeness_count=complete,
                         preview_chars=length, quality_policy_version="fixture")


def test_quality_then_shorter_content_not_difficulty_quota_or_recency():
    choices = [candidate("a", difficulty="L1", length=10),
               candidate("b", difficulty="L1", length=11),
               candidate("c", difficulty="L4", length=100),
               candidate("unlabeled", length=9)]
    snapshot = PoolSnapshot(PoolScope(**SCOPE, group_id="detail"))
    plan = plan_group_update(snapshot=snapshot, candidates=choices,
                             current_versions={c.execution_id: "v1" for c in choices}, capacity=3)
    assert [key.execution_id for key in plan.selected] == ["unlabeled", "a", "b"]
    assert plan.difficulty_counts["L4"] == 0
    incumbent = candidate("zz-original")
    newcomer = candidate("aa-new")
    tied = plan_group_update(snapshot=replace(snapshot, members=(incumbent.key,)),
                              candidates=[incumbent, newcomer],
                              current_versions={"zz-original": "v1", "aa-new": "v1"}, capacity=1)
    assert tied.selected == (incumbent.key,)
    assert select_fifo_members(incumbents=(), candidates=[incumbent], new_keys=(incumbent.key,),
                               current_versions={"zz-original": "v1"}, **SCOPE, capacity=1) == (incumbent.key,)


def test_labeled_v3_working_copy_migration_preserves_members_and_index(stored):
    # Materialize an actual v3 annotation and its index, then migrate a copy only.
    entry = stored.pool.load_inputs(**SCOPE)[0]
    difficulty = DifficultyResult(**entry["identity"], cache_hit=False, usage=DifficultyUsage(),
        assessment=DifficultyAssessment(cache_key="label-1", prompt_version="fixture",
            prompt_sha256="fixture", model_signature="fixture", status="assessed", level="L1",
            reason="fixture", missing_information=(), origin_usage=DifficultyUsage()))
    stored.pool.put(difficulty.assessment)
    grouping = classify(GroupingInput(**entry["identity"], question=stored.case["question"],
                                      sql=stored.record["operations"][0]["sql"]))
    stored.pool.save_annotation(entry, grouping=grouping, difficulty=difficulty)
    stored.pool.sync(**SCOPE)
    assert OfflineIndexBuilder(stored.index, stored.embeddings).build(**SCOPE)["index_complete"]
    with sqlite3.connect(stored.path) as db:
        schema = db.execute("SELECT sql FROM sqlite_master WHERE name='psu_pool_annotations'").fetchone()[0]
        schema = schema.replace('psu_pool_annotations (', 'old_annotations (', 1)
        schema = schema.replace('cache_key TEXT,', 'cache_key TEXT NOT NULL,').replace(
            'assessment_sha256 TEXT,', 'assessment_sha256 TEXT NOT NULL,')
        db.execute(schema)
        db.execute("INSERT INTO old_annotations SELECT * FROM psu_pool_annotations")
        db.execute("DROP TABLE psu_pool_annotations")
        db.execute("ALTER TABLE old_annotations RENAME TO psu_pool_annotations")
        db.execute("UPDATE psu_pool_meta SET schema_version=3")
    frozen_bytes = stored.path.read_bytes()
    copy = stored.path.parent / "v2-copy.sqlite3"
    shutil.copyfile(stored.path, copy)
    pool = SQLitePoolStore(copy)
    pool.initialize()
    assert stored.path.read_bytes() == frozen_bytes
    assert len(pool.load_active_members(**SCOPE)) == 1
    migrated_index = SQLiteRetrievalIndex(SQLiteVectorCache(pool), stored.index.config)
    assert migrated_index.inspect(**SCOPE)["complete"]
    with pool.connection() as db:
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("SELECT schema_version FROM psu_pool_meta").fetchone()[0] == POOL_SCHEMA_VERSION
    new_entry = pool.load_inputs(**SCOPE)[0]
    pool.save_annotation(new_entry, grouping=grouping)
    assert pool.load_active_members(**SCOPE)[0]["difficulty"] is None


def test_mutated_source_is_excluded_without_an_explicit_quality_refresh(stored):
    maintain(stored)
    OfflineIndexBuilder(stored.index, stored.embeddings).build(**SCOPE)
    stored.source.write_text("changed result")
    assert stored.pool.load_active_members(**SCOPE) == []
    assert stored.index.inspect(**SCOPE)["member_count"] == 0
    found = SemanticPrefilter(stored.index, stored.embeddings).prefilter(
        RetrievalRequest(task_id="later", question="列出顾客姓名", **SCOPE))
    assert not found.candidates


def test_missing_standard_turn_source_is_not_retrievable(stored):
    # Legacy imported logs allow missing historical files; standard turns keep
    # immutable framework-owned sources and must fail closed when one disappears.
    with stored.pool.connection(write=True) as db:
        raw = db.execute("SELECT record_json FROM history_records WHERE execution_id=?",
                         (stored.execution_id,)).fetchone()[0]
        historical = json.loads(raw)
        historical["mode"] = "sql_learning_v2"
        db.execute("UPDATE history_records SET record_json=? WHERE execution_id=?",
                   (json.dumps(historical), stored.execution_id))
    maintain(stored)
    OfflineIndexBuilder(stored.index, stored.embeddings).build(**SCOPE)
    assert stored.pool.load_active_members(**SCOPE)
    stored.source.unlink()
    assert stored.pool.load_active_members(**SCOPE) == []
    assert stored.index.inspect(**SCOPE)["member_count"] == 0
    found = SemanticPrefilter(stored.index, stored.embeddings).prefilter(
        RetrievalRequest(task_id="later", question="列出顾客姓名", **SCOPE))
    assert not found.candidates
