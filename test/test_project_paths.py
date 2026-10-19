import json
from pathlib import Path

from src import project_paths
from src.chatflow_agent.feedback import save_feedback
from src.lifelong_learning.tasks.enterprise_query_bench.dataset import load_enterprise_query_dataset


def make_project(tmp_path, monkeypatch):
    root = tmp_path / "relocated-project"
    catalog = root / "data/catalog"
    catalog.mkdir(parents=True)
    old_root = tmp_path / "original-project"
    mapping = {
        "original_project_root": str(old_root),
        "files": {"legacy/duplicate.json": "data/sql/canonical.json"},
        "directories": {
            "legacy": "data/archive",
            "outputs/business/query": "data/logs/business/query",
            "outputs/business/query/sessions": "data/logs/business/query/legacy_unscoped/sessions",
        },
    }
    (catalog / "path_map.json").write_text(json.dumps(mapping))
    monkeypatch.setattr(project_paths, "PROJECT_ROOT", root)
    return root, old_root


def test_historical_absolute_path_works_after_project_relocation(tmp_path, monkeypatch):
    root, old_root = make_project(tmp_path, monkeypatch)
    target = root / "data/sql/canonical.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"unchanged":true}')
    # The exact duplicate alias must beat the broader archive directory alias.
    for original in ["legacy/duplicate.json", old_root / "legacy/duplicate.json", root / "legacy/duplicate.json"]:
        assert project_paths.resolve_project_path(original).read_bytes() == target.read_bytes()
    assert not (root / "legacy").exists()


def test_early_logs_and_future_outputs_use_different_directories(tmp_path, monkeypatch):
    root, _ = make_project(tmp_path, monkeypatch)
    early = project_paths.resolve_project_path("outputs/business/query/sessions/early/0001.json")
    new = project_paths.resolve_project_path("outputs/business/query/baseline/new/psu/sessions/id/0001.json")
    assert early == root / "data/logs/business/query/legacy_unscoped/sessions/early/0001.json"
    assert new == root / "data/logs/business/query/baseline/new/psu/sessions/id/0001.json"
    external = tmp_path / "unrelated/record.json"
    assert project_paths.resolve_project_path(external) == external
    assert project_paths.resolve_project_path(new) == new


def test_feedback_follows_migrated_record_without_rewriting_provenance(tmp_path, monkeypatch):
    root, old_root = make_project(tmp_path, monkeypatch)
    original = old_root / "outputs/business/query/base/experiment/psu/sessions/session/0001.json"
    result = {"record_path": str(original), "session_id": "session", "turn_index": 1, "baseline_id": "base", "experiment_id": "experiment", "mode": "psu"}
    target = project_paths.resolve_project_path(original)
    target.parent.mkdir(parents=True)
    payload = json.dumps(result).encode()
    target.write_bytes(payload)
    feedback = save_feedback(result, rating="helpful")
    assert feedback == root / "data/logs/business/query/base/experiment/psu/feedback/session/0001.json"
    assert json.loads(feedback.read_text())["source_record_path"] == str(original)
    assert target.read_bytes() == payload
    assert result["record_path"] == str(original)
    assert not old_root.exists()


def test_old_dataset_and_baseline_paths_load_the_migrated_frozen_package():
    dataset = load_enterprise_query_dataset(
        "data/enterprise_query_bench/dev/enterprise_query_bench_dev_v0.1.1/entry_dict_colloquial_instructions.json",
        "docs/虚拟企业数据集/baselines/synthetic_enterprise_v21_40_v1.0.0",
    )
    assert dataset.data_file_path.parent == project_paths.PROJECT_ROOT / "data/questions/enterprise_query_bench/enterprise_query_bench_dev_v0.1.1"
    assert dataset.baseline_dir == project_paths.PROJECT_ROOT / "data/sql/synthetic_enterprise_v21_40_v1.0.0"
