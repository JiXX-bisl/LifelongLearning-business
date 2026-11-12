"""Controlled v2 runs; public generation inputs are separate from evaluator gold."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from src.psu_memory.integration.bootstrap import initialize_empty_store
from src.psu_memory.pool.pool_models import PoolConfig
from src.sql_learning.comparison import COMPARISON_VERSION, ComparisonPolicy, compare_results
from src.sql_learning.feedback import TurnFeedback
from src.sql_learning.query import Scene, SQLiteExecutor, SqlQueryService

CONDITIONS = ("no_experience", "fixed", "continual")


def load_comparison_policy(bundle, policy_file=None):
    path = Path(policy_file) if policy_file else Path(bundle) / "comparison_policy.json"
    if not path.exists() and policy_file is None:
        return ComparisonPolicy()
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("column_map") is not None:
        values["column_map"] = tuple(values["column_map"])
    return ComparisonPolicy(**values)


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_store(source, target):
    with closing(sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)


def run_experiment(bundle, output, *, model_factory, retriever_factory, retrieval_config,
                   model_settings, seed_store=None, max_context_chars=3000,
                   limit=None, policy=None, validation_kind="model_experiment"):
    """Each condition starts from the same database and identical memory snapshot.

    Factories must create fresh stateless model clients with the recorded settings.
    SQL generation sees only public cases/schema; private.json is read solely here.
    With no seed store the fixed condition is explicitly an empty-memory control.
    """
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    public_path, gold_path = bundle / "public.json", bundle / "private.json"
    public, gold = json.loads(public_path.read_text()), json.loads(gold_path.read_text())
    cases = public["cases"]
    if limit is not None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be positive")
        cases = cases[:limit]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("cases must have unique IDs")
    if any(c["id"] not in gold for c in cases):
        raise ValueError("every test case requires evaluator-side gold")
    if set(public) - {"namespace", "data_version", "schema", "default_rules", "cases"}:
        raise ValueError("unexpected fields in public bundle")
    if any(set(c) - {"id", "question", "context", "constraints"} for c in cases):
        raise ValueError("public cases must not contain gold or task-specific SQL")
    if type(max_context_chars) is not int or max_context_chars < 0:
        raise ValueError("invalid memory budget")
    policy = policy or load_comparison_policy(bundle)
    output.mkdir(parents=True, exist_ok=False)
    scope = {k: public[k] for k in ("namespace", "data_version")}
    pool_config = PoolConfig()
    initial = output / "initial.sqlite3"
    if seed_store is None:
        initialize_empty_store(initial, **scope, retrieval_config=retrieval_config,
                               pool_config=pool_config, record_runs=False)
    else:
        _copy_store(seed_store, initial)
    database = output / "database.sqlite3"
    _copy_store(bundle / "database.sqlite3", database)
    metadata = {
        "implementation_version": "dev_v2.0", "comparison_version": COMPARISON_VERSION,
        "selection_version": "quality_without_difficulty_v2", "validation_kind": validation_kind,
        "model": model_settings, "conditions": list(CONDITIONS), "case_order": [c["id"] for c in cases],
        "initial_memory": "empty" if seed_store is None else "provided_snapshot",
        "initial_store_sha256": _sha(initial), "database_sha256": _sha(database),
        "public_sha256": _sha(public_path), "private_sha256": _sha(gold_path),
        "max_context_chars": max_context_chars, "pool_config": pool_config.to_dict(),
        "retrieval_config": asdict(retrieval_config),
        "execution_limits": {"max_total_rows": 500, "statement_timeout_seconds": 10.0},
        "comparison_policy": {"ordered": policy.ordered, "column_map": policy.column_map,
                              "absolute_tolerance": str(policy.absolute_tolerance),
                              "relative_tolerance": str(policy.relative_tolerance),
                              "numeric_strings": policy.numeric_strings}, **scope,
    }
    _write(output / "config.json", metadata)
    summaries = {}
    for condition in CONDITIONS:
        folder = output / condition
        folder.mkdir()
        store = folder / "history.sqlite3"
        _copy_store(initial, store)
        retriever = retriever_factory(store)
        scene = Scene(SQLiteExecutor(database), public["schema"], **scope,
                      default_rules=public.get("default_rules", ""))
        service = None
        results = []
        try:
            inspection = retriever.index.inspect(**scope)
            if not inspection["complete"]:
                raise ValueError("seed index is incompatible or incomplete")
            model = model_factory()
            service = SqlQueryService(scene, model,
                                      retriever=None if condition == "no_experience" else retriever,
                                      max_context_chars=max_context_chars)
            feedback = TurnFeedback(retriever, scene, pool_config=pool_config)
            for index, case in enumerate(cases):
                turn = service.query(case["question"], context=case.get("context", []),
                                     constraints=case.get("constraints", ""), include_trace=True)
                actual = {"columns": turn.columns, "rows": turn.rows,
                          "is_complete": turn.is_complete and turn.execution_success
                                         and turn.status == "completed"}
                comparison = compare_results(actual, gold[case["id"]], policy=policy)
                update = None
                if condition == "continual":
                    feedback.submit(turn)
                    update = feedback.evaluate(turn.turn_id, comparison.status,
                                               policy_version=comparison.version)
                result = {"case_id": case["id"], "turn": turn.to_dict(),
                          "comparison": comparison.to_dict(), "update": update}
                _write(folder / f"{index:05d}.json", result)
                results.append(result)
            counts = {s: sum(r["comparison"]["status"] == s for r in results)
                      for s in ("PASS", "FAIL", "UNKNOWN")}
            summaries[condition] = {"case_count": len(results), **counts,
                                    "pass_rate": counts["PASS"] / len(results),
                                    "active_members": len(retriever.index.pool_store.load_active_members(**scope))}
            _write(folder / "summary.json", summaries[condition])
        finally:
            if service is not None:
                service.close()
            else:
                scene.executor.close()
            retriever.close()
    summary = {"config": metadata, "conditions": summaries}
    _write(output / "summary.json", summary)
    return summary
