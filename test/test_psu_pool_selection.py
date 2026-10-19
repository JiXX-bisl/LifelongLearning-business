"""dev_v0.1 选择规则与 FIFO 持久化边界；由用户独立执行。"""
from dataclasses import replace
import json
import unittest

from src.psu_memory.pool.pool_models import ContentKey, PoolCandidate, PoolConfig, PoolScope, PoolSnapshot
from src.psu_memory.pool.selection import candidate_from_evaluations, has_sql_result_pass, plan_group_update, select_fifo_members
from src.psu_memory.evaluation.quality_policy import evaluate_candidate, original_content
from test.test_psu_active_members import ActiveMemberTests


def candidate(number, *, group="detail", level="L1", validation=3, complete=3, chars=100):
    return PoolCandidate(str(number), "v", "unit", "v1", "eligible", group, level,
                         validation, complete, chars, "test")


class PoolSelectionTests(unittest.TestCase):
    def plan(self, candidates, *, capacity=8, incumbents=()):
        return plan_group_update(
            snapshot=PoolSnapshot(PoolScope("unit", "v1", "detail"), tuple(incumbents)),
            candidates=candidates, current_versions={item.execution_id: item.content_version for item in candidates},
            capacity=capacity,
        )

    def test_adaptive_balances_difficulty_and_lends_missing_levels(self):
        values = [candidate(index, level=level) for index, level in enumerate(
            ["L1"] * 10 + ["L2"] * 4 + ["L3"] * 4 + ["L4"] * 4)]
        self.assertEqual(self.plan(values).difficulty_counts, {"L1": 2, "L2": 2, "L3": 2, "L4": 2})
        self.assertEqual(self.plan(values[:14]).difficulty_counts, {"L1": 4, "L2": 4, "L3": 0, "L4": 0})

    def test_adaptive_preserves_quality_order_then_incumbent(self):
        strongest = candidate("strong", validation=3, complete=3, chars=10)
        longer = candidate("longer", validation=3, complete=3, chars=20)
        incomplete = candidate("incomplete", validation=3, complete=2, chars=1)
        weaker = candidate("weaker", validation=2, complete=3, chars=1)
        self.assertEqual(self.plan([weaker, incomplete, longer, strongest], capacity=4).selected,
                         (strongest.key, longer.key, incomplete.key, weaker.key))
        tie = replace(strongest, execution_id="aaa")
        self.assertEqual(self.plan([tie, strongest], capacity=1, incumbents=(strongest.key,)).selected,
                         (strongest.key,))

    def test_adaptive_old_candidate_can_reenter_after_current_member_invalidated(self):
        old, current = candidate("old", chars=20), candidate("current", chars=10)
        first = self.plan([old, current], capacity=1)
        self.assertEqual(first.selected, (current.key,))
        current = replace(current, admission_state="suspended")
        self.assertEqual(self.plan([old, current], capacity=1, incumbents=first.selected).selected, (old.key,))

    def test_scope_and_zero_capacity(self):
        value = candidate("x")
        self.assertEqual(self.plan([value], capacity=0).selected, ())
        with self.assertRaises(ValueError):
            self.plan([replace(value, namespace="other")])
        self.assertEqual(PoolConfig().total_capacity, 32)
        self.assertEqual(PoolConfig(strategy="fifo").total_capacity, 18)

    def test_fifo_keeps_insertion_order_ignores_quality_and_history(self):
        values = [candidate(index, chars=100 - index) for index in range(22)]
        heads = {item.execution_id: item.content_version for item in values}
        first = select_fifo_members(incumbents=(), candidates=values,
                                   new_keys=tuple(item.key for item in values[:20]), current_versions=heads,
                                   namespace="unit", data_version="v1", capacity=18)
        self.assertEqual(first, tuple(item.key for item in values[2:20]))
        second = select_fifo_members(incumbents=first, candidates=values, new_keys=(values[20].key,),
                                    current_versions=heads, namespace="unit", data_version="v1", capacity=18)
        self.assertEqual(second, tuple(item.key for item in values[3:21]))
        self.assertNotIn(values[21].key, second)


class SQLPassGateTests(unittest.TestCase):
    def decision(self, policy, *, review_query_id="q1"):
        operation = {"kind": "sql", "query_id": "q1", "sql": "SELECT 1",
                     "success": True, "is_complete": True, "columns": ["value"], "rows": [[1]]}
        record = {"source_sha256": "source", "task": {
            "task_id": "case", "question": "查询当前数值", "status": "completed",
            "evaluation_verdict": "pass", "evidence_text": json.dumps([operation]),
        }}
        details = {"legacy_b1": {"query_id": "q1"}}
        if policy is not None:
            details["independent_review"] = {
                "admission_policy": policy, "scope": "sql", "verdict": "pass", "query_id": review_query_id,
            }
        profile = {"reference_result_match": True, "evidence": [{
            "signals": {"reference_result_match": True}, "details": details,
        }]}
        return evaluate_candidate(record, profile, original_content(record, profile), source={"status": "verified"})

    def test_legacy_reference_and_legacy_judge_pass_cannot_enter_current_pool(self):
        for policy in (None, "sql_result_only_v1"):
            with self.subTest(policy=policy):
                decision = self.decision(policy)
                self.assertEqual(decision.admission_state, "eligible")
                self.assertTrue(decision.execution_verified)
                self.assertIn("reference_result_match", decision.basis)
                self.assertFalse(decision.direct_sql_result_verified)
                self.assertFalse(has_sql_result_pass(decision))
                projected = candidate_from_evaluations(decision, None, None, namespace="unit", data_version="v1")
                self.assertEqual(projected.admission_state, "suspended")

    def test_current_direct_pass_is_admitted_without_changing_score(self):
        direct = self.decision("sql_result_direct_v1")
        legacy = self.decision("sql_result_only_v1")
        self.assertTrue(direct.direct_sql_result_verified)
        self.assertTrue(has_sql_result_pass(direct))
        self.assertEqual(direct.validation_level, legacy.validation_level)
        self.assertEqual(direct.completeness, legacy.completeness)
        projected = candidate_from_evaluations(direct, None, None, namespace="unit", data_version="v1")
        self.assertEqual(projected.admission_state, "eligible")

    def test_direct_policy_label_cannot_replace_current_query_binding(self):
        decision = self.decision("sql_result_direct_v1", review_query_id="another-query")
        self.assertFalse(decision.direct_sql_result_verified)
        self.assertFalse(has_sql_result_pass(decision))


class FIFOPersistenceTests(ActiveMemberTests):
    def test_fifo_eighteen_and_repeated_batch_does_not_resurrect(self):
        config = PoolConfig(strategy="fifo", fifo_capacity=18)
        self.pool.configure(namespace="unit", data_version="v1", config=config)
        keys = [self._add_sample(f"读取测试企业 {index} 的记录。") for index in range(20)]
        self.pool.sync(namespace="unit", data_version="v1", execution_ids=tuple(keys))
        actual = [item["candidate"].execution_id for item in self._members()]
        self.assertEqual(actual, keys[-18:])
        self.pool.sync(namespace="unit", data_version="v1", execution_ids=tuple(keys))
        self.assertEqual([item["candidate"].execution_id for item in self._members()], actual)
        self.pool.sync(namespace="unit", data_version="v1")
        self.assertEqual([item["candidate"].execution_id for item in self._members()], actual)

    def test_fifo_current_annotation_refresh_keeps_position(self):
        self.pool.configure(namespace="unit", data_version="v1", config=PoolConfig(strategy="fifo"))
        before = [item["candidate"].key for item in self._members()]
        entry = self._members()[0]
        assessment = replace(entry["difficulty"].assessment, cache_key="new-difficulty-fixture", level="L2")
        self.pool.put(assessment)
        self.pool.save_annotation(entry, grouping=entry["grouping"],
                                  difficulty=replace(entry["difficulty"], assessment=assessment))
        self.pool.sync(namespace="unit", data_version="v1", execution_ids=(entry["identity"]["execution_id"],))
        self.assertEqual([item["candidate"].key for item in self._members()], before)

    def test_fifo_snapshot_configuration_preserves_members_without_backfill(self):
        extra = self._add_sample("尚未入池的候选")
        before = {item["candidate"].key for item in self._members()}
        self.pool.configure(namespace="unit", data_version="v1", config=PoolConfig(strategy="fifo"))
        self.assertEqual({item["candidate"].key for item in self._members()}, before)
        self.assertNotIn(extra, {item["candidate"].execution_id for item in self._members()})
        with self.assertRaises(ValueError):
            self.pool.configure(namespace="unit", data_version="v1", config=PoolConfig(strategy="fifo", fifo_capacity=1))


if __name__ == "__main__":
    unittest.main()
