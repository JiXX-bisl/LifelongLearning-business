"""由用户独立运行：人工业务口径授权不能放宽其他标准目标或响应契约。"""
from __future__ import annotations

import unittest
from copy import deepcopy

from scripts.enterprise_query_bench.validate_v30_120_question_draft import (
    canonical_sha256, validate_goal_preservation,
)


class ReviewedGoalAuthorizationTest(unittest.TestCase):
    def setUp(self):
        self.old_facts = {
            "A4-L2-01": {"id": "A4-L2-01", "response_type": "answer",
                         "target_scope": "截至7月的最新收入",
                         "expected_result": {"columns": ["month", "income"], "rows": [["2026-07", "100.00"]]}},
            "A4-L2-02": {"id": "A4-L2-02", "response_type": "answer",
                         "expected_result": {"columns": ["income"], "rows": [["200.00"]]}},
        }
        self.old_rules = {"expected_types": {case_id: "answer" for case_id in self.old_facts},
                          "sql_requirements": {case_id: "required" for case_id in self.old_facts},
                          "references": {"A4-L2-01": {"sql": "SELECT income WHERE month <= '2026-07'",
                                                       "aliases": {"income": "income"}}}}
        self.facts = deepcopy(self.old_facts)
        self.facts["A4-L2-01"].update(target_scope="数据库中最新收入",
                                      expected_result={"columns": ["month", "income"],
                                                       "rows": [["2026-08", "120.00"]]},
                                      expected_intent="用户已确认删除截止日")
        self.rules = deepcopy(self.old_rules)
        self.rules["references"]["A4-L2-01"]["sql"] = "SELECT income ORDER BY month DESC LIMIT 1"
        self.approval = {"status": "HUMAN_REVIEW_APPROVED", "authorized_goal_changes": {
            "A4-L2-01": self.authorize(self.old_facts["A4-L2-01"], self.facts["A4-L2-01"], "goal")},
                         "authorized_reference_changes": {
            "A4-L2-01": self.authorize(self.old_rules["references"]["A4-L2-01"],
                                       self.rules["references"]["A4-L2-01"], "reference")}}

    @staticmethod
    def authorize(old, new, kind):
        return {"old_" + kind + "_sha256": canonical_sha256(old),
                "new_" + kind + "_sha256": canonical_sha256(new),
                "field_changes": {key: {"old": value, "new": new.get(key)}
                                  for key, value in old.items() if new.get(key) != value},
                "reason": "用户删除截至7月要求数据库中最新数据", "source": "用户自然题审阅稿 A4-L2-01"}

    def verify(self, *, facts=None, rules=None, approval=None):
        return validate_goal_preservation(self.old_facts, self.facts if facts is None else facts,
                                          self.old_rules, self.rules if rules is None else rules,
                                          self.approval if approval is None else approval)

    def test_exact_authorization_allows_only_named_case_goal_and_sql(self):
        self.assertEqual(self.verify(), {"authorized_goal_change_ids": ["A4-L2-01"],
                                         "authorized_reference_change_ids": ["A4-L2-01"]})

    def test_unapproved_pack_preserves_original_fields_and_allows_private_annotations(self):
        annotated = deepcopy(self.old_facts)
        annotated["A4-L2-01"]["expected_intent"] = "自然问法的私有审核注释"
        self.assertEqual(validate_goal_preservation(self.old_facts, annotated, self.old_rules, self.old_rules),
                         {"authorized_goal_change_ids": [], "authorized_reference_change_ids": []})
        with self.assertRaises(ValueError):
            validate_goal_preservation(self.old_facts, self.facts, self.old_rules, self.rules)

    def test_change_to_other_case_cannot_use_existing_approval(self):
        changed = deepcopy(self.facts)
        changed["A4-L2-02"]["expected_result"]["rows"] = [["999.00"]]
        with self.assertRaises(ValueError):
            self.verify(facts=changed)

    def test_signature_or_exact_field_diff_mismatch_fails(self):
        for prefix in ("old_goal_sha256", "new_goal_sha256"):
            broken = deepcopy(self.approval)
            broken["authorized_goal_changes"]["A4-L2-01"][prefix] = "0" * 64
            with self.subTest(field=prefix), self.assertRaises(ValueError):
                self.verify(approval=broken)
        broken = deepcopy(self.approval)
        del broken["authorized_goal_changes"]["A4-L2-01"]["field_changes"]["expected_result"]
        with self.assertRaises(ValueError):
            self.verify(approval=broken)

    def test_approval_requires_user_source_reason_and_known_case(self):
        for key in ("reason", "source"):
            broken = deepcopy(self.approval)
            broken["authorized_goal_changes"]["A4-L2-01"][key] = ""
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.verify(approval=broken)
        broken = deepcopy(self.approval)
        broken["authorized_goal_changes"]["UNKNOWN"] = deepcopy(broken["authorized_goal_changes"]["A4-L2-01"])
        with self.assertRaises(ValueError):
            self.verify(approval=broken)

    def test_approval_cannot_change_response_or_sql_requirement(self):
        for key, value in (("expected_types", "insufficient"), ("sql_requirements", "optional")):
            changed = deepcopy(self.rules)
            changed[key]["A4-L2-01"] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.verify(rules=changed)
        changed = deepcopy(self.facts)
        changed["A4-L2-01"]["response_type"] = "insufficient"
        approval = deepcopy(self.approval)
        approval["authorized_goal_changes"]["A4-L2-01"] = self.authorize(
            self.old_facts["A4-L2-01"], changed["A4-L2-01"], "goal")
        with self.assertRaises(ValueError):
            self.verify(facts=changed, approval=approval)

    def test_reference_approval_does_not_allow_alias_changes(self):
        changed = deepcopy(self.rules)
        changed["references"]["A4-L2-01"]["aliases"]["income"] = "different"
        approval = deepcopy(self.approval)
        approval["authorized_reference_changes"]["A4-L2-01"] = self.authorize(
            self.old_rules["references"]["A4-L2-01"], changed["references"]["A4-L2-01"], "reference")
        with self.assertRaises(ValueError):
            self.verify(rules=changed, approval=approval)


if __name__ == "__main__":
    unittest.main()
