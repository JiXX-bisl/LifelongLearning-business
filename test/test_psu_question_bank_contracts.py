"""用户独立运行：题库来源的静态契约；不参与查询或SQL结果评分。"""
from __future__ import annotations

from copy import deepcopy
import unittest

from src.psu_memory.question_bank_contracts import bind_case_contracts, validate_intent_contracts


class QuestionBankContractsTests(unittest.TestCase):
    def test_private_response_contract_does_not_modify_public_questions(self):
        public = [{"id": str(index), "question": "公开问题"} for index in range(3)]
        before = deepcopy(public)
        cases = bind_case_contracts(public, {"expected_types": {
            "0": "answer", "1": "clarify", "2": "insufficient"}})
        self.assertEqual(public, before)
        self.assertEqual([case["kind"] for case in cases], ["query", "clarification", "insufficient"])
        self.assertEqual([case["sql_requirement"] for case in cases],
                         ["required", "not_applicable", "optional"])

    def test_legacy_kinds_remain_available_for_question_sources(self):
        cases = bind_case_contracts([{"id": "old-q", "kind": "query"},
                                     {"id": "old-c", "kind": "clarification"}], {})
        self.assertEqual([case["expected_response_type"] for case in cases], ["answer", "clarify"])
        self.assertTrue(all(case["response_contract_version"] == "legacy" for case in cases))

    def test_private_response_contract_must_cover_the_public_cases(self):
        with self.assertRaises(ValueError):
            bind_case_contracts([{"id": "a"}, {"id": "b"}], {"expected_types": {"a": "answer"}})
        with self.assertRaises(ValueError):
            bind_case_contracts([{"id": "a", "kind": "query"}], {"expected_types": {"a": "clarify"}})

    def test_intent_rubric_is_not_copied_into_public_questions(self):
        public = {"id": "q1", "question": "看看最近的盈利情况"}
        before = deepcopy(public)
        rules = {"expected_types": {"q1": "answer"}, "intent_contract_version": "natural_intent_v1",
                 "intent_rubrics": {"q1": "私有还原依据"}}
        bound = bind_case_contracts([public], rules)[0]
        self.assertEqual(public, before)
        self.assertEqual(bound["intent_contract_version"], "natural_intent_v1")
        self.assertNotIn("intent_rubric", bound)
        with self.assertRaises(ValueError):
            bind_case_contracts([public], {**rules, "intent_rubrics": {}})

    def test_user_approval_preserves_the_original_alignment_diagnosis(self):
        for alignment in ("convention_needs_review", "scope_conflict"):
            with self.subTest(alignment=alignment):
                target = {"target_id": "goal", "description": "用户确认的盈利分析目的",
                          "implicit": True, "alignment_status": alignment,
                          "restoration_basis": [{"kind": "business_convention", "text": "用户确认业务用途",
                                                 "source": "用户审阅稿", "observable": False}],
                          "review_status": "APPROVED",
                          "review_resolution": {"reason": "用户确认此业务目的", "source": "用户指令"}}
                document = {"intent_contract_version": "natural_intent_v1", "data_version": "v1",
                            "reviewed_by_user": True, "pending_reviews": [],
                            "cases": [{"id": "q1", "basis_reviewed_by_user": True,
                                       "required_business_targets": [target]}]}
                actual = validate_intent_contracts(document, [{"id": "q1"}], data_version="v1")
                self.assertEqual(actual["cases"][0]["required_business_targets"][0]["alignment_status"], alignment)
                for missing in ("reason", "source"):
                    broken = deepcopy(document)
                    del broken["cases"][0]["required_business_targets"][0]["review_resolution"][missing]
                    with self.assertRaises(ValueError):
                        validate_intent_contracts(broken, [{"id": "q1"}])


if __name__ == "__main__":
    unittest.main()
