"""用户独立运行：查询内检索复用及未知成本保留；只使用模拟对象。"""
from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.psu_memory.integration.common import add_costs, zero_cost
from src.psu_memory.integration.local_query import _PromptStrategy
from src.psu_memory.models import ContextItem, RetrievalRequest, RetrievalResult, UsageMetrics


def retrieval_reply(applied=True, *, seconds=.3):
    result = RetrievalResult(
        items=[ContextItem("memory-1", "reference", "local", "v1")] if applied else [],
        reason_code="enhanced" if applied else "enhancement_rejected",
        usage=UsageMetrics(seconds, 1, 10, 20, 0),
    )
    result.diagnostics = {
        "cost_summary": {"http_request_count": 1, "retry_count": 0, "token_usage_complete": True},
        "augmentation": {"usage": zero_cost(.1)},
    }
    return result


class QueryIntegrationTests(unittest.TestCase):
    def test_same_task_reuses_the_initial_selection(self):
        first, checked = retrieval_reply(), retrieval_reply(seconds=.1)
        retriever = Mock()
        retriever.enhance_prompt.side_effect = [("base\nreference", first), ("base\nreference", checked)]
        delegate = Mock()
        delegate.agent.system_prompt = "base"
        strategy = _PromptStrategy(delegate, retriever, RetrievalRequest("task", "q", "scope", "v1"))
        strategy.decide([], task_id="task", round_index=1)
        strategy.decide([], task_id="task", round_index=2)
        calls = retriever.enhance_prompt.call_args_list
        self.assertIsNone(calls[0].kwargs["retrieval_result"])
        self.assertIs(calls[1].kwargs["retrieval_result"], first)
        usage = strategy.metadata()["usage"]
        self.assertEqual(usage["model_call_count"], 1)
        self.assertEqual(usage["total_llm_tokens"], 30)
        self.assertAlmostEqual(usage["elapsed_seconds"], .4)

    def test_recheck_can_remove_an_experience_without_erasing_its_cost(self):
        retriever = Mock()
        retriever.enhance_prompt.side_effect = [("base\nreference", retrieval_reply()),
                                               ("base", retrieval_reply(False))]
        delegate = Mock()
        delegate.agent.system_prompt = "base"
        strategy = _PromptStrategy(delegate, retriever, RetrievalRequest("t", "q", "n", "v"))
        strategy.decide([], task_id="t", round_index=1)
        strategy.decide([], task_id="t", round_index=2)
        self.assertEqual(delegate.agent.system_prompt, "base")
        self.assertEqual(strategy.metadata()["usage"]["model_call_count"], 1)

    def test_unknown_token_cost_is_not_filled_with_zero(self):
        unknown = {**zero_cost(), "prompt_tokens": None, "token_usage_complete": False}
        known = {**zero_cost(), "prompt_tokens": 12}
        total = add_costs([unknown, known])
        self.assertIsNone(total["total_llm_tokens"])
        self.assertEqual(total["reported_prompt_tokens"], 12)

    def test_known_partial_cost_survives_nested_sums(self):
        unknown = {**zero_cost(), "model_call_count": None}
        known = {**zero_cost(), "model_call_count": 2}
        total = add_costs([add_costs([unknown, known])])
        self.assertIsNone(total["model_call_count"])
        self.assertEqual(total["reported_model_call_count"], 2)


if __name__ == "__main__":
    unittest.main()
