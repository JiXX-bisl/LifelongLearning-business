"""SQL 输出动作归组回归用例；执行由用户负责。"""
from __future__ import annotations

import unittest
from dataclasses import replace

from src.psu_memory.pool.classification import apply_group_override, classify
from src.psu_memory.pool.pool_models import GroupingInput, PoolConfig


def make_request(sql: str, **kwargs) -> GroupingInput:
    return GroupingInput(
        execution_id="lesson-execution-001", content_version="lesson-content-001",
        question=kwargs.pop("question", "看看这些企业的情况。"),
        namespace="lesson", data_version="v1", sql=sql, **kwargs,
    )


class PoolClassificationTests(unittest.TestCase):
    def assert_group(self, sql, expected):
        result = classify(make_request(sql))
        self.assertEqual(result.group_id, expected)
        self.assertEqual(result.text_source, "sql")
        self.assertEqual(result.method, "rule")
        return result

    def test_four_groups(self):
        cases = (
            ("SELECT enterprise_id, revenue FROM monthly", "detail"),
            ("SELECT SUM(revenue) FROM monthly", "aggregate"),
            ("SELECT revenue - LAG(revenue) OVER (ORDER BY report_month) FROM monthly", "compare"),
            ("SELECT enterprise_id, revenue FROM monthly ORDER BY revenue DESC LIMIT 3", "rank"),
        )
        for sql, expected in cases:
            with self.subTest(sql=sql):
                self.assert_group(sql, expected)

    def test_question_context_and_advice_do_not_change_group(self):
        request = make_request("SELECT enterprise_id, revenue FROM monthly")
        baseline = classify(request)
        changed = classify(replace(
            request, question="比较并排名前十，再统计总额。",
            context_text='[{"role":"user","content":"按收入排名"}]',
            data_advice="COUNT 和排名都必须执行",
        ))
        self.assertEqual(baseline.group_id, changed.group_id)
        self.assertEqual(baseline.input_sha256, changed.input_sha256)

    def test_internal_dedup_and_version_ranking_are_not_rank(self):
        prefix = """WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (
              PARTITION BY enterprise_id, source_record_ref
              ORDER BY report_revision DESC, ingested_at DESC, report_id DESC
            ) AS rn FROM monthly
        ) """
        self.assert_group(prefix + "SELECT revenue FROM ranked WHERE rn = 1", "detail")
        self.assert_group(prefix + "SELECT SUM(revenue) FROM ranked WHERE rn = 1", "aggregate")
        self.assert_group(prefix + "SELECT rn, revenue FROM ranked", "detail")

    def test_latest_date_and_presentation_order_are_not_rank(self):
        for sql in (
            "SELECT * FROM monthly ORDER BY report_month DESC LIMIT 1",
            "SELECT * FROM monthly ORDER BY report_month DESC, revenue DESC LIMIT 1",
            "SELECT enterprise_id, revenue FROM monthly ORDER BY enterprise_id LIMIT 5",
            "SELECT enterprise_id, revenue FROM monthly ORDER BY revenue DESC",
        ):
            with self.subTest(sql=sql):
                self.assert_group(sql, "detail")
        self.assert_group("SELECT MAX(report_month) FROM monthly", "aggregate")

    def test_rank_by_metric_alias_ordinal_count_and_extreme(self):
        for sql in (
            "SELECT enterprise_id, SUM(revenue) AS total FROM monthly GROUP BY enterprise_id ORDER BY total DESC LIMIT 3",
            "SELECT enterprise_id, SUM(revenue) FROM monthly GROUP BY enterprise_id ORDER BY 2 DESC LIMIT 3",
            "SELECT enterprise_id, COUNT(*) AS n FROM monthly GROUP BY enterprise_id ORDER BY n DESC LIMIT 3",
            "SELECT MAX(revenue) FROM monthly",
            "SELECT MIN(revenue) FROM monthly",
            "WITH a AS (SELECT enterprise_id, revenue FROM monthly ORDER BY revenue DESC LIMIT 3) SELECT * FROM a",
        ):
            with self.subTest(sql=sql):
                self.assert_group(sql, "rank")

    def test_business_window_ranking_and_final_aggregation(self):
        for function in ("ROW_NUMBER", "RANK", "DENSE_RANK"):
            prefix = f"WITH x AS (SELECT enterprise_id, revenue, {function}() OVER (ORDER BY revenue DESC) AS r FROM monthly) "
            with self.subTest(function=function):
                self.assert_group(prefix + "SELECT enterprise_id, revenue FROM x WHERE r <= 3", "rank")
                self.assert_group(prefix + "SELECT COUNT(*) FROM x WHERE r <= 3", "aggregate")
                self.assert_group(prefix + "SELECT SUM(r) FROM x", "aggregate")
                self.assert_group(prefix + "SELECT revenue FROM x", "detail")

    def test_comparison_and_priority(self):
        sql = """SELECT
          SUM(CASE WHEN report_month = '2026-04' THEN revenue ELSE 0 END) AS april,
          SUM(CASE WHEN report_month = '2026-05' THEN revenue ELSE 0 END) AS may
          FROM monthly"""
        self.assert_group(sql, "compare")
        self.assert_group(sql + " ORDER BY april DESC LIMIT 3", "rank")
        self.assert_group(
            "SELECT a.revenue - b.revenue FROM monthly a JOIN monthly b ON a.enterprise_id = b.enterprise_id",
            "compare",
        )
        self.assert_group(
            "WITH totals AS (SELECT enterprise_id, SUM(revenue) AS total FROM monthly GROUP BY enterprise_id) SELECT a.total, b.total FROM totals a JOIN totals b ON a.enterprise_id <> b.enterprise_id",
            "compare",
        )
        self.assert_group("SELECT revenue - cost FROM monthly", "detail")

    def test_conditional_pivot_is_not_extreme(self):
        self.assert_group(
            "SELECT MAX(CASE WHEN report_month = '2026-04' THEN revenue END) FROM monthly",
            "aggregate",
        )
        self.assert_group(
            "SELECT SUM(revenue * CASE amount_unit WHEN '万元' THEN 10000 ELSE 1 END) FROM monthly",
            "aggregate",
        )

    def test_final_projection_resolves_cte_and_subquery(self):
        self.assert_group(
            "WITH a AS (SELECT SUM(revenue) AS total FROM monthly) SELECT total FROM a",
            "aggregate",
        )
        self.assert_group(
            "SELECT q.total FROM (SELECT SUM(revenue) AS total FROM monthly) q",
            "aggregate",
        )
        self.assert_group(
            "WITH unused AS (SELECT MAX(revenue) FROM monthly) SELECT enterprise_id FROM enterprise",
            "detail",
        )
        self.assert_group(
            "SELECT enterprise_id FROM enterprise WHERE enterprise_id IN (SELECT enterprise_id FROM monthly GROUP BY enterprise_id HAVING COUNT(*) > 1)",
            "detail",
        )

    def test_alias_projection_does_not_create_a_comparison(self):
        self.assert_group(
            "SELECT t.revenue FROM (SELECT m.revenue FROM monthly m) t",
            "detail",
        )
        self.assert_group(
            "SELECT t.total FROM (SELECT SUM(m.revenue) AS total FROM monthly m) t",
            "aggregate",
        )

    def test_final_aggregate_hides_intermediate_extreme_or_change(self):
        self.assert_group(
            "SELECT SUM(max_revenue) FROM (SELECT MAX(revenue) AS max_revenue FROM monthly GROUP BY enterprise_id) t",
            "aggregate",
        )
        self.assert_group(
            "SELECT SUM(delta) FROM (SELECT revenue - LAG(revenue) OVER (ORDER BY report_month) AS delta FROM monthly) t",
            "aggregate",
        )
        self.assert_group(
            "SELECT LAG(report_month) OVER (ORDER BY revenue) FROM monthly",
            "detail",
        )

    def test_unrecognized_sql_has_one_detail_group(self):
        for sql in ("", "SELECT (", "SELECT 1; SELECT 2", "UPDATE monthly SET revenue=1"):
            with self.subTest(sql=sql):
                self.assert_group(sql, "detail")

    def test_fingerprint_tracks_sql_not_execution(self):
        request = make_request("SELECT SUM(revenue) FROM monthly")
        original = classify(request)
        duplicate = classify(replace(request, execution_id="another", content_version="another"))
        changed = classify(replace(request, sql="SELECT revenue FROM monthly"))
        self.assertEqual(original.input_sha256, duplicate.input_sha256)
        self.assertNotEqual(original.input_sha256, changed.input_sha256)

    def test_legacy_override_is_explicit_and_idempotent(self):
        original = classify(make_request("SELECT revenue FROM monthly"))
        changed = apply_group_override(original, "aggregate", reason="旧接口兼容")
        replayed = apply_group_override(changed, "aggregate", reason="旧接口兼容")
        self.assertEqual(changed, replayed)
        self.assertEqual(original.group_id, "detail")
        self.assertEqual(len(changed.manual_history), 1)
        with self.assertRaises(ValueError):
            apply_group_override(original, "other", reason="非法分组")

    def test_capacity_and_input_validation(self):
        self.assertEqual(PoolConfig().capacity_for("detail"), 8)
        for config in ({"detail": 8}, {"detail": 8, "aggregate": 8, "compare": 8, "rank": -1}):
            with self.assertRaises(ValueError):
                PoolConfig(capacities=config)
        request = make_request("SELECT 1")
        with self.assertRaises(ValueError):
            replace(request, sql=None)
        with self.assertRaises(ValueError):
            classify(request, max_context_questions=-1)


if __name__ == "__main__":
    unittest.main()
