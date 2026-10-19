"""Basic prompt composition and resource boundaries; run independently by user."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.chatflow_agent.memory import BusinessMemorySnapshot
from src.chatflow_agent.prompt_resources import (
    BASIC_PROMPT_PROFILE, load_prompt_resources, parse_prompt_config,
)
from src.chatflow_agent.protocol import build_business_prompt, parse_business_response
from src.chatflow_agent.strategy import BusinessQueryStrategy, EXPERIENCE_BOUNDARY
from src.chatflow_agent.query import QueryService
from src.lifelong_learning.core.models import AgentAction, ChatHistory, Role
from src.lifelong_learning.tasks.enterprise_query_bench.schema_catalog import (
    build_basic_schema_catalog,
)

ROOT = Path(__file__).resolve().parents[1]


class BasicPromptTests(unittest.TestCase):
    def make_prompt(self):
        config = parse_prompt_config({"profile": BASIC_PROMPT_PROFILE})
        resources = load_prompt_resources(config)
        schema = (ROOT / "data/sql/synthetic_enterprise_v30_120_v1.0.2/database_schema.sql").read_text()
        catalog = build_basic_schema_catalog(
            schema, resources["field_mapping"],
            table_descriptions=resources["table_descriptions"],
        )
        prompt = build_business_prompt(
            "synthetic_enterprise_v30_120", "8.0", "2026-08-31",
            schema_version="3.0", prompt_profile=BASIC_PROMPT_PROFILE,
            basic_schema_catalog=catalog, basic_rules=resources["rules"],
        )
        return prompt, catalog

    def test_all_fields_have_one_mapping_without_task_rules(self):
        prompt, catalog = self.make_prompt()
        self.assertEqual(len(catalog["table_fields"]), 7)
        self.assertEqual(sum(map(len, catalog["table_fields"].values())), 120)
        self.assertEqual(len(catalog["field_mapping"]), 80)
        for name in catalog["field_mapping"]:
            self.assertEqual(sum(line.startswith(name + " → ") for line in prompt.splitlines()), 1)
        self.assertEqual(prompt.count("Action: Operation"), 1)
        self.assertEqual(prompt.count("Action: Answer"), 1)
        self.assertIn("Result Query ID:", prompt)
        for private_or_old in ("默认营收", "公开数据口径", "A1-L1-01", "reference_sql",
                               "required_business_targets", "不同自然人6位", "任职事实7段"):
            self.assertNotIn(private_or_old, prompt)
        self.assertLess(len(prompt), 9000)

    def test_basic_strategy_preserves_base_and_legacy_adds_boundary(self):
        prompt, _ = self.make_prompt()
        snapshot = BusinessMemorySnapshot(version=0, sessions=())
        basic = BusinessQueryStrategy(
            language_model=Mock(), system_prompt=prompt, learning={"mode": "standard"},
            snapshot=snapshot, prompt_profile=BASIC_PROMPT_PROFILE,
        )
        self.assertEqual(basic.agent.system_prompt, prompt)
        self.assertEqual(basic.agent.system_prompt.count("经验用于参考方法"), 1)
        legacy = BusinessQueryStrategy(
            language_model=Mock(), system_prompt="legacy", learning={"mode": "standard"},
            snapshot=snapshot,
        )
        self.assertEqual(legacy.agent.system_prompt, "legacy\n\n" + EXPERIENCE_BOUNDARY)

    def test_action_protocol_remains_compatible(self):
        query = 'Action: Operation\nScope: {"enterprise": null, "time_range": null}\n```sql\nSELECT 1;\n```'
        self.assertEqual(parse_business_response(query).action, AgentAction.EXECUTE)
        self.assertEqual(parse_business_response("Action: Answer\nFinal Answer:\n资料不足。").action,
                         AgentAction.FINISH)

    def test_answer_binds_actual_query_without_exposing_id_in_body(self):
        parsed = parse_business_response(
            "Action: Answer\nResult Query ID: query_123\nFinal Answer:\n共有40家企业。"
        )
        self.assertEqual(parsed.action, AgentAction.FINISH)
        self.assertEqual(parsed.result_query_id, "query_123")
        self.assertEqual(parsed.content, "共有40家企业。")
        for reference in ("", "Result Query ID: null\n"):
            parsed = parse_business_response(
                "Action: Answer\n" + reference + "Final Answer:\n资料不足。"
            )
            self.assertEqual(parsed.action, AgentAction.FINISH)
            self.assertIsNone(parsed.result_query_id)

    def test_answer_rejects_malformed_query_reference(self):
        for reference in ("", "query 123", "query1,query2", "query1\nResult Query ID: query2"):
            with self.subTest(reference=reference):
                parsed = parse_business_response(
                    "Action: Answer\nResult Query ID: " + reference + "\nFinal Answer:\n回复。"
                )
                self.assertEqual(parsed.action, AgentAction.INVALID)

    def test_legacy_prompt_remains_default(self):
        prompt = build_business_prompt("old", "8.0", "2026-08-31")
        self.assertIn("默认营收概览范围：2026-06-01至2026-08-31", prompt)
        self.assertEqual(parse_prompt_config(), {"profile": "legacy_v1"})

    def test_resources_reject_private_extra_metadata(self):
        config = parse_prompt_config({"profile": BASIC_PROMPT_PROFILE})
        mapping = json.loads(Path(config["field_mapping_file"]).read_text())
        mapping["required_business_targets"] = "private"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(mapping))
            with self.assertRaisesRegex(ValueError, "结构"):
                load_prompt_resources({**config, "field_mapping_file": str(path)})

    def test_invalid_profile_and_basic_without_structure_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_prompt_config({"profile": "unknown"})
        with self.assertRaises(ValueError):
            build_business_prompt("x", "8.0", "2026-08-31", prompt_profile=BASIC_PROMPT_PROFILE)

    def test_record_carries_profile_and_resource_hashes_without_adapter_override(self):
        with tempfile.TemporaryDirectory() as directory:
            service = QueryService.__new__(QueryService)
            service.session_id = "session"
            service._turn_index = 0
            service.config = {"output_dir": directory, "data": {"baseline_id": "v1"},
                              "model": {"model_name": "mock"}}
            service.data_context = {"reference_date": "2026-08-31", "schema_version": "3.0",
                                    "prompt_profile": BASIC_PROMPT_PROFILE}
            service.prompt_resource_hashes = {"rules_file": "rules-hash", "field_mapping_file": "map-hash"}
            result = {"psu": {"base_prompt": "actual input"}}
            service._save_result(result)
            saved = json.loads(Path(result["record_path"]).read_text())
            self.assertEqual(saved["prompt_profile"], BASIC_PROMPT_PROFILE)
            self.assertEqual(saved["prompt_resource_sha256"], service.prompt_resource_hashes)
            self.assertEqual(saved["base_prompt_chars"], len("actual input"))

    def test_public_dates_are_added_only_for_empty_context(self):
        for supplied in (False, True):
            with self.subTest(supplied=supplied):
                service = QueryService.__new__(QueryService)
                service.environment = object()
                service.session_id = "session"
                service._turn_index = 0
                service._history = ChatHistory()
                if supplied:
                    service._history.append(Role.USER, "已有公开日期背景。")
                service.data_context = {"prompt_profile": BASIC_PROMPT_PROFILE}
                service.public_date_context = {"reference_date": "2026-08-31",
                                               "knowledge_cutoff_at": "2026-09-15 23:59:59",
                                               "timezone": "Asia/Shanghai"}
                service.learning = {"mode": "standard", "experiment_id": "mock"}
                service.language_model = Mock()
                service.experience_store = Mock()
                service._make_strategy = Mock(return_value=Mock(memory_version=0, used_experience_ids=[]))
                service._run_query = Mock(return_value={"question": "当前问题", "status": "completed",
                    "answer": "当前回答", "rounds": 1, "operations": [], "messages": []})
                service._save_result = Mock()
                service._remember_result = Mock()
                result = service.query("当前问题")
                self.assertEqual(result["context_message_count"], 1)
                self.assertEqual(len(service._history.messages), 1)
                if supplied:
                    self.assertEqual(service._history.messages[0].content, "已有公开日期背景。")
                else:
                    self.assertIn("2026-08-31", service._history.messages[0].content)


if __name__ == "__main__":
    unittest.main()
