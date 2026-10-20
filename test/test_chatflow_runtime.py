from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatflow_agent.runtime import (
    BusinessRuntimeError,
    check_business_runtime,
)


class FakeEnvironment:
    instances: list["FakeEnvironment"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.__class__.instances.append(self)

    def execute(self, sql: str) -> list[tuple[int]]:
        if sql != "SELECT 1":
            raise AssertionError(sql)
        return [(1,)]

    def close(self) -> None:
        self.closed = True


class FailingEnvironment(FakeEnvironment):
    def execute(self, sql: str) -> list[tuple[int]]:
        raise ConnectionError("service unavailable")


class BusinessRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeEnvironment.instances.clear()
        FailingEnvironment.instances.clear()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.baseline_dir = self.root / "baseline"
        self.baseline_dir.mkdir()
        (self.baseline_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "baseline_id": "baseline-v1",
                    "mysql": {"database": "enterprise_test"},
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_config(self, baseline_id: str = "baseline-v1") -> Path:
        config_path = self.root / "query.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "type": "openai_compatible",
                        "model_name": "test-model",
                        "api_key_env": "TEST_API_KEY",
                        "base_url_env": "TEST_BASE_URL",
                    },
                    "database": {
                        "host": "127.0.0.1",
                        "port": 3306,
                        "user": "tester",
                        "password_env": "TEST_DB_PASSWORD",
                        "connection_timeout_seconds": 1,
                    },
                    "data": {
                        "baseline_dir": str(self.baseline_dir),
                        "expected_baseline_id": baseline_id,
                    },
                    "output": {"directory": str(self.root / "output")},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return config_path

    @staticmethod
    def environment_values() -> dict[str, str]:
        return {
            "TEST_API_KEY": "secret-api-key",
            "TEST_BASE_URL": "https://model.invalid/v1",
            "TEST_DB_PASSWORD": "secret-db-password",
        }

    def test_success_returns_no_credentials_and_closes_connection(self) -> None:
        config_path = self.write_config()
        with patch.dict(os.environ, self.environment_values(), clear=False), patch(
            "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment",
            FakeEnvironment,
        ):
            result = check_business_runtime(config_path)

        serialized = json.dumps(result)
        self.assertNotIn("secret-api-key", serialized)
        self.assertNotIn("secret-db-password", serialized)
        self.assertFalse(result["model"]["probe_performed"])
        self.assertEqual(result["database"]["database"], "enterprise_test")
        self.assertEqual(result["database"]["connection_timeout_seconds"], 1)
        self.assertEqual(result["database"]["fetch_batch_size"], 50)
        self.assertEqual(result["database"]["max_total_rows"], 500)        
        self.assertTrue(FakeEnvironment.instances[0].closed)
        self.assertEqual(
            result["learning"],
            {
                "mode": "standard",
                "experiment_id": "q05_dev",
                "utilized_sample_count": 4,
                "group_count": 2,
                "sample_count_per_group": 2,
            },
        )


    def test_missing_environment_variable_stops_before_connection(self) -> None:
        config_path = self.write_config()
        with patch.dict(os.environ, {}, clear=True), patch(
            "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment",
            FakeEnvironment,
        ):
            with self.assertRaisesRegex(
                BusinessRuntimeError,
                "缺少环境变量 TEST_API_KEY",
            ):
                check_business_runtime(config_path)
        self.assertEqual(FakeEnvironment.instances, [])

    def test_exported_model_name_overrides_yaml_and_is_returned_to_query(self) -> None:
        config_path = self.write_config()
        original = config_path.read_bytes()
        with patch.dict(os.environ, {
            **self.environment_values(), "OPENAI_MODEL_NAME": "  exported-query-model  ",
        }, clear=True), patch(
            "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment", FakeEnvironment,
        ):
            result = check_business_runtime(config_path)
        self.assertEqual(result["model"]["model_name"], "exported-query-model")
        self.assertEqual(config_path.read_bytes(), original)
        self.assertFalse(result["model"]["probe_performed"])

    def test_missing_or_blank_model_environment_falls_back_to_yaml(self) -> None:
        config_path = self.write_config()
        for value in (None, "", " \t "):
            environment = self.environment_values()
            if value is not None:
                environment["OPENAI_MODEL_NAME"] = value
            with self.subTest(value=value), patch.dict(os.environ, environment, clear=True), patch(
                "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment", FakeEnvironment,
            ):
                result = check_business_runtime(config_path)
            self.assertEqual(result["model"]["model_name"], "test-model")

    def test_exported_model_name_can_supply_missing_yaml_name(self) -> None:
        config_path = self.write_config()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        del config["model"]["model_name"]
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with patch.dict(os.environ, {
            **self.environment_values(), "OPENAI_MODEL_NAME": "exported-query-model",
        }, clear=True), patch(
            "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment", FakeEnvironment,
        ):
            result = check_business_runtime(config_path)
        self.assertEqual(result["model"]["model_name"], "exported-query-model")

    def test_baseline_mismatch_stops_before_connection(self) -> None:
        config_path = self.write_config(baseline_id="baseline-v2")
        with patch.dict(os.environ, self.environment_values(), clear=False), patch(
            "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment",
            FakeEnvironment,
        ):
            with self.assertRaisesRegex(BusinessRuntimeError, "数据基线版本不匹配"):
                check_business_runtime(config_path)
        self.assertEqual(FakeEnvironment.instances, [])

    def test_malformed_yaml_returns_concise_cli_error(self) -> None:
        import contextlib
        import io

        from scripts.check_business_runtime import main

        try:
            yaml.safe_load("model: [")
        except yaml.YAMLError as error:
            parse_error = error

        stderr = io.StringIO()
        with patch(
            "sys.argv", ["check_business_runtime.py", "--config", "invalid.yaml"]
        ), patch(
            "src.chatflow_agent.runtime.load_yaml_config", side_effect=parse_error
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(), 1)
        self.assertIn("YAML格式", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_blank_environment_variables_stop_before_connection(self) -> None:
        config_path = self.write_config()
        for name in self.environment_values():
            with self.subTest(environment=name), patch.dict(
                os.environ,
                {**self.environment_values(), name: " \t "},
                clear=True,
            ), patch(
                "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment"
            ) as connection:
                with self.assertRaisesRegex(BusinessRuntimeError, name):
                    check_business_runtime(config_path)
                connection.assert_not_called()

    def test_unwritable_output_stops_before_connection(self) -> None:
        config_path = self.write_config()
        with patch.dict(os.environ, self.environment_values(), clear=False), patch(
            "src.chatflow_agent.runtime.tempfile.NamedTemporaryFile",
            side_effect=PermissionError("read only"),
        ), patch(
            "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment",
            FakeEnvironment,
        ):
            with self.assertRaisesRegex(BusinessRuntimeError, "输出目录不可写"):
                check_business_runtime(config_path)
        self.assertEqual(FakeEnvironment.instances, [])

    def test_database_failure_still_closes_connection(self) -> None:
        config_path = self.write_config()
        with patch.dict(os.environ, self.environment_values(), clear=False), patch(
            "src.chatflow_agent.runtime.FixedMySQLReadOnlyEnvironment",
            FailingEnvironment,
        ):
            with self.assertRaisesRegex(BusinessRuntimeError, "数据库连接失败"):
                check_business_runtime(config_path)
        self.assertTrue(FailingEnvironment.instances[0].closed)
