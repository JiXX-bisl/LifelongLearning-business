from __future__ import annotations

import json
import re
from pathlib import Path
from src.project_paths import resolve_project_path
from typing import Any

from src.lifelong_learning.core.models import (
    AgentAction,
    ChatMessage,
    EvaluationOutcome,
    Role,
    SampleStatus,
    Session,
    TaskName,
)
from src.lifelong_learning.environments.fixed_mysql_environment import (
    FixedMySQLReadOnlyEnvironment,
)
from src.lifelong_learning.tasks.db_bench.response_parser import (
    parse_dbbench_agent_response,
)
from src.lifelong_learning.tasks.enterprise_query_bench.dataset import (
    EnterpriseQueryDataset,
    EnterpriseQueryEntry,
    JSONScalar,
    load_enterprise_query_dataset,
)
from src.lifelong_learning.tasks.enterprise_query_bench.evaluator import (
    compare_query_results,
    normalize_mysql_rows,
)

from src.lifelong_learning.tasks.enterprise_query_bench.schema_catalog import (
    GUIDED_INTERFACE_PROFILE,
)


PSU_PLACEHOLDER = "{previous_sample_utilization_target_position}"


class EnterpriseQueryTask:
    """Run enterprise query tasks against one frozen read-only database."""

    def __init__(
        self,
        *,
        data_file_path: str | Path,
        baseline_dir: str | Path,
        prompt_file_path: str | Path,
        environment: FixedMySQLReadOnlyEnvironment,
        interface_profile: str = GUIDED_INTERFACE_PROFILE,
        max_round: int = 3,
        max_schema_requests: int = 6,
        verify_live_baseline: bool = True,
    ) -> None:
        if max_round <= 0:
            raise ValueError("max_round must be positive.")

        self.dataset_bundle: EnterpriseQueryDataset = (
            load_enterprise_query_dataset(
                data_file_path=data_file_path,
                baseline_dir=baseline_dir,
                interface_profile=interface_profile,
            )
        )
        # Keep the same public shape used by the existing experiment runner.
        self.dataset = self.dataset_bundle.entries
        self.initial_message_list = self._load_initial_messages(prompt_file_path)
        self.environment = environment
        self.max_round = max_round

        if max_schema_requests <= 0:
            raise ValueError(
                "max_schema_requests must be positive."
            )

        self.max_schema_requests = max_schema_requests

        if environment.database != self.dataset_bundle.database_name:
            raise ValueError(
                "Environment database differs from the frozen baseline manifest."
            )

        if verify_live_baseline:
            self.environment.verify_record_counts(
                self.dataset_bundle.expected_record_counts
            )

        self.current_entry: EnterpriseQueryEntry | None = None
        self.current_sample_index: str | None = None
        self.current_round = 0
        self.last_successful_result: list[list[JSONScalar]] | None = None
        self.sql_attempt_list: list[dict[str, Any]] = []

    @staticmethod
    def _load_initial_messages(
        prompt_file_path: str | Path,
    ) -> list[ChatMessage]:
        path = resolve_project_path(prompt_file_path)
        with path.open("r", encoding="utf-8") as file:
            raw_prompt: dict[str, Any] = json.load(file)

        raw_messages = raw_prompt["value"]
        return [
            ChatMessage.model_validate(raw_messages[index])
            for index in sorted(raw_messages, key=int)
        ]

    def get_sample_index_list(self) -> list[str]:
        return self.dataset_bundle.get_sample_index_list()

    def _reset_current_sample(self) -> None:
        self.current_entry = None
        self.current_sample_index = None
        self.current_round = 0
        self.last_successful_result = None
        self.sql_attempt_list = []

    def _mark_limit_if_needed(self, session: Session) -> None:
        """Stop without making an extra LLM call after the last allowed round."""
        if (
            session.sample_status == SampleStatus.RUNNING
            and self.current_round >= self.max_round
        ):
            session.sample_status = SampleStatus.TASK_LIMIT_REACHED
            session.finish_reason = f"Task limit reached: {self.max_round}"

    def start_session(
        self,
        sample_index: str,
        previous_sample_text: str = "",
    ) -> Session:
        if self.current_entry is not None:
            raise RuntimeError("Another enterprise query sample is still active.")

        if sample_index not in self.dataset:
            raise KeyError(f"Unknown sample index: {sample_index}")

        self.current_entry = self.dataset[sample_index]
        self.current_sample_index = sample_index
        self.current_round = 0
        self.last_successful_result = None
        self.sql_attempt_list = []

        session = Session(
            task_name=TaskName.ENTERPRISE_QUERY_BENCH,
            sample_index=sample_index,
            sample_status=SampleStatus.RUNNING,
            task_output={
                "answer": None,
                "last_successful_sql": None,
                "last_successful_result": None,
                "sql_attempts": [],
                "schema_attempts": [],
            },
        )

        for initial_message in self.initial_message_list:
            session.chat_history.append(
                role=initial_message.role,
                content=(
                    initial_message.content
                    .replace(
                        PSU_PLACEHOLDER,
                        previous_sample_text,
                    )
                    .replace(
                        "{schema_lookup_budget}",
                        str(self.max_schema_requests),
                    )
                    .replace(
                        "{query_round_budget}",
                        str(self.max_round),
                    )
                ),
            )

        session.chat_history.append(
            role=Role.USER,
            content=self.dataset_bundle.build_agent_instruction(sample_index),
        )

        return session
    
    def _handle_schema_request(
        self,
        session: Session,
        agent_response: ChatMessage,
    ) -> bool:
        """
        返回True表示已经处理了结构请求。

        结构说明来自冻结Schema，不执行MySQL查询，
        不占用业务轮数，也不覆盖B1最后成功结果。
        """
        parsed = parse_dbbench_agent_response(
            agent_response.content
        )

        if parsed.action != AgentAction.EXECUTE:
            return False

        sql = parsed.content or ""

        if not re.match(
            r"^(?:DESCRIBE|DESC)\b",
            sql,
            re.IGNORECASE,
        ):
            return False

        output = session.task_output
        if output is None:
            raise RuntimeError("Missing task_output.")

        attempts = output["schema_attempts"]

        attempt = {
            "request": sql,
            "success": False,
            "result": None,
            "error": None,
        }

        if len(attempts) >= self.max_schema_requests:
            attempt["error"] = (
                "Schema request limit reached."
            )
            attempts.append(attempt)

            session.sample_status = (
                SampleStatus.TASK_LIMIT_REACHED
            )
            session.finish_reason = attempt["error"]
            session.chat_history.append(
                role=Role.USER,
                content=attempt["error"],
            )
            return True

        try:
            match = re.fullmatch(
                r"(?:DESCRIBE|DESC)\s+"
                r"(?:`([A-Za-z_][A-Za-z0-9_]*)`"
                r"|([A-Za-z_][A-Za-z0-9_]*))"
                r"\s*;?",
                sql.strip(),
                flags=re.IGNORECASE,
            )

            if match is None:
                raise ValueError(
                    "Use exactly: DESCRIBE table_name;"
                )

            table_name = (
                match.group(1) or match.group(2)
            )
            schema_text = (
                self.dataset_bundle.describe_table(
                    table_name
                )
            )

            attempt["success"] = True
            attempt["result"] = schema_text

            observation = (
                "表结构说明（不是业务查询结果）：\n"
                f"{schema_text}"
            )

        except ValueError as error:
            attempt["error"] = str(error)
            observation = (
                f"查看结构失败：{error}"
            )

        attempts.append(attempt)

        remaining = (
            self.max_schema_requests - len(attempts)
        )

        session.chat_history.append(
            role=Role.USER,
            content=(
                f"{observation}\n"
                f"剩余结构查看次数：{remaining}"
            ),
        )

        return True

    def interact(self, session: Session, agent_response: ChatMessage) -> None:
        if self.current_entry is None:
            raise RuntimeError("No active enterprise query sample.")

        session.chat_history.append(
            role=agent_response.role,
            content=agent_response.content,
        )

        if self._handle_schema_request(
            session,
            agent_response,
        ):
            return

        if self.current_round >= self.max_round:
            session.sample_status = SampleStatus.TASK_LIMIT_REACHED
            session.finish_reason = f"Task limit reached: {self.max_round}"
            return

        self.current_round += 1
        parsed_response = parse_dbbench_agent_response(agent_response.content)

        if parsed_response.action == AgentAction.EXECUTE:
            sql = parsed_response.content
            if sql is None:
                raise RuntimeError("Parsed SQL content is missing.")

            attempt: dict[str, Any] = {
                "round": self.current_round,
                "sql": sql,
                "success": False,
                "result": None,
                "error": None,
            }

            try:
                raw_rows = self.environment.execute(sql)
                normalized_rows = normalize_mysql_rows(raw_rows)
            except Exception as error:
                attempt["error"] = str(error)
                self.sql_attempt_list.append(attempt)
                session.task_output["sql_attempts"] = list(self.sql_attempt_list)
                session.chat_history.append(
                    role=Role.USER,
                    content=f"SQL execution failed: {error}",
                )
                self._mark_limit_if_needed(session)
                return

            attempt["success"] = True
            attempt["result"] = normalized_rows
            self.sql_attempt_list.append(attempt)
            self.last_successful_result = normalized_rows

            session.task_output.update(
                {
                    "last_successful_sql": sql,
                    "last_successful_result": normalized_rows,
                    "sql_attempts": list(self.sql_attempt_list),
                }
            )
            session.chat_history.append(
                role=Role.USER,
                content=str(raw_rows),
            )
            self._mark_limit_if_needed(session)
            return

        if parsed_response.action == AgentAction.FINISH:
            session.sample_status = SampleStatus.COMPLETED
            session.task_output["answer"] = parsed_response.content
            return

        session.sample_status = SampleStatus.AGENT_VALIDATION_FAILED
        session.finish_reason = parsed_response.finish_reason

    def complete(self, session: Session) -> None:
        if self.current_entry is None:
            raise RuntimeError("No active enterprise query sample.")

        expected = self.current_entry.answer_info.direct
        actual = self.last_successful_result

        correct = (
            actual is not None
            and compare_query_results(
                actual=actual,
                expected=expected,
                row_order=(
                    self.current_entry.answer_info.comparison_policy.row_order
                ),
            )
        )

        session.evaluation_record.outcome = (
            EvaluationOutcome.CORRECT
            if correct
            else EvaluationOutcome.INCORRECT
        )
        self._reset_current_sample()

    def close(self) -> None:
        self._reset_current_sample()
        self.environment.close()
