from __future__ import annotations

import json
from pathlib import Path
from src.project_paths import resolve_project_path
from typing import Any

from src.lifelong_learning.core.models import (
    AgentAction, ChatMessage, EvaluationOutcome, Role, SampleStatus, Session, TaskName
)

from src.lifelong_learning.environments.mysql_environment import (
    MySQLEnvironment
)

from src.lifelong_learning.tasks.db_bench.dataset import(
    AnswerType, DBBenchEntry, DBValue, load_dbbench_dataset
)

from src.lifelong_learning.tasks.db_bench.evaluator import (
    validate_direct_answer, validate_table_state
)

from src.lifelong_learning.tasks.db_bench.response_parser import (
    parse_dbbench_agent_response
)

PSU_PLACEHOLDER = (
    "{previous_sample_utilization_target_position}"
)


class DBBenchTask:
    """Mange a full DBBench life long"""

    def __init__(
        self, 
        data_file_path: str | Path, 
        prompt_file_path: str | Path,
        max_round: int = 3,
        environment: MySQLEnvironment | None = None
    ) -> None:
        self.dataset = load_dbbench_dataset(data_file_path)
        self.initial_message_list = self._load_initial_messages(
            prompt_file_path
        )
        self.max_round = max_round

        if environment is None:
            environment = MySQLEnvironment()
        
        self.environment = environment
        self.current_entry: DBBenchEntry | None = None
        self.current_sample_index: str | None = None
        self.current_round = 0

    @staticmethod
    def _load_initial_messages(prompt_file_path: str | Path) -> list[ChatMessage]:
        """Read Standard or PSU initial message"""
        prompt_file_path = resolve_project_path(prompt_file_path)

        with prompt_file_path.open("r", encoding="utf-8") as file:
            raw_prompt: dict[str, Any] = json.load(file)

        raw_message_dict = raw_prompt["value"]

        message_index_list = sorted(raw_message_dict, key=int)

        return [
            ChatMessage.model_validate(
                raw_message_dict[message_index]
            )
            for message_index in message_index_list
        ]

    @staticmethod
    def _format_sql_value(value: DBValue) -> str:
        """Transfer Python value into SQL value for INSERT using"""
        if isinstance(value, str):
            escaped_value = (
                value.replace("\\", "\\\\").replace("'", "''")
            )
            return f"'{escaped_value}'"
        return str(value)
    
    def _initialize_database(self, entry: DBBenchEntry) -> None:
        """Initialize database and table accotding to the TableInfo of Case"""
        self.environment.execute(
            f"CREATE DATABASE `{entry.database_name}`"
        )

        column_definition_sql = ", ".join(
            f"`{column.name}` {column.type}"
            for column in entry.table_info.column_info_list
        )

        self.environment.execute(
            (
                f"CREATE TABLE `{entry.table_info.name}` "
                f"({column_definition_sql})"
            ),
            database=entry.database_name
        )

        column_name_sql = ", ".join(
            f"`{column.name}`"
            for column in entry.table_info.column_info_list
        )

        row_sql_list = []

        for row in entry.table_info.row_list:
            value_sql = ", ".join(
                self._format_sql_value(value) for value in row
            )
            row_sql_list.append(f"({value_sql})")

        row_list_sql = ", ".join(row_sql_list)

        self.environment.execute(
            (
                f"INSERT INTO `{entry.table_info.name}` "
                f"({column_name_sql}) "
                f"VALUES {row_list_sql}"
            ),
            database=entry.database_name
        )

    def get_sample_index_list(self) -> list[str]:
        return list(self.dataset.keys())
    
    def start_session(self, sample_index: str, previous_sample_text: str = "") -> Session:
        """Create Database and initialize a Session for a Case"""
        entry = self.dataset[sample_index]
        self.current_entry = entry
        self.current_sample_index = sample_index
        self.current_round = 0

        self._initialize_database(entry)

        session = Session(
            task_name=TaskName.DB_BENCH,
            sample_index=sample_index,
            sample_status=SampleStatus.RUNNING
        )

        for initial_message in self.initial_message_list:
            content = initial_message.content.replace(
                PSU_PLACEHOLDER, 
                previous_sample_text
            )
            session.chat_history.append(
                role=initial_message.role,
                content=content
            )

        session.chat_history.append(
            role=Role.USER,
            content=entry.build_task_instruction()
        )

        return session
    
    def interact(self, session: Session, agent_response: ChatMessage) -> None:
        """Analyze and execute one round response of Agent"""
        session.chat_history.append(
            role=agent_response.role,
            content=agent_response.content
        )

        if self.current_round >= self.max_round:
            session.sample_status = (
                SampleStatus.TASK_LIMIT_REACHED
            )
            session.finish_reason = (
                f"Task limit reached: {self.max_round}"
            )
            session.task_output = {"answer": None}
            return
        
        self.current_round += 1

        parsed_response = parse_dbbench_agent_response(
            agent_response=agent_response.content
        )

        if parsed_response.action == AgentAction.EXECUTE:
            sql = parsed_response.content
            if sql is None:
                raise RuntimeError(
                    "Parsed SQL content is missing."
                )
            
            try:
                result = self.environment.execute(
                    sql, database=self.current_entry.database_name
                )
            except Exception as error:
                session.sample_status = (
                    SampleStatus.TASK_ENVIRONMENT_ERROR
                )
                session.finish_reason = str(error)
                session.task_output = {"answer": None}
                return
            
            session.chat_history.append(
                role=Role.USER,
                content=str(result)
            )
            return
        
        if parsed_response.action == AgentAction.FINISH:
            session.sample_status = SampleStatus.COMPLETED
            session.task_output = {
                "answer": parsed_response.content
            }
            return
        
        session.sample_status = (
            SampleStatus.AGENT_VALIDATION_FAILED
        )
        session.finish_reason = parsed_response.finish_reason
        session.task_output = {"answer": None}
    
    def complete(
        self,
        session: Session,
    ) -> None:
        """评价任务结果并删除当前 Case 的数据库。"""

        if self.current_entry is None:
            raise RuntimeError("No active DBBench entry.")

        entry = self.current_entry

        try:
            correct = False
            answer_type = entry.answer_info.get_answer_type()

            if answer_type == AnswerType.DIRECT:
                expected_answer = entry.answer_info.direct

                if (
                    expected_answer is not None
                    and session.task_output is not None
                    and session.task_output["answer"] is not None
                ):
                    correct = validate_direct_answer(
                        agent_answer=session.task_output[
                            "answer"
                        ],
                        expected_answer=expected_answer,
                    )

            else:
                correct = validate_table_state(
                    environment=self.environment,
                    entry=entry,
                )

            if correct:
                session.evaluation_record.outcome = (
                    EvaluationOutcome.CORRECT
                )
            else:
                session.evaluation_record.outcome = (
                    EvaluationOutcome.INCORRECT
                )

        except Exception:
            session.evaluation_record.outcome = (
                EvaluationOutcome.UNKNOWN
            )
            raise

        finally:
            self.environment.execute(
                (
                    "DROP DATABASE IF EXISTS "
                    f"`{entry.database_name}`"
                )
            )
            self.current_entry = None
            self.current_sample_index = None
            self.current_round = 0
        
    def close(self) -> None:
        """Release MySQL docker of DBBench"""
        if self.current_entry is not None:
            self.environment.execute(
                (
                    "DROP DATABASE IF EXISTS "
                    f"`{self.current_entry.database_name}`"
                )
            )
        self.current_round = 0
        self.current_entry = None
        self.current_sample_index = None

        self.environment.close()
        
