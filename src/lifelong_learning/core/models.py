from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    """Message sender"""
    USER = "user"
    AGENT = "agent"

class ChatMessage(BaseModel):
    """Message in a chat history"""
    role: Role
    content: str

class ChatHistory(BaseModel):
    """Save chats within agent and env under time sequence"""
    messages: list[ChatMessage] = Field(default_factory=list)

    def append(self, role:Role, content:str) -> None:
        self.messages.append(
            ChatMessage(
                role=role,
                content=content
            )
        )
    def last(self) -> ChatMessage:
        return self.messages[-1]
    
class TaskName(StrEnum):
    DB_BENCH = "db_bench"
    ENTERPRISE_QUERY_BENCH = "enterprise_query_bench"
    BUSINESS_QUERY = "business_query"

class AgentAction(StrEnum):
    """Action after agent decoder output"""
    EXECUTE = "execute"
    FINISH = "finish"
    INVALID = "invalid"

class ParsedAgentResponse(BaseModel):
    """Agent The structure of natural language output after parsing """   
    action: AgentAction
    content: str | None = None
    finish_reason: str | None = None

class SampleStatus(StrEnum):
    """A sample status during processing"""
    INITIAL = "initial"
    RUNNING = "running"
    COMPLETED = "completed"
    AGENT_VALIDATION_FAILED = "agent_validation_failed"
    TASK_LIMIT_REACHED = "task_limit_reached"
    TASK_ENVIRONMENT_ERROR = "task_environment_error"
    AGENT_CONTEXT_LIMIT = "agent_context_limit"
    AGENT_UNKNOWN_ERROR = "agent_unknown_error"
    TASK_UNKNOWN_ERROR = "task_unknown_error"

class EvaluationOutcome(StrEnum):
    """Evaluation after task completion"""
    UNSET = "unset"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"

class EvaluationRecord(BaseModel):
    """Evaluation after task comoletation"""
    outcome: EvaluationOutcome = EvaluationOutcome.UNSET

class Session(BaseModel):
    """A DBBench case full record from begin to end"""
    task_name: TaskName
    sample_index : str
    sample_status: SampleStatus = SampleStatus.INITIAL
    chat_history: ChatHistory = Field(default_factory=ChatHistory)
    finish_reason: str | None = None
    # DBBench only stores a final text answer. Enterprise Query Bench also
    # records SQL attempts and normalized result rows for B1 evaluation.
    task_output: dict[str, Any] | None = None
    evaluation_record: EvaluationRecord = Field(
        default_factory=EvaluationRecord
    )
