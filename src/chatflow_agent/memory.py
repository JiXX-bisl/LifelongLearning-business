from __future__ import annotations

import json
from copy import deepcopy
from typing import Any
from dataclasses import dataclass
from pathlib import Path


from src.chatflow_agent.runtime import BusinessRuntimeError, parse_learning_config
from src.project_paths import resolve_project_path
from src.lifelong_learning.core.models import (
    ChatHistory,
    EvaluationOutcome,
    EvaluationRecord,
    Role,
    SampleStatus,
    Session,
    TaskName
)


def build_business_experience(result: dict[str, Any], *, context_message_count: int) -> Session | None:
    evaluation = result.get("evaluation") or {}
    if (result.get("status") != "completed" or 
        evaluation.get("verdict") != "pass"):
        return None
    
    acutal_history = ChatHistory.model_validate(
        {"messages": result["messages"]}
    )
    messages = acutal_history.messages

    if not 0 <= context_message_count < len(messages):
        raise BusinessRuntimeError("经验转化失败：对话上下文位置无效。")
    
    question_message = messages[context_message_count]
    if (question_message.role != Role.USER or
        question_message.content != result["question"]):
        raise BusinessRuntimeError("经验转换失败：当前问题与轨迹不一致")
    
    question_text = result["question"]
    context_messages = messages[:context_message_count]

    if context_messages:
        question_text += (
            "\n\n本案例开始前的对话上下文：\n"
            + json.dumps(
                [
                    message.model_dump(mode="json")
                    for message in context_messages
                ],
                ensure_ascii=False,
            )
        )

    # 与GSC模块兼容：说明+确认+问题+实际执行轨迹
    experience_history = ChatHistory()
    experience_history.append(
        role=Role.USER,
        content="以下是一次业务查询案例。历史经验只能参考处理方法，案例中的企业数据不能作为新任务的事实依据。"
    )
    experience_history.append(Role.AGENT, "明白。")
    experience_history.append(Role.USER, question_text)

    experience_history.messages.extend(
        message.model_copy(deep=True)
        for message in messages[context_message_count + 1:]
    )

    return Session(
        task_name=TaskName.BUSINESS_QUERY,
        sample_index=f"{result['session_id']}:{result['turn_index']:04d}",
        sample_status=SampleStatus.COMPLETED,
        chat_history=experience_history,
        evaluation_record=EvaluationRecord(outcome=EvaluationOutcome.CORRECT),
        task_output=deepcopy({
            "question": result["question"],
            "answer": result["answer"],
            "operations": result["operations"],
            "evaluation": evaluation,
            "admission_basis": "business_reply_pass_v1",
            "source_record_path": result["record_path"],
            "baseline_id": result["baseline_id"],
            "reference_date": result["reference_date"],
        })
    )


@dataclass(frozen=True)
class BusinessMemorySnapshot:
    version: int
    sessions: tuple[Session, ...]


class BusinessExperienceStore:
    """单进程业务经验存储，按基线、实验和模式隔离。"""
    def __init__(self, output_dir: str | Path, *, baseline_id: str, learning: dict[str, Any]) -> None:
        self.learning = parse_learning_config(learning)
        self.baseline_id = baseline_id
        self.mode = self.learning["mode"]

        self.directory = (resolve_project_path(output_dir) / baseline_id / self.learning["experiment_id"] / self.mode)
        self.memory_path = self.directory / "memory.json"
        self._sessions: list[Session] = []

        if self.mode != "standard":
            self._load()
    
    @property
    def version(self) -> int:
        return len(self._sessions)
    
    def _check_session(self, session: Session) -> None:
        output = session.task_output or {}
        evaluation = output.get("evaluation") or {}
        if (
            session.task_name != TaskName.BUSINESS_QUERY
            or session.sample_status != SampleStatus.COMPLETED
            or session.evaluation_record.outcome != EvaluationOutcome.CORRECT
            or output.get("admission_basis") != "business_reply_pass_v1"
            or evaluation.get("verdict") != "pass"
            or output.get("baseline_id") != self.baseline_id
        ):
            raise BusinessRuntimeError(
                "经验内容与当前业务基线或准入条件不一致。"
            )
        
    def _load(self) -> None:
        try:
            text = self.memory_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError:
            raise BusinessRuntimeError("读取失败，检查文件权限。") from None
        
        try:
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError("经验文件必须保存Session列表。")
            
            sessions = [
                Session.model_validate(item) 
                for item in payload
            ]

        except (ValueError, TypeError):
            raise BusinessRuntimeError(
                "经验文件格式无效，未清空或覆盖原文件。"
            ) from None
        
        for session in sessions:
            self._check_session(session=session)

        ids = [session.sample_index for session in sessions]
        if len(ids) != len(set(ids)):
            raise BusinessRuntimeError("经验文件存在重复经验ID。")
        self._sessions = sessions

    def snapshot(self) -> BusinessMemorySnapshot:
        return BusinessMemorySnapshot(
            version=self.version,
            sessions=tuple(
                session.model_copy(deep=True)
                for session in self._sessions
            )
        )
    
    def observe(self, result: dict[str, Any], *, context_message_count: int) -> bool:
        if self.mode == "standard":
            return False
        
        session = build_business_experience(result=result, context_message_count=context_message_count)
        if session is None:
            return False
        
        self._check_session(session=session)

        if any(
            existing.sample_index == session.sample_index
            for existing in self._sessions
        ):
            return False
        
        updated_sessions = [*self._sessions, session]

        try:
            serizalized = json.dumps(
                [
                    item.model_dump(mode="json")
                    for item in updated_sessions
                ],
                ensure_ascii=False,
                indent=2
            )
            self.directory.mkdir(parents=True, exist_ok=True)
            self.memory_path.write_text(serizalized + "\n", encoding="utf-8")
        except (OSError, TypeError, ValueError):
            raise BusinessRuntimeError(
                "经验保存失败，请检查目录和磁盘空间。"
            ) from None
        
        self._sessions = updated_sessions
        return True
        
