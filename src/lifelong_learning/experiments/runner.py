from __future__ import annotations

import json
from pathlib import Path
from src.project_paths import resolve_project_path
from typing import Any, Sequence, Callable, Protocol

from src.lifelong_learning.agents.base import Agent
from src.lifelong_learning.core.models import (
    EvaluationOutcome, SampleStatus, Session
)
from src.lifelong_learning.memory.previous_sample_utilization import PreviousSampleUtilizationMemory
from src.lifelong_learning.agents.gsc_decision import GSCDecisionEngine, GSCInferenceRecord
from src.lifelong_learning.memory.group_self_consistency import GroupSelfConsistencyMemory


def save_json(file_path: Path, data: Any) -> None:
    """Transfer results to JSON format"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(
            data, file, ensure_ascii=False, indent=2
        )

class BenchmarkTask(Protocol):
    """Standard, PSU and GSC runner interface"""
    dataset: dict[str, Any]

    def start_session(self, sample_index: str, previous_sample_text: str="") -> Session:
        ...
    
    def complete(self, session: Session) -> None:
        ...

    def close(self) -> None:
        ...

MetricCalculator = Callable[
    [BenchmarkTask, Sequence[Session]], dict[str, Any]
]

def calculate_dbbench_metric(task: BenchmarkTask, session_list: Sequence[Session]) -> dict[str, Any]:
    """Calculate the overall metrics and each SQL Skill metric based on all sessions"""
    session_count = len(session_list)

    evaluation_outcome_count = {
        outcome.value: 0
        for outcome in EvaluationOutcome
    }
    sample_status_count = {
        status.value: 0
        for status in SampleStatus
    }

    for session in session_list:
        evaluation_outcome_count[session.evaluation_record.outcome.value] += 1
        sample_status_count[session.sample_status.value] += 1

    if session_count == 0:
        evaluation_outcome_rate = {
            key: 0.0 for key in evaluation_outcome_count
        }
        sample_status_rate = {
            key: 0.0 for key in sample_status_count
        }
    else:
        evaluation_outcome_rate = {
            key: count / session_count for key, count in evaluation_outcome_count.items()
        }
        sample_status_rate = {
            key: count / session_count for key, count in sample_status_count.items()
        }

    skill_metric: dict[str, dict[str, int | float]] = {}

    for session in session_list:
        entry = task.dataset[session.sample_index]
        correct = (
            session.evaluation_record.outcome == EvaluationOutcome.CORRECT
        )

        for skill in entry.skill_list:
            if skill not in skill_metric:
                skill_metric[skill] = {
                    "sample_count": 0,
                    "correct_count": 0,
                    "correct_rate": 0.0,
                }

            skill_metric[skill]["sample_count"] += 1
                
            if correct:
                skill_metric[skill]["correct_count"] += 1
    
    for metric in skill_metric.values():
        sample_count = int(metric["sample_count"])
        correct_count = int(metric["correct_count"])

        metric["correct_rate"] = (
            correct_count / sample_count 
            if sample_count > 0 else 0.0
        )

    correct_count = evaluation_outcome_count[EvaluationOutcome.CORRECT.value]

    return {
        "overall": {
            "session_count": session_count,
            "correct_count": correct_count,
            "correct_rate": (correct_count / session_count if session_count > 0 else 0.0),
            "evaluation_outcome_count": (evaluation_outcome_count),
            "evaluation_outcome_rate": (evaluation_outcome_rate),
            "sample_status_count": sample_status_count,
            "sample_status_rate": sample_status_rate
        },
        "skill": skill_metric,
    }

class DBBenchExperimentRunner:
    """Run Standard, PSU, or GSC experiments and save results."""

    def __init__(
        self,
        task: BenchmarkTask,
        agent: Agent,
        sample_order: Sequence[str | int],
        output_dir: str | Path,
        previous_sample_utilization_memory: (
            PreviousSampleUtilizationMemory | None
        ) = None,
        group_self_consistency_memory: (
            GroupSelfConsistencyMemory | None
        ) = None,
        metric_calculator: MetricCalculator | None = None,
    ) -> None:
        if (
            previous_sample_utilization_memory is not None
            and group_self_consistency_memory is not None
        ):
            raise ValueError(
                "PSU and GSC cannot be enabled together "
                "in the current reproduction."
            )

        self.task = task
        self.agent = agent
        self.sample_order = [
            str(sample_index)
            for sample_index in sample_order
        ]
        self.output_dir = resolve_project_path(output_dir)

        self.metric_calculator = (
            metric_calculator or calculate_dbbench_metric
        )

        self.previous_sample_utilization_memory = previous_sample_utilization_memory
        self.group_self_consistency_memory = group_self_consistency_memory

        self.gsc_decision_engine: GSCDecisionEngine | None = None

        if group_self_consistency_memory is not None:
            self.gsc_decision_engine = GSCDecisionEngine(
                agent=agent,
                memory=group_self_consistency_memory,
            )

    def run(self) -> list[Session]:
        """Execute samples sequentially and save their results."""
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        session_list: list[Session] = []
        gsc_record_list: list[GSCInferenceRecord] = []

        psu_memory = self.previous_sample_utilization_memory
        gsc_memory = self.group_self_consistency_memory
        gsc_engine = self.gsc_decision_engine

        psu_state_path = (
            self.output_dir / "utilized_session_list.json"
        )
        gsc_state_path = (
            self.output_dir / "gsc_experience_list.json"
        )
        gsc_record_path = (
            self.output_dir / "gsc_inference_record_list.json"
        )

        try:
            for position, sample_index in enumerate(
                self.sample_order,
                start=1,
            ):
                # PSU injects its history into the initial prompt.
                # GSC starts from the Standard dialogue.
                previous_sample_text = (
                    ""
                    if psu_memory is None
                    else psu_memory.build_prompt_text()
                )

                session = self.task.start_session(
                    sample_index=sample_index,
                    previous_sample_text=previous_sample_text,
                )

                round_index = 0

                while session.sample_status == SampleStatus.RUNNING:
                    round_index += 1

                    if gsc_engine is None:
                        # Preserve the Standard and PSU inference path.
                        agent_response = self.agent.inference(
                            session.chat_history
                        )

                        if psu_memory is not None:
                            agent_response = (
                                psu_memory.sanitize_agent_response(
                                    agent_response=agent_response
                                )
                            )

                    else:
                        # The engine already generates the base candidate.
                        # Do not call the ordinary Agent again here.
                        agent_response, record = gsc_engine.decide(
                            sample_index=sample_index,
                            round_index=round_index,
                            current_chat_history=session.chat_history,
                        )

                        gsc_record_list.append(record)

                        save_json(
                            gsc_record_path,
                            [
                                item.model_dump(mode="json")
                                for item in gsc_record_list
                            ],
                        )

                        print(
                            f"[GSC] sample={sample_index}, "
                            f"round={round_index}, "
                            f"candidates={len(record.candidate_list)}, "
                            "selected="
                            f"{record.selection_result.selected_candidate_index}, "
                            "reason="
                            f"{record.selection_result.selected_reason}, "
                            "calls="
                            f"{record.logical_inference_call_count}"
                        )

                    # Only the selected original response reaches the task.
                    self.task.interact(
                        session=session,
                        agent_response=agent_response,
                    )

                # Evaluate the completed task before admitting experience.
                self.task.complete(session=session)

                memory_accepted: bool | None = None

                if psu_memory is not None:
                    memory_accepted = psu_memory.observe(
                        session=session
                    )
                    psu_memory.save_state(psu_state_path)

                elif gsc_memory is not None:
                    memory_accepted = gsc_memory.observe(
                        session=session
                    )
                    gsc_memory.save_state(gsc_state_path)

                session_list.append(session)

                save_json(
                    self.output_dir / "runs.json",
                    [
                        saved_session.model_dump(mode="json")
                        for saved_session in session_list
                    ],
                )

                progress_text = (
                    f"[{position}/{len(self.sample_order)}] "
                    f"sample={sample_index}, "
                    f"status={session.sample_status.value}, "
                    "outcome="
                    f"{session.evaluation_record.outcome.value}"
                )

                if psu_memory is not None:
                    progress_text += (
                        f", memory_accepted={memory_accepted}, "
                        f"memory_sample={psu_memory.sample_index_list}"
                    )

                elif gsc_memory is not None:
                    progress_text += (
                        f", memory_accepted={memory_accepted}, "
                        f"gsc_memory_count={gsc_memory.experience_count}"
                    )

                print(progress_text)

            metric = self.metric_calculator(
                task=self.task,
                session_list=session_list,
            )

            save_json(
                self.output_dir / "metric.json",
                metric,
            )

            return session_list

        finally:
            self.task.close()