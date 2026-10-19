from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from src.lifelong_learning.core.models import (
    EvaluationOutcome, Role, Session
)


class GSCExperience(BaseModel):
    """A successful experience which can be reused by GSC module"""
    session: Session
    sample_index: str
    question: str
    solution: str
    created_order: int


class GSCExperienceGroup(BaseModel):
    """A group of experiences and its prompt text"""
    group_index: int
    experience_sample_index_list: list[str]
    prompt_text: str


class GroupSelfConsistencyMemory:
    """Save successful experience, group the most recent one according to a fix number"""

    def __init__(self, group_count: int, sample_count_per_group: int) -> None:
        if group_count <= 0:
            raise ValueError("group_count must be positive")
        
        if sample_count_per_group <= 0:
            raise ValueError("sample_count_per_group must be positive")
        
        self.group_count = group_count
        self.sample_count_per_group = sample_count_per_group

        self.experience_list: list[GSCExperience] = []
        self._next_created_order = 0

    @property
    def experience_count(self) -> int:
        """Successful experience count in current library"""
        return len(self.experience_list)
    

    @property
    def sample_index_list(self) -> list[str]:
        """Return sample index in the order of entry into the experience database """
        return [experience.sample_index for experience in self.experience_list]
    
    @staticmethod
    def _get_role_name(role: Role) -> str:
        if role == Role.USER: return "user"
        return "assistant"
    
    @classmethod
    def _build_solution_text(cls, session: Session) -> str:
        """Extract the complete successful interaction trajectory after the problem"""
        trajectory_line_list = [
            (
                f"{cls._get_role_name(message.role)}: {message.content}"
            )
            for message in session.chat_history.messages[3:]
        ]

        return "\n".join(trajectory_line_list)
    
    def observe(self, session: Session) -> bool:
        accepted = (
            session.evaluation_record.outcome == EvaluationOutcome.CORRECT
        )

        if not accepted: return False

        message_list = session.chat_history.messages

        if len(message_list) >= 3:
            question = message_list[2].content
        else:
            question = ""

        experience = GSCExperience(
            session=session.model_copy(deep=True),
            sample_index=session.sample_index,
            question=question,
            solution=self._build_solution_text(session=session),
            created_order=self._next_created_order,
        )

        self.experience_list.append(experience)
        self._next_created_order += 1
        
        return True
    
    def select_recent_experience_list(self) -> list[GSCExperience]:
        """Select the most recent successful experiences that can be used in this round of GSC"""
        maximum_experience_count = (self.group_count * self.sample_count_per_group)

        recent_experience_list = list(reversed(self.experience_list))

        selected_experience_list = (recent_experience_list[:maximum_experience_count])

        return [experience.model_copy(deep=True) for experience in selected_experience_list]
    
    def build_experience_group_list(self) -> list[list[GSCExperience]]:
        """Group the most recent successful experience in a fixed order"""
        selected_experience_list = self.select_recent_experience_list()
        return [
            selected_experience_list[start_index: start_index + self.sample_count_per_group]
            for start_index in range(0, len(selected_experience_list), self.sample_count_per_group)
        ]
    
    @staticmethod
    def _build_single_experience_prompt(
        experience: GSCExperience,
    ) -> str:
        """Build the upstream prompt for a single experience."""
        return (
            "Before giving you the question that you need to solve, "
            "I will provide you with an experience. "
            "You can use the experience to help you solve the question.\n"
            "- Experience question:\n"
            f"{experience.question}\n"
            "- Experience solution:\n"
            f"{experience.solution}"
        )

    @staticmethod
    def _build_multiple_experience_prompt(
        experience_group: list[GSCExperience],
    ) -> str:
        """Build the upstream prompt for multiple experiences."""
        experience_text_list: list[str] = []

        for experience_index, experience in enumerate(
            experience_group,
            start=1,
        ):
            experience_text_list.append(
                f"Experience {experience_index}:\n"
                "- Experience question:\n"
                f"{experience.question}\n"
                "- Experience solution:\n"
                f"{experience.solution}"
            )

        concatenated_experience = "\n\n".join(
            experience_text_list
        )

        return (
            "Before giving you the question that you need to solve, "
            "I will provide you with same experience. "
            "You can use these experience to help you solve the question.\n"
            f"{concatenated_experience}"
        )
    
    def build_experience_group_prompt_list(self) -> list[GSCExperienceGroup]:
        """Build one independent experience prompt for each group."""
        experience_group_list = self.build_experience_group_list()

        group_prompt_list: list[GSCExperienceGroup] = []

        for group_index, experience_group in enumerate(experience_group_list):
            if self.sample_count_per_group == 1:
                prompt_text = self._build_single_experience_prompt(experience=experience_group[0])
            else:
                prompt_text = self._build_multiple_experience_prompt(experience_group=experience_group)

            group_prompt_list.append(
                GSCExperienceGroup(
                    group_index=group_index,
                    experience_sample_index_list=[experience.sample_index for experience in experience_group],
                    prompt_text=prompt_text
                )
            )
        return group_prompt_list
    
    def save_state(self, file_path: str | Path) -> None:
        """Save current experience to JSON"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as file:
            json.dump([
                experience.model_dump(mode="json")
                for experience in self.experience_list
            ], file, ensure_ascii=False, indent=2)