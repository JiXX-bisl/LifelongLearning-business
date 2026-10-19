from __future__ import annotations

from pydantic import BaseModel

from src.lifelong_learning.agents.base import Agent
from src.lifelong_learning.core.models import ChatHistory, ChatMessage
from src.lifelong_learning.prompts.gsc_prompt_builder import GSCGroupInput


class GSCCandidate(BaseModel):
    """One candidate response generated for the current decision"""
    candidate_index: int
    experience_group_index: int | None
    experience_sample_index_list: list[str]
    agent_response: ChatMessage


class GSCCandidateGenerator:
    """Generate a base response and experience-conditioned responses"""
    def __init__(self, agent: Agent, *, verbose: bool = True) -> None:
        self.agent = agent
        self.verbose = verbose

    def generate(self, current_chat_history: ChatHistory, group_input_list: list[GSCGroupInput]) -> list[GSCCandidate]:
        candidate_list: list[GSCCandidate] = []

        # Candidate 0: the original input without added experience.
        if self.verbose:
            print("[GSC] Generagting candidate 0: no added experience")

        base_response = self.agent.inference(current_chat_history.model_copy(deep=True))

        candidate_list.append(
            GSCCandidate(
                candidate_index=0,
                experience_group_index=None,
                experience_sample_index_list=[],
                agent_response=base_response
            )
        )

        # Otrher candidates: one response per experience group
        for group_input in group_input_list:
            candidate_index = len(candidate_list)
            if self.verbose:
                print(
                    f"[GSC] Generating candidate {candidate_index}: "
                    f"group={group_input.group_index}, "
                    f"experiences={group_input.experience_sample_index_list}"
                )
            group_response = self.agent.inference(group_input.chat_history.model_copy(deep=True))
            candidate_list.append(
                GSCCandidate(
                    candidate_index=candidate_index,
                    experience_group_index=group_input.group_index,
                    experience_sample_index_list=list(group_input.experience_sample_index_list),
                    agent_response=group_response
                )
            )
        return candidate_list
    