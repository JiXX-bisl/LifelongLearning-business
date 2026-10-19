from __future__ import annotations

from pydantic import BaseModel
from typing import Callable

from src.lifelong_learning.agents.base import Agent
from src.lifelong_learning.agents.gsc_candidate_generator import (
    GSCCandidate,
    GSCCandidateGenerator,
)
from src.lifelong_learning.agents.gsc_selector import (
    GSCActionSelector,
    GSCSelectionResult,
)
from src.lifelong_learning.core.models import (
    ChatHistory,
    ChatMessage,
)
from src.lifelong_learning.memory.group_self_consistency import (
    GSCExperienceGroup,
    GroupSelfConsistencyMemory,
)
from src.lifelong_learning.prompts.gsc_prompt_builder import (
    build_gsc_group_input_list,
)
from src.lifelong_learning.agents.gsc_voting import (
    GSCVoteResult, normalize_dbbench_action, vote_gsc_candidates
)


class GSCInferenceRecord(BaseModel):
    """The complex GSC decision record for one interactioni round."""

    sample_index: str
    round_index: int
    group_prompt_list: list[GSCExperienceGroup]
    candidate_list: list[GSCCandidate]
    vote_result: GSCVoteResult
    selection_result: GSCSelectionResult
    logical_inference_call_count: int


class GSCDecisionEngine:
    """Combine memory, candidate generation, voting, and selection"""
    def __init__(self, agent: Agent, memory: GroupSelfConsistencyMemory, 
                 *, action_normalizer: Callable[[str], str | None] = normalize_dbbench_action, 
                 selector_agent: Agent | None = None, verbose: bool = True) -> None:
        self.memory = memory
        self.action_normalizer = action_normalizer

        self.candidate_generator = GSCCandidateGenerator(agent=agent, verbose=verbose)
        self.action_selector = GSCActionSelector(agent=selector_agent if selector_agent is not None else agent, verbose=verbose)

    def decide(self, sample_index: str | int, round_index: int, current_chat_history: ChatHistory) -> tuple[ChatMessage, GSCInferenceRecord]:
        # 1. Select experiences and build group prompts.
        group_prompt_list = self.memory.build_experience_group_prompt_list()

        # 2. Construct independent group inputs.
        group_input_list = build_gsc_group_input_list(
            current_chat_history=current_chat_history, 
            group_prompt_list=group_prompt_list
        )

        # 3. Generate the base candidate and group candidates.
        candidate_list = self.candidate_generator.generate(
            current_chat_history=current_chat_history,
            group_input_list=group_input_list
        )

        # 4. Normalize actions and count votes.
        vote_result = vote_gsc_candidates(candidate_list=candidate_list, action_normalizer=self.action_normalizer)

        # 5. Resolve ties or use the configured fallback.
        selection_result = self.action_selector.select(
            current_chat_history=current_chat_history,
            vote_result=vote_result,
        )

        # 6. Retrieve the selected original response.
        candidate_by_index = {
            candidate.candidate_index: candidate
            for candidate in candidate_list
        }
        selected_candidate = candidate_by_index[selection_result.selected_candidate_index]
        agent_response = (
            selected_candidate.agent_response.model_copy(deep=True)
        )
        logical_inference_call_count = len(candidate_list) + int(selection_result.selector_called)

        # 7. Return the process record for later saving
        record = GSCInferenceRecord(
            sample_index=str(sample_index),
            round_index=round_index,
            group_prompt_list = group_prompt_list,
            candidate_list=candidate_list,
            vote_result=vote_result,
            selection_result=selection_result,
            logical_inference_call_count=logical_inference_call_count
        )


        return agent_response, record