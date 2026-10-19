from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.lifelong_learning.agents.base import Agent
from src.lifelong_learning.agents.gsc_voting import GSCVoteResult
from src.lifelong_learning.core.models import ChatHistory, ChatMessage, Role


class GSCSelectionResult(BaseModel):
    """The final candidate choice and its selection process"""
    selected_candidate_index: int
    selected_reason: str
    used_fallback: bool = False
    removed_base_action: bool = False

    selection_action_list: list[str] = Field(default_factory=list)

    selector_called: bool = False
    selector_prompt: str | None = None
    selector_response: str | None =None
    selector_error: str | None = None


class GSCActionSelector:
    """Resolve ties and return a candidate index"""

    def __init__(self, agent: Agent, *, verbose: bool = True) -> None:
        self.agent = agent
        self.verbose = verbose

    @staticmethod
    def _build_selection_prompt(original_user_content: str, action_list: list[str]) -> str:
        action_selection_text = "".join(
            f"{chr(65 + index)}. {action}\n"
            for index, action in enumerate(action_list)
        )
        return (
            "The response of your last action is: "
            f"{original_user_content}. "
            "Now, Please determine the best action for me "
            "to take from the following options:\n"
            f"{action_selection_text}"
            "Analyze the choices carefully, explain your "
            "reasoning, and then select the best option.\n"
            "Format your selection as follows "
            "(e.g., if choosing option A):\n"
            "Selection: A\n"
        )

    def select(self, current_chat_history: ChatHistory, vote_result: GSCVoteResult) -> GSCSelectionResult:
        if (0 not in vote_result.normalized_action_by_candidate_index):
            raise ValueError("GSC selection requires base candidate 0.")
        
        # Default fallback: preserve the base candidate.
        result = GSCSelectionResult(
            selected_candidate_index=0,
            selected_reason="fallback_no_valid_action", 
            used_fallback=True
        )

        # A unique highest-vote action was already selected.
        if vote_result.selected_candidate_index is not None:
            result.selected_candidate_index = vote_result.selected_candidate_index
            result.selected_reason = "unique_highest_vote"
            result.used_fallback = False
            return result
        
        # No valid action: return the original base response
        if not vote_result.highest_vote_action_list: return result

        # Copy the options; do not modify the original vote record
        action_list = list(vote_result.highest_vote_action_list)
        base_action = vote_result.normalized_action_by_candidate_index[0]

        # Follow the upstream tie-handling rule.
        if base_action in action_list:
            action_list.remove(base_action)
            result.removed_base_action = True

        result.selection_action_list = action_list

        # Removing the base action may resolve the tie:
        if len(action_list) == 1:
            selected_action = action_list[0]
            result.selected_candidate_index = vote_result.action_to_candidate_index_list[selected_action][0]
            result.selected_reason = "tie_resolved_after_removing_base_action"
            result.used_fallback = False
            return result
        
        # This version uses single-letter optiion labels A-Z
        if len(action_list) > 26:
            raise ValueError("This selector supports at most 26 action options")
        
        if (not current_chat_history.messages or current_chat_history.messages[-1].role != Role.USER):
            raise ValueError("Selection input must end with a user message")
        
        selection_history = current_chat_history.model_copy(deep=True)
        selection_prompt = self._build_selection_prompt(
            original_user_content=selection_history.messages[-1].content, 
            action_list=action_list
        )
        selection_history.messages[-1] = ChatMessage(
            role=Role.USER, content=selection_prompt
        )

        result.selector_prompt = selection_prompt
        result.selector_called = True
        result.selected_reason = "fallback_invalid_selection"
        
        if self.verbose:
            print(
                "[GSC] Resolving a tie with the LLM: "
                f"{len(action_list)} action options"
            )

        try: 
            selection_response = self.agent.inference(selection_history)
        except Exception as error:
            result.selected_reason = "fallback_selector_error"
            result.selector_error = f"{type(error).__name__}: {error}"
            return result
        
        # Read the actual model response
        result.selector_response = selection_response.content
        selection_matches = re.findall(
            r"^\s*Selection:\s*([A-Z])\s*$", 
            selection_response.content,
            flags=re.MULTILINE
        )

        if len(selection_matches) != 1:
            result.selector_error = "Expected exactlly one line such as Selection: A."
            return result
        
        selected_option_index = ord(selection_matches[0]) - ord("A")

        if not (0 <= selected_option_index < len(action_list)):
            result.selector_error = "The selected option is outside the available range."
            return result
        
        selected_action = action_list[selected_option_index]
        result.selected_candidate_index = vote_result.action_to_candidate_index_list[selected_action][0]

        result.selected_reason = "selected_by_llm"
        result.used_fallback = False
        
        return result