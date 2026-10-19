from __future__ import annotations

import sqlglot
from pydantic import BaseModel
from sqlglot.errors import SqlglotError
from typing import Callable

from src.lifelong_learning.agents.gsc_candidate_generator import GSCCandidate
from src.lifelong_learning.core.models import AgentAction, ParsedAgentResponse
from src.lifelong_learning.tasks.db_bench.response_parser import parse_dbbench_agent_response


class GSCVoteResult(BaseModel):
    """Normalized actions, vote counts, and the current selection"""
    normalized_action_by_candidate_index: dict[int, str | None]
    action_to_candidate_index_list: dict[str, list[int]]
    invalid_candidate_index_list: list[int]
    maximum_vote_count: int
    highest_vote_action_list: list[str]
    selected_candidate_index: int | None
    selected_reason: str


def normalize_dbbench_action(response_text: str, 
                             *, response_parser: Callable[[str], ParsedAgentResponse] = parse_dbbench_agent_response) -> str | None:
    """Extract an action and create its comparison key"""
    parsed_response = response_parser(response_text)

    if parsed_response.action == AgentAction.EXECUTE:
        sql = parsed_response.content
        if sql is None: return None

        normalized_sql = sql

        try:
            statements = sqlglot.parse(sql, read="mysql")
            # Only normalize a signgle SQL statement.
            if (len(statements) == 1 and statements[0] is not None):
                normalized_sql = statements[0].sql(dialect="mysql")
        except SqlglotError:
            # If normalize fails, keep the original SQL
            normalized_sql = sql
        
        return (
            "Action: Operation\n"
            "```sql\n"
            f"{normalized_sql}\n"
            "```"
        )
    
    if parsed_response.action == AgentAction.FINISH:
        return (
            "Action: Answer\n"
            f"Final Answer: {parsed_response.content}"
        )
    
    return None


def vote_gsc_candidates(candidate_list: list[GSCCandidate], 
                        *, action_normalizer: Callable[[str], str | None] = normalize_dbbench_action) -> GSCVoteResult:
    """Count identical actions and select a unique highest-vote action"""
    normalized_action_by_candidate_index: dict[int, str | None] = {}
    action_to_candidate_index_list: dict[str, list[int]] = {}
    invalid_candidate_index_list: list[int] = []

    for candidate in candidate_list:
        # normalized_action = normalize_dbbench_action(response_text=candidate.agent_response.content)
        normalized_action = action_normalizer(candidate.agent_response.content)
        normalized_action_by_candidate_index[candidate.candidate_index] = normalized_action

        if normalized_action is None:
            invalid_candidate_index_list.append(candidate.candidate_index)
            continue
        
        action_to_candidate_index_list.setdefault(normalized_action, []).append(candidate.candidate_index)

    maximum_vote_count = max(
        (
            len(candidate_indices) for candidate_indices in action_to_candidate_index_list.values()
        ),default=0
    )
    highest_vote_action_list = [
        action for action, candidate_indices in action_to_candidate_index_list.items()
        if len(candidate_indices) == maximum_vote_count
    ]

    selected_candidate_index: int | None = None

    if not highest_vote_action_list: selected_reason = "no_valid_action"
    elif len(highest_vote_action_list) == 1:
        selected_action = highest_vote_action_list[0]
        selected_candidate_index = action_to_candidate_index_list[selected_action][0]
        selected_reason = "unique_highest_vote"
    else:
        selected_reason = "tie"
    
    return GSCVoteResult(
        normalized_action_by_candidate_index=normalized_action_by_candidate_index,
        action_to_candidate_index_list=action_to_candidate_index_list,
        invalid_candidate_index_list=invalid_candidate_index_list,
        maximum_vote_count=maximum_vote_count,
        highest_vote_action_list=highest_vote_action_list,
        selected_candidate_index=selected_candidate_index,
        selected_reason=selected_reason
    )

