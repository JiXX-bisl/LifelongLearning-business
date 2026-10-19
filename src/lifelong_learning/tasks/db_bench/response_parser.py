from __future__ import annotations

import re

from src.lifelong_learning.core.models import (
    AgentAction, ParsedAgentResponse
)

def parse_dbbench_agent_response(agent_response: str) -> ParsedAgentResponse:
    """Transfer DBBench Agent language decoder results into structural action"""
    action_match = re.search(
        r"Action: (Operation|Answer)",
        agent_response
    )

    if action_match is None:
        return ParsedAgentResponse(
            action=AgentAction.INVALID,
            finish_reason=(
                'Cannot find Action: Operation or '
                '"Action: Answer" in the Agent response'
            )
        )
    
    action_name = action_match.group(1)
    if action_name == "Operation":
        sql_match = re.search(
            r"```sql\n([\s\S]*?)\n```", agent_response
        )
        # ```sql: SQL Markdown beginning
        # \n: line feed
        # [\s\S]*?: extract all the characters in the middle 
        # ```: Ending of Markdown

        if sql_match is None:
            return ParsedAgentResponse(
                action=AgentAction.INVALID,
                finish_reason=(
                    "Cannot find a valid SQL markdown block "
                    "in the Agent response."
                ),
            )
        
        sql = (
            sql_match.group(1).strip().replace("\n", " ")
        )
        return ParsedAgentResponse(
            action=AgentAction.EXECUTE,
            content=sql
        )
    answer_match = re.search(
        r"\nFinal Answer:(.*)", agent_response
    )

    if answer_match is None:
        return ParsedAgentResponse(
            action=AgentAction.INVALID,
            finish_reason=(
                'Cannot find "Final Answer:" in the Agent Response'
            ),
        )
    final_answer = answer_match.group(1).strip()

    return ParsedAgentResponse(
        action=AgentAction.FINISH,
        content=final_answer
    )