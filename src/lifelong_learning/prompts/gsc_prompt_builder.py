from __future__ import annotations

from pydantic import BaseModel

from src.lifelong_learning.core.models import ChatHistory, ChatMessage, Role
from src.lifelong_learning.memory.group_self_consistency import GSCExperienceGroup



class GSCGroupInput(BaseModel):
    """Onde experience group's independent iinference input"""

    group_index: int
    experience_sample_index_list: list[str]
    chat_history: ChatHistory


def build_gsc_group_input_list(current_chat_history: ChatHistory, group_prompt_list: list[GSCExperienceGroup]) -> list[GSCGroupInput]:
    """Create independent dialogue inputs for all experience groups."""
    message_list = current_chat_history.messages

    if len(message_list) < 3:
        raise ValueError("Gsc requires task instructions, acknowledgement, and the current question.")
    
    if message_list[0].role != Role.USER:
        raise ValueError("The first DBBench message must be a user message.")
    
    if message_list[-1].role != Role.USER:
        raise ValueError("GSC input must end with the current question or an environment observations.")
    
    original_initial_prompt = message_list[0].content.rstrip()
    
    group_input_list: list[GSCGroupInput] = []

    for group_prompt in group_prompt_list:
        group_chat_history = current_chat_history.model_copy(deep=True)
        first_user_prompt = (
            f"{original_initial_prompt}\n\n"
            f"{group_prompt.prompt_text}"
        )
        group_chat_history.messages[0] = ChatMessage(role=Role.USER, content=first_user_prompt)
        group_input_list.append(
            GSCGroupInput(
                group_index=group_prompt.group_index,
                experience_sample_index_list=list(group_prompt.experience_sample_index_list),
                chat_history=group_chat_history
            )
        )
    return group_input_list