from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

MAX_STEPS = 5

STATUS_RUNNING = "running"
STATUS_WAITING_TOOL = "waiting_tool"
STATUS_COMPLETED = "completed"


class AgentState(TypedDict):

    messages: Annotated[list[AnyMessage], add_messages]
    current_skill: str | None
    task_context: dict[str, Any]
    conversation_context: str
    plan: list[str]
    status: str
    steps: int
    decision_reason: str
