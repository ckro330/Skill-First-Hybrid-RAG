from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# ReAct 循环最大工具调用轮数（方案第九节：max_step = 5）
MAX_STEPS = 5

STATUS_RUNNING = "running"
STATUS_WAITING_TOOL = "waiting_tool"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class AgentState(TypedDict):
    """ReAct 主 Agent 的 LangGraph 状态（方案第七节）。"""

    messages: Annotated[list[AnyMessage], add_messages]
    current_skill: str | None
    task_context: dict[str, Any]
    conversation_context: str
    plan: list[str]
    status: str
    steps: int
    decision_reason: str
