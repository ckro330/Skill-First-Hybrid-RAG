from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from graph.agent_state import (
    MAX_STEPS,
    STATUS_COMPLETED,
    STATUS_RUNNING,
    STATUS_WAITING_TOOL,
    AgentState,
)

MAX_STEPS_INSTRUCTION = (
    f"已达到最大工具调用轮数（{MAX_STEPS}）。请停止调用工具，"
    "直接基于当前已有信息给出最终答案。"
)

logger = logging.getLogger("react_agent")

SystemMessageBuilder = Callable[[AgentState], str]


def _has_tool_calls(message: AnyMessage | None) -> bool:
    return bool(isinstance(message, AIMessage) and (message.tool_calls or []))


def _route_after_model(state: AgentState) -> str:
    messages = state["messages"]
    if _has_tool_calls(messages[-1] if messages else None):
        return "tools"
    return END


def build_react_agent(
    model: Any,
    tools: list[Any],
    *,
    system_message_builder: SystemMessageBuilder,
    max_steps: int = MAX_STEPS,
):
    """构建 LangGraph ReAct Agent（方案第三、九节）。

    节点：model（bind_tools + 决策）↔ tools（ToolNode 执行工具）。
    max_steps 轮工具调用后强制结束循环。
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        steps = int(state.get("steps", 0) or 0)
        logger.debug(
            "[Agent] call_model steps=%s last_type=%s",
            steps,
            type(messages[-1]).__name__ if messages else None,
        )
        system_content = system_message_builder(state)
        prompt: list[AnyMessage] = [SystemMessage(content=system_content), *messages]

        if steps >= max_steps and messages and isinstance(messages[-1], ToolMessage):
            # 已用完全部工具轮数：注入强制结束指令，并剥离可能的 tool_calls
            force_prompt = [*prompt, SystemMessage(content=MAX_STEPS_INSTRUCTION)]
            response = model.bind_tools(tools).invoke(force_prompt)
            if _has_tool_calls(response):
                response = AIMessage(content=response.content or MAX_STEPS_INSTRUCTION)
            return {
                "messages": [response],
                "status": STATUS_COMPLETED,
                "decision_reason": f"达到 max_steps={max_steps}，强制结束工具循环。",
            }

        response = model.bind_tools(tools).invoke(prompt)
        if _has_tool_calls(response):
            names = ", ".join(call.get("name", "?") for call in response.tool_calls or [])
            return {
                "messages": [response],
                "steps": steps + 1,
                "status": STATUS_WAITING_TOOL,
                "decision_reason": f"调用工具：{names}",
            }
        return {
            "messages": [response],
            "status": STATUS_COMPLETED,
            "decision_reason": "模型直接给出最终答案。",
        }

    def call_tools(state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        for tool_call in last_message.tool_calls or []:
            name = str(tool_call.get("name", ""))
            call_id = str(tool_call.get("id", ""))
            args = tool_call.get("args", {})
            tool = tools_by_name.get(name)
            if tool is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool {name} 不存在。",
                        tool_call_id=call_id,
                        name=name,
                    )
                )
                continue
            try:
                result = tool.invoke(args)
                content = result if isinstance(result, str) else str(result)
            except Exception as exc:  # noqa: BLE001 - 工具异常必须转为可观测的 ToolMessage
                content = f"Tool execution failed: {exc}"
            tool_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=call_id,
                    name=name,
                )
            )
        return {
            "messages": tool_messages,
            "status": STATUS_RUNNING,
            "decision_reason": "工具执行完成，回到主 Agent 继续决策。",
        }

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("model")
    graph.add_conditional_edges(
        "model",
        _route_after_model,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "model")
    return graph.compile()
