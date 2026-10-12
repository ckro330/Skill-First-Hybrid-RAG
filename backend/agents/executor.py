from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from langchain_core.messages import HumanMessage

from agents.a2a import A2ABus, PLANNER_ID
from agents.planner import TaskSpec
from agents.registry import AgentSpec
from graph.agent_state import MAX_STEPS, STATUS_RUNNING
from graph.prompt_builder import build_system_prompt
from graph.react_agent import build_react_agent
from graph.skill_loader import SkillLoader
from graph.skill_tool_registry import SkillToolRegistry


logger = logging.getLogger("agent_executor")

MAX_SYSTEM_PROMPT_CHARS = 60_000


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


class AgentExecutor:
    """用现有 ReAct 链路执行单个下游 Agent 的子任务。"""

    def __init__(
        self,
        base_dir: Path,
        model_builder: Callable[[], Any],
        skill_loader: SkillLoader,
        tool_registry: SkillToolRegistry,
    ) -> None:
        self.base_dir = base_dir
        self._model_builder = model_builder
        self._skill_loader = skill_loader
        self._tool_registry = tool_registry

    def _skill_sections(self, agent: AgentSpec) -> list[str]:
        sections: list[str] = []
        for skill_id in agent.skills:
            loaded = self._skill_loader.load(skill_id)
            if loaded is None:
                sections.append(f"## Skill: {skill_id}\n（未找到对应 SKILL.md）")
                continue
            sections.append(
                f"## Skill: {skill_id}\n"
                f"Description: {loaded.description or '(无描述)'}\n\n"
                f"{loaded.instructions}"
            )
        return sections

    def _build_system_prompt(self, agent: AgentSpec) -> str:
        """子 Agent 的**静态** system prompt。

        只包含跨任务恒定的内容：共享基础提示 + 角色 + 技能 + 输出契约。
        任务指令、上游结论、对话上下文都是每轮变化的，放进 user 消息（见
        _build_task_message），否则同一 Agent 每次调用都从第一个 token 起失配。
        """
        parts: list[str] = [build_system_prompt(self.base_dir)]
        parts.append(
            "## Agent Role\n"
            f"Name: {agent.name} ({agent.agent_id})\n"
            f"Responsibility: {agent.description}\n"
            "你只负责本子任务，完成后向 planner 汇报，不要越权处理其他子任务。"
        )
        parts.extend(self._skill_sections(agent))
        parts.append(
            "## Output Contract\n"
            "完成后输出：结论 / 依据（文件路径与行号）/ 未完成或风险项。"
            "使用与用户请求相同的语言作答：用户用中文提问就用中文回答。"
            "不要伪造工具结果，不要修改与子任务无关的文件。"
        )
        content = "\n\n".join(parts)
        if len(content) > MAX_SYSTEM_PROMPT_CHARS:
            content = content[:MAX_SYSTEM_PROMPT_CHARS] + "\n...[truncated]"
        return content

    @staticmethod
    def _build_task_message(
        task: TaskSpec,
        upstream_text: str,
        conversation_context: str,
    ) -> str:
        """当轮动态内容，全部放进 user 消息，永远只追加在末尾。"""
        parts = [
            "## Assigned Sub-Task\n"
            f"Task ID: {task.task_id}\n"
            f"Title: {task.title}\n"
            f"Instruction: {task.instruction}"
        ]
        if upstream_text:
            parts.append(f"## Upstream Findings\n{upstream_text}")
        if conversation_context:
            parts.append(f"## Relevant Conversation Context\n{conversation_context}")
        return "\n\n".join(parts)

    async def astream(
        self,
        *,
        agent: AgentSpec,
        task: TaskSpec,
        upstream_text: str = "",
        conversation_context: str = "",
        bus: A2ABus | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        tools = self._tool_registry.get_tools_for_agent(agent.agent_id)
        system_prompt = self._build_system_prompt(agent)
        task_message = self._build_task_message(task, upstream_text, conversation_context)

        yield {
            "type": "agent_start",
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "title": task.title,
        }
        if bus is not None:
            message = bus.task(
                from_agent=PLANNER_ID,
                to_agent=agent.agent_id,
                task_id=task.task_id,
                content=task.instruction,
                payload={"title": task.title, "agent_name": agent.name},
            )
            yield {"type": "a2a", "message": message.to_dict()}

        graph = build_react_agent(
            self._model_builder(),
            tools,
            system_message_builder=lambda _state: system_prompt,
            max_steps=MAX_STEPS,
        )
        input_state = {
            "messages": [HumanMessage(content=task_message)],
            "current_skill": agent.skills[0] if agent.skills else None,
            "task_context": {
                "goal": task.instruction,
                "completed": [],
                "findings": [],
                "missing": [],
            },
            "conversation_context": conversation_context,
            "plan": [],
            "status": STATUS_RUNNING,
            "steps": 0,
            "decision_reason": "",
        }

        final_parts: list[str] = []
        last_ai_message = ""
        pending_tools: dict[str, dict[str, str]] = {}

        async for mode, payload in graph.astream(
            input_state,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") != "model":
                    continue
                text = _stringify_content(getattr(chunk, "content", ""))
                if text:
                    final_parts.append(text)
                    yield {"type": "token", "content": text}
                continue

            if mode != "updates":
                continue

            for update in payload.values():
                for agent_message in update.get("messages", []):
                    message_type = getattr(agent_message, "type", "")
                    tool_calls = getattr(agent_message, "tool_calls", []) or []

                    if message_type == "ai" and not tool_calls:
                        candidate = _stringify_content(getattr(agent_message, "content", ""))
                        if candidate:
                            last_ai_message = candidate

                    if tool_calls:
                        for tool_call in tool_calls:
                            call_id = str(tool_call.get("id") or tool_call.get("name"))
                            tool_name = str(tool_call.get("name", "tool"))
                            tool_args = tool_call.get("args", "")
                            if not isinstance(tool_args, str):
                                tool_args = json.dumps(tool_args, ensure_ascii=False)
                            pending_tools[call_id] = {
                                "tool": tool_name,
                                "input": str(tool_args),
                            }
                            yield {
                                "type": "tool_start",
                                "tool": tool_name,
                                "input": str(tool_args),
                            }

                    if message_type == "tool":
                        call_id = str(getattr(agent_message, "tool_call_id", ""))
                        pending = pending_tools.pop(
                            call_id,
                            {"tool": getattr(agent_message, "name", "tool"), "input": ""},
                        )
                        output = _stringify_content(getattr(agent_message, "content", ""))
                        yield {
                            "type": "tool_end",
                            "tool": pending["tool"],
                            "output": output,
                        }
                        # 这里刻意不发 new_response：一次子任务 = 一个气泡，
                        # 否则每轮工具调用都会另起一段，Agent 徽章与 A2A 记录只留在第一段。

        final_content = "".join(final_parts).strip() or last_ai_message.strip()
        logger.info(
            "[AgentExecutor] %s task=%s 完成，输出 %s 字",
            agent.agent_id,
            task.task_id,
            len(final_content),
        )

        if bus is not None:
            message = bus.result(
                from_agent=agent.agent_id,
                to_agent=PLANNER_ID,
                task_id=task.task_id,
                content=final_content,
                payload={"agent_name": agent.name},
            )
            yield {"type": "a2a", "message": message.to_dict()}

        yield {
            "type": "agent_result",
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "content": final_content,
        }
