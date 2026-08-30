from __future__ import annotations

from pathlib import Path
from typing import Any

from config import get_settings
from graph.prompt_builder import build_system_prompt
from graph.skill_loader import LoadedSkill


class AgentContextBuilder:
    """组装 Main Agent 的 SystemMessage 结构化小节（方案第六、十一节）。

    SystemMessage
    ├── Agent Role
    ├── Current Skill
    ├── SKILL.md Instructions
    ├── Current Task State
    ├── Current Plan（预留）
    └── Relevant Conversation Context
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def build_agent_role(self) -> str:
        return build_system_prompt(self.base_dir)

    @staticmethod
    def build_skill_section(skill: LoadedSkill | None) -> str:
        if skill is None:
            return ""
        return (
            "## Current Skill\n\n"
            f"Name: {skill.skill_id}\n"
            f"Description: {skill.description or '(无描述)'}\n\n"
            "## SKILL.md Instructions\n\n"
            f"{skill.instructions}"
        )

    @staticmethod
    def build_task_state_section(task_context: dict[str, Any] | None) -> str:
        if not task_context:
            return ""
        lines = ["## Current Task State"]
        goal = str(task_context.get("goal", "")).strip()
        if goal:
            lines.append(f"\n目标：\n{goal}")
        for label, key in (
            ("已完成：", "completed"),
            ("主要发现：", "findings"),
            ("当前缺少：", "missing"),
        ):
            values = task_context.get(key) or []
            if values:
                lines.append(f"\n{label}")
                lines.extend(f"- {value}" for value in values)
        return "\n".join(lines)

    @staticmethod
    def build_plan_section(plan: list[str] | None) -> str:
        if not plan:
            return ""
        lines = ["## Current Plan"]
        lines.extend(f"- {item}" for item in plan)
        return "\n".join(lines)

    @staticmethod
    def build_conversation_context_section(conversation_context: str) -> str:
        if not conversation_context:
            return ""
        return (
            "## Relevant Conversation Context\n\n"
            f"{conversation_context}\n\n"
            "仅在与当前任务相关时使用。"
        )

    def build_system_message_content(
        self,
        skill: LoadedSkill | None,
        task_context: dict[str, Any] | None,
        plan: list[str] | None,
        conversation_context: str,
    ) -> str:
        parts: list[str] = [self.build_agent_role()]
        skill_section = self.build_skill_section(skill)
        if skill_section:
            parts.append(skill_section)
        task_section = self.build_task_state_section(task_context)
        if task_section:
            parts.append(task_section)
        plan_section = self.build_plan_section(plan)
        if plan_section:
            parts.append(plan_section)
        conversation_section = self.build_conversation_context_section(conversation_context)
        if conversation_section:
            parts.append(conversation_section)

        settings = get_settings()
        content = "\n\n".join(parts)
        if len(content) > settings.component_char_limit:
            return content[: settings.component_char_limit] + "\n...[truncated]"
        return content
