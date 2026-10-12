from __future__ import annotations

from pathlib import Path

from config import get_settings
from graph.prompt_builder import build_system_prompt
from graph.skill_loader import LoadedSkill


class AgentContextBuilder:
    """组装 Main Agent 的 SystemMessage（方案第六、十一节）。

    SystemMessage 里**只允许放跨请求恒定不变的内容**，顺序固定：

    ├── Agent Role
    ├── Current Skill
    ├── SKILL.md Instructions

    为什么这么严格：上下文缓存按前缀精确匹配，system 排在请求最前面，
    只要里面有一个字每轮变化，它后面的整段对话历史都会一起失效。
    所以任务目标、计划、记忆召回、对话摘要这类动态内容一律不进 system，
    改由 agent.py 拼进"当轮 user 消息"，永远只追加在末尾。
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
    def build_conversation_context_section(conversation_context: str) -> str:
        if not conversation_context:
            return ""
        return (
            "## Relevant Conversation Context\n\n"
            f"{conversation_context}\n\n"
            "仅在与当前任务相关时使用。"
        )

    def build_system_content(self, skill: LoadedSkill | None) -> str:
        """静态 system prompt：同一 skill 下必须逐字节一致。"""
        parts: list[str] = [self.build_agent_role()]
        skill_section = self.build_skill_section(skill)
        if skill_section:
            parts.append(skill_section)

        settings = get_settings()
        content = "\n\n".join(parts)
        if len(content) > settings.component_char_limit:
            return content[: settings.component_char_limit] + "\n...[truncated]"
        return content
