from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from agents.registry import get_agent, skill_tool_overrides
from tools import build_tool_catalog


class SkillToolRegistry:
    """Skill / Agent → Tool 映射（方案第五、十一节）。

    全量工具目录来自 tools.build_tool_catalog（基础 7 个 + 代码助手 10 个）。
    四个下游 Agent 的 Skill 按 agents.registry 的白名单裁剪；未登记的
    Skill 与 NONE 默认使用全量工具。
    """

    def __init__(self, base_dir: Path) -> None:
        self._tools_by_name: dict[str, BaseTool] = build_tool_catalog(base_dir)
        # Skill ID -> 工具名列表；由下游 Agent 定义汇总而来
        self._skill_overrides: dict[str, list[str]] = skill_tool_overrides()

    @property
    def all_tools(self) -> list[BaseTool]:
        return list(self._tools_by_name.values())

    def get_tools_for_skill(self, skill_id: str | None) -> list[BaseTool]:
        if not skill_id:
            return self.all_tools
        names = self._skill_overrides.get(skill_id)
        if not names:
            return self.all_tools
        return [self._tools_by_name[name] for name in names if name in self._tools_by_name]

    def get_tools_for_agent(self, agent_id: str | None) -> list[BaseTool]:
        """按 Agent 白名单取工具；未知 Agent 退回全量工具。"""
        agent = get_agent(agent_id)
        if agent is None:
            return self.all_tools
        return [
            self._tools_by_name[name]
            for name in agent.tools
            if name in self._tools_by_name
        ]
