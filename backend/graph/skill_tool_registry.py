from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from tools import get_all_tools


class SkillToolRegistry:
    """Skill → Tool 映射（方案第五、十一节）。

    第一版所有 Skill / NONE 默认绑定全量 4 个工具（terminal / python_repl /
    fetch_url / read_file）。接口保留按 Skill 裁剪能力，供后续扩展。
    """

    def __init__(self, base_dir: Path) -> None:
        self._tools_by_name: dict[str, BaseTool] = {
            tool.name: tool for tool in get_all_tools(base_dir)
        }
        # 预留：Skill ID -> 工具名列表；为空表示使用全量工具
        self._skill_overrides: dict[str, list[str]] = {}

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
