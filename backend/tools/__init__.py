from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool

from tools.add_longterm_memory_tool import AddLongtermMemoryTool
from tools.edit_file_tool import EditFileTool
from tools.fetch_url_tool import FetchURLTool
from tools.git_add_tool import GitAddTool
from tools.git_branch_tool import GitBranchTool
from tools.git_commit_tool import GitCommitTool
from tools.git_diff_tool import GitDiffTool
from tools.git_log_tool import GitLogTool
from tools.git_status_tool import GitStatusTool
from tools.grep_tool import GrepTool
from tools.log_search_tool import LogSearchTool
from tools.log_stats_tool import LogStatsTool
from tools.python_repl_tool import PythonReplTool
from tools.read_file_tool import ReadFileTool
from tools.run_tests_tool import RunTestsTool
from tools.terminal_tool import TerminalTool
from tools.write_file_tool import WriteFileTool


def get_all_tools(base_dir: Path) -> list[BaseTool]:
    """基础工具集：对话主链路默认绑定的 7 个工具。"""
    return [
        TerminalTool(root_dir=base_dir),
        PythonReplTool(root_dir=base_dir),
        FetchURLTool(),
        ReadFileTool(root_dir=base_dir),
        WriteFileTool(root_dir=base_dir),
        EditFileTool(root_dir=base_dir),
        AddLongtermMemoryTool(),
    ]


def get_agent_tools(base_dir: Path) -> list[BaseTool]:
    """代码助手下游 Agent 的工具集：日志分析 / 代码搜索 / 测试 / git。"""
    return [
        GrepTool(root_dir=base_dir),
        LogSearchTool(root_dir=base_dir),
        LogStatsTool(root_dir=base_dir),
        RunTestsTool(root_dir=base_dir),
        GitStatusTool(root_dir=base_dir),
        GitDiffTool(root_dir=base_dir),
        GitLogTool(root_dir=base_dir),
        GitAddTool(root_dir=base_dir),
        GitCommitTool(root_dir=base_dir),
        GitBranchTool(root_dir=base_dir),
    ]


def build_tool_catalog(base_dir: Path) -> dict[str, BaseTool]:
    """name -> tool 的全量目录，供 Skill / Agent 级工具裁剪使用。"""
    catalog: dict[str, BaseTool] = {}
    for tool in [*get_all_tools(base_dir), *get_agent_tools(base_dir)]:
        catalog[tool.name] = tool
    return catalog
