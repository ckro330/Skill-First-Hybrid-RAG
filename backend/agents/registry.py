from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    """下游 Agent 的静态定义：负责的领域、绑定的 Skill 与可用工具。"""

    agent_id: str
    name: str
    description: str
    skills: tuple[str, ...]
    tools: tuple[str, ...]
    domain: str = "coding"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "skills": list(self.skills),
            "tools": list(self.tools),
            "domain": self.domain,
        }


LOG_ANALYSIS_AGENT = AgentSpec(
    agent_id="log-analysis-agent",
    name="日志分析 Agent",
    description=(
        "负责读取、过滤与聚合日志文件，定位报错时间线、异常堆栈、错误频次与根因线索，"
        "输出可被其他 agent 直接使用的结构化发现。"
    ),
    skills=("log-analysis",),
    tools=("read_file", "grep", "log_search", "log_stats", "terminal", "python_repl"),
)


CODE_MODIFY_AGENT = AgentSpec(
    agent_id="code-modify-agent",
    name="代码修改 Agent",
    description=(
        "负责按最小化原则定位并修改代码：先读懂上下文，再精确替换，"
        "改完做语法与静态自检，输出改动清单与影响范围。"
    ),
    skills=("code-modification",),
    tools=(
        "read_file",
        "grep",
        "edit_file",
        "write_file",
        "terminal",
        "python_repl",
        # code-modification/SKILL.md 明确要求改完自检与复查差异，故允许这两个只读/验证工具
        "git_diff",
        "run_tests",
    ),
)


TEST_AGENT = AgentSpec(
    agent_id="test-agent",
    name="测试 Agent",
    description=(
        "负责运行既有测试与编写新测试：复现问题、验证改动、回归检查，"
        "输出通过/失败结果与失败定位。"
    ),
    skills=("test-execution", "test-authoring"),
    tools=(
        "read_file",
        "grep",
        "write_file",
        "edit_file",
        "terminal",
        "python_repl",
        "run_tests",
        # test-execution/SKILL.md 要求用差异确认失败是否由本次改动引入
        "git_diff",
    ),
)


GIT_AGENT = AgentSpec(
    agent_id="git-agent",
    name="Git 管理 Agent",
    description=(
        "负责查看仓库状态、差异、历史与分支，并在明确授权下暂存与提交改动；"
        "只提交不 push，破坏性操作需先确认。"
    ),
    skills=("git-management",),
    tools=(
        "read_file",
        "grep",
        "terminal",
        "git_status",
        "git_diff",
        "git_log",
        "git_add",
        "git_commit",
        "git_branch",
    ),
)


DOWNSTREAM_AGENTS: tuple[AgentSpec, ...] = (
    LOG_ANALYSIS_AGENT,
    CODE_MODIFY_AGENT,
    TEST_AGENT,
    GIT_AGENT,
)

AGENTS_BY_ID: dict[str, AgentSpec] = {agent.agent_id: agent for agent in DOWNSTREAM_AGENTS}

AGENT_ALIASES: dict[str, str] = {
    "log": LOG_ANALYSIS_AGENT.agent_id,
    "log-analysis": LOG_ANALYSIS_AGENT.agent_id,
    "code": CODE_MODIFY_AGENT.agent_id,
    "code-modify": CODE_MODIFY_AGENT.agent_id,
    "test": TEST_AGENT.agent_id,
    "git": GIT_AGENT.agent_id,
}


def get_agent(agent_id: str | None) -> AgentSpec | None:
    if not agent_id:
        return None
    key = agent_id.strip()
    return AGENTS_BY_ID.get(key) or AGENTS_BY_ID.get(AGENT_ALIASES.get(key, ""))


def list_agents() -> list[AgentSpec]:
    return list(DOWNSTREAM_AGENTS)


def skill_tool_overrides() -> dict[str, list[str]]:
    """按 Skill 汇总每个 Agent 的工具白名单，供 SkillToolRegistry 使用。"""

    overrides: dict[str, list[str]] = {}
    for agent in DOWNSTREAM_AGENTS:
        for skill_id in agent.skills:
            merged = list(overrides.get(skill_id, []))
            for tool_name in agent.tools:
                if tool_name not in merged:
                    merged.append(tool_name)
            overrides[skill_id] = merged
    return overrides
