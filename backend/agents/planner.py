from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agents.a2a import new_task_id
from agents.registry import DOWNSTREAM_AGENTS, get_agent


logger = logging.getLogger("planner")

DOMAIN_CODING = "coding"
DOMAIN_RAG = "rag"
DOMAIN_GENERAL = "general"

MAX_TASKS = 4


CODING_KEYWORDS: tuple[str, ...] = (
    "代码", "函数", "方法", "类", "模块", "接口", "api", "重构", "编译", "语法",
    "报错", "异常", "堆栈", "traceback", "bug", "崩溃", "crash", "调用链",
    "日志", "log", "测试", "单测", "单元测试", "pytest", "jest", "vitest",
    "git", "commit", "提交", "分支", "merge", "冲突", "diff", "仓库", "回滚",
    "依赖", "import", "脚本", "性能", "内存泄漏", "源码",
    ".py", ".ts", ".js", ".java", ".go", ".rs",
)

RAG_KEYWORDS: tuple[str, ...] = (
    "知识库", "文档", "资料", "pdf", "excel", "表格", "faq", "手册", "规范",
    "报告", "论文", "检索", "查资料", "报表", "白皮书", "财报", "说明书",
)


# 领域判定与任务拆解合并为一次 LLM 调用。
# 早先拆成两次时，两次的 system prompt 完全不同，第二次永远从第一个 token
# 起就是冷启动；合并后只有一次调用，且这份 system 跨请求恒定，可被缓存复用。
PLANNER_SYSTEM_PROMPT = """你是代码助手系统的任务规划器（Planner），只负责判断领域并（必要时）拆解任务，不直接回答用户。

## 一、领域判断

可选领域：
- coding：与本地代码仓库直接相关的工作，例如阅读/修改代码、排查报错、日志分析、运行或编写测试、git 操作（状态/差异/提交/分支）。
- rag：需要从本地知识库/文档/PDF/Excel/FAQ 中检索资料才能回答。
- general：与本地代码仓库、知识库都无关的普通问答或闲聊。

判断规则：
1. 只要任务需要读写、运行或检查本地代码仓库内容，就判为 coding。
2. 需要查询本地知识库文档资料才判为 rag。
3. 其余判为 general，不要为了使用工具而强行归为 coding。

## 二、任务拆解（仅当领域为 coding）

把用户请求拆解成 1~4 个可独立执行、可验证的子任务，并分配给最合适的下游 agent。

下游 agent 能力：
{agent_menu}

拆解规则：
1. 任务简单、只涉及单一环节时，只输出 1 个子任务（single agent）。
2. 任务复杂、跨多个环节时，拆成 2~4 个子任务，按执行顺序排列（multi agent）。
3. 子任务必须具体、可执行、可验收，写清目标、文件范围与期望产出，不要输出“分析一下”这类空泛描述。
4. 后置子任务若依赖前置结果，用 depends_on 填写其前置任务在数组中的序号（从 1 开始），没有依赖就用空数组。
5. 领域不是 coding 时，tasks 必须是空数组 []。

## 三、输出

只输出一个 JSON 对象，不要输出任何其他内容，不要用 markdown 代码块包裹：
{"domain": "coding|rag|general", "reason": "简短理由", "tasks": [{"title": "简短标题", "agent_id": "下游 agent id", "instruction": "给该 agent 的具体指令", "depends_on": []}]}"""


@dataclass
class TaskSpec:
    """Planner 拆解出的单个子任务。"""

    task_id: str
    title: str
    agent_id: str
    instruction: str
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "agent_id": self.agent_id,
            "instruction": self.instruction,
            "depends_on": list(self.depends_on),
        }


@dataclass
class PlanDecision:
    """Planner 的最终决策：领域 + （coding 时的）子任务列表。"""

    domain: str
    reason: str
    tasks: list[TaskSpec] = field(default_factory=list)
    multi_agent: bool = False
    stage: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "reason": self.reason,
            "stage": self.stage,
            "multi_agent": self.multi_agent,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).strip().strip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _agent_menu() -> str:
    lines = []
    for agent in DOWNSTREAM_AGENTS:
        skills = "、".join(agent.skills)
        lines.append(
            f"- {agent.agent_id}（{agent.name}）：{agent.description} 绑定 Skill：{skills}"
        )
    return "\n".join(lines)


def build_planner_prompt() -> str:
    """规划器的 system prompt：领域判断 + 任务拆解 + 输出契约，一次调用完成。

    这份文本跨请求恒定，因此可以被上下文缓存复用。
    """
    return PLANNER_SYSTEM_PROMPT.replace("{agent_menu}", _agent_menu())


def _history_snippet(history: list[dict[str, Any]], limit: int = 6) -> str:
    lines: list[str] = []
    for item in history[-limit:]:
        role = item.get("role")
        content = str(item.get("content", "") or "").strip().replace("\n", " ")
        if role in {"user", "assistant"} and content:
            lines.append(f"{role}: {content[:200]}")
    return "\n".join(lines)


class Planner:
    """外层 Planner：领域判定 + coding 任务拆解 + 分派建议。"""

    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self._model_builder: Callable[[], Any] | None = None

    def configure(self, base_dir: Path, model_builder: Callable[[], Any]) -> None:
        self.base_dir = base_dir
        self._model_builder = model_builder

    @property
    def is_ready(self) -> bool:
        return self.base_dir is not None and self._model_builder is not None

    async def decide(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> PlanDecision:
        history = history or []
        if not self.is_ready:
            return self._heuristic_decision(message, stage="unconfigured")

        try:
            return await self._plan_llm(message, history)
        except Exception:
            logger.exception("[Planner] 规划调用失败，回退启发式")
            return self._heuristic_decision(message, stage="fallback")

    async def _plan_llm(
        self, message: str, history: list[dict[str, Any]]
    ) -> PlanDecision:
        """一次调用同时拿到领域与（coding 时的）子任务。"""
        if self._model_builder is None:
            raise RuntimeError("Planner is not configured")
        parts = []
        snippet = _history_snippet(history)
        if snippet:
            parts.append(f"最近对话：\n{snippet}")
        parts.append(f"当前用户请求：\n{message}")
        response = await self._model_builder().ainvoke(
            [
                {"role": "system", "content": build_planner_prompt()},
                {"role": "user", "content": "\n\n".join(parts)},
            ]
        )
        raw = _stringify(response)
        data = extract_json_object(raw)
        if not data:
            raise ValueError("planner 未返回合法 JSON")
        domain = str(data.get("domain", "")).strip().lower()
        if domain not in {DOMAIN_CODING, DOMAIN_RAG, DOMAIN_GENERAL}:
            raise ValueError(f"planner 返回未知领域: {domain}")
        reason = str(data.get("reason", "")).strip() or "planner 领域判定"

        tasks: list[TaskSpec] = []
        if domain == DOMAIN_CODING:
            raw_tasks = data.get("tasks")
            if isinstance(raw_tasks, list):
                tasks = self._normalize_tasks(raw_tasks)
            # 模型没给出可用子任务时，用启发式兜底成单任务，保证下游一定有活干
            if not tasks:
                logger.warning("[Planner] coding 领域未给出有效子任务，回退单任务")
                tasks = [self._fallback_task(message)]

        return PlanDecision(
            domain=domain,
            reason=reason,
            tasks=tasks,
            multi_agent=len(tasks) > 1,
            stage="llm",
        )

    def _normalize_tasks(self, raw_tasks: list[Any]) -> list[TaskSpec]:
        """把 LLM 的原始拆解结果规范化为 TaskSpec。

        task_id 逐请求唯一，避免不同会话的子任务 id 在 A2A 总线上互相覆盖；
        depends_on 先按 LLM 给的数组序号（从 1 起）解析，再映射成真实 task_id。
        """
        tasks: list[TaskSpec] = []
        index_to_id: dict[int, str] = {}
        for index, raw in enumerate(raw_tasks[:MAX_TASKS], start=1):
            if not isinstance(raw, dict):
                continue
            agent = get_agent(str(raw.get("agent_id", "")).strip())
            if agent is None:
                continue
            instruction = str(raw.get("instruction", "")).strip()
            if not instruction:
                continue
            task_id = new_task_id("t")
            index_to_id[index] = task_id
            depends_on: list[str] = []
            raw_deps = raw.get("depends_on") or []
            if isinstance(raw_deps, list):
                for dep in raw_deps:
                    try:
                        dep_index = int(dep)
                    except (TypeError, ValueError):
                        continue
                    if 1 <= dep_index < index and dep_index in index_to_id:
                        depends_on.append(index_to_id[dep_index])
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    title=str(raw.get("title", "")).strip() or f"子任务 {index}",
                    agent_id=agent.agent_id,
                    instruction=instruction,
                    depends_on=sorted(set(depends_on)),
                )
            )
        return tasks

    def _heuristic_decision(self, message: str, *, stage: str) -> PlanDecision:
        lowered = message.lower()
        coding_hits = sum(1 for keyword in CODING_KEYWORDS if keyword in lowered)
        rag_hits = sum(1 for keyword in RAG_KEYWORDS if keyword in lowered)
        if coding_hits and coding_hits >= rag_hits:
            domain = DOMAIN_CODING
            reason = f"启发式命中 {coding_hits} 个编码关键词"
        elif rag_hits:
            domain = DOMAIN_RAG
            reason = f"启发式命中 {rag_hits} 个知识库关键词"
        else:
            domain = DOMAIN_GENERAL
            reason = "启发式未命中编码/知识库关键词"

        tasks: list[TaskSpec] = []
        if domain == DOMAIN_CODING:
            tasks = [self._fallback_task(message)]
        return PlanDecision(
            domain=domain,
            reason=reason,
            tasks=tasks,
            multi_agent=False,
            stage=stage,
        )

    def _fallback_task(self, message: str) -> TaskSpec:
        lowered = message.lower()
        if any(keyword in lowered for keyword in ("日志", "log", "堆栈", "traceback", "异常", "报错")):
            agent_id = "log-analysis-agent"
        elif any(keyword in lowered for keyword in ("测试", "单测", "pytest", "jest", "vitest", "覆盖率")):
            agent_id = "test-agent"
        # 不用裸 "git" 做关键词：像"不要执行 git 操作"这类否定表述会被误判成 git 任务
        elif any(
            keyword in lowered
            for keyword in ("提交", "commit", "分支", "merge", "合并", "冲突", "回滚", "暂存", "stash")
        ):
            agent_id = "git-agent"
        else:
            agent_id = "code-modify-agent"
        return TaskSpec(
            task_id=new_task_id("t"),
            title="完成用户编码请求",
            agent_id=agent_id,
            instruction=message,
            depends_on=[],
        )


def _stringify(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


planner = Planner()
