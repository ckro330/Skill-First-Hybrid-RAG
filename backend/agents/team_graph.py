from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from agents.a2a import A2ABus, PLANNER_ID
from agents.executor import AgentExecutor
from agents.planner import TaskSpec
from agents.registry import DOWNSTREAM_AGENTS, get_agent


logger = logging.getLogger("team_graph")

MAX_UPSTREAM_CHARS = 6000
MAX_RESULT_CHARS = 4000

# 内部事件：只用于把汇总结果传回 orchestrator，不向前端透出
TEAM_FINAL_EVENT = "_team_final"

AGGREGATE_PROMPT = """你是代码助手系统的汇总器。下面是 planner 拆解并派发给多个 agent 执行后的子任务结果。
请合并成一份面向用户的最终答复：
1. 先给结论：本次做了什么、是否成功。
2. 再按重要程度列出关键改动、验证结果、涉及文件路径。
3. 保留各 agent 的失败信息与未完成/风险项，不要隐藏失败。
4. 不要编造子任务结果中没有的信息。
5. 用与用户请求相同的语言输出（用户用中文就用中文），条理清晰。"""


def _merge_results(current: dict[str, str], update: dict[str, str]) -> dict[str, str]:
    """合并并行子 agent 返回的结果字典（LangGraph reducer）。"""
    return {**current, **(update or {})}


class TeamState(TypedDict, total=False):
    """父图（planner/supervisor）与子 agent 共享的编排状态。

    并行字段：
    - results：各分支只返回自己的 {task_id: content}，由 reducer 合并；
    - dispatched_ids：supervisor 每轮追加本轮派发的 id，防止重复派发。
    剩余任务集合不存 state，由 supervisor 每轮用
    tasks - dispatched_ids 计算，避免 reducer 初始化语义的坑。
    """

    request: str
    tasks: list[TaskSpec]
    # 本轮并发派发的任务 id（supervisor 产出，fan-out 路由用）
    dispatch: list[str]
    # 每个派发任务对应的上游依赖文本（按 task_id 索引）
    dispatch_texts: dict[str, str]
    # 已派发过的任务 id（用 operator.add 累积，等价于并集）
    dispatched_ids: Annotated[list[str], operator.add]
    results: Annotated[dict[str, str], _merge_results]
    conversation_context: str
    dispatched: int
    final_answer: str


# 下游 agent -> 父图中的子节点名
NODE_BY_AGENT: dict[str, str] = {
    "log-analysis-agent": "log_analysis_agent",
    "code-modify-agent": "code_modify_agent",
    "test-agent": "test_agent",
    "git-agent": "git_agent",
}


def order_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """按 depends_on 做稳定拓扑排序；发现环或缺失依赖时退回原始顺序。"""

    by_id = {task.task_id: task for task in tasks}
    ordered: list[TaskSpec] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task: TaskSpec) -> None:
        if task.task_id in visited:
            return
        if task.task_id in visiting:
            raise ValueError(f"dependency cycle at {task.task_id}")
        visiting.add(task.task_id)
        for dependency in task.depends_on:
            dependency_task = by_id.get(dependency)
            if dependency_task is not None:
                visit(dependency_task)
        visiting.discard(task.task_id)
        visited.add(task.task_id)
        ordered.append(task)

    try:
        for task in tasks:
            visit(task)
    except ValueError:
        logger.warning("[TeamGraph] 任务依赖存在环，退回原始顺序")
        return list(tasks)
    return ordered


def _writer() -> Callable[[dict[str, Any]], None]:
    """取 custom 流写入器；非流式调用场景下退化为空操作。"""
    try:
        return get_stream_writer()
    except Exception:
        return lambda _payload: None


def _find_task(state: TeamState, task_id: str) -> TaskSpec | None:
    for task in state.get("tasks", []) or []:
        if task.task_id == task_id:
            return task
    return None


def _upstream_text(task: TaskSpec, results: dict[str, str]) -> str:
    if not task.depends_on:
        return ""
    blocks: list[str] = []
    for dependency in task.depends_on:
        content = results.get(dependency)
        if content:
            blocks.append(f"[{dependency}]\n{content[:MAX_UPSTREAM_CHARS]}")
    return "\n\n".join(blocks)


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


async def aggregate_results(
    model_builder: Callable[[], Any] | None,
    request: str,
    tasks: list[TaskSpec],
    results: dict[str, str],
) -> str:
    """子任务结果 -> 面向用户的最终答复。"""

    if not tasks:
        return "Planner 未产出可执行的子任务。"
    if len(tasks) == 1:
        only = tasks[0]
        return results.get(only.task_id, "").strip() or "（子任务未产出内容）"

    sections: list[str] = []
    for task in tasks:
        content = results.get(task.task_id, "").strip()[:MAX_RESULT_CHARS]
        sections.append(
            f"### {task.title}（{task.agent_id}, task_id={task.task_id}）\n{content}"
        )
    body = "\n\n".join(sections)
    if model_builder is None:
        return body
    try:
        response = await model_builder().ainvoke(
            [
                {"role": "system", "content": AGGREGATE_PROMPT},
                {
                    "role": "user",
                    "content": f"用户原始请求：\n{request}\n\n子任务结果：\n{body}",
                },
            ]
        )
        text = _stringify(response).strip()
        if text:
            return text
    except Exception:
        logger.exception("[TeamGraph] 汇总失败，返回子任务结果拼接")
    return body


def build_team_graph(
    *,
    executor: AgentExecutor,
    bus: A2ABus,
    model_builder: Callable[[], Any] | None,
):
    """父图：supervisor 负责动态 fan-out 派发，子节点是各下游 agent，最后聚合。

    拓扑：supervisor ->(条件边，可返回多个节点) [child1, child2, ...] 并行
          -> supervisor -> ... -> aggregate -> END

    每轮 supervisor 把「所有依赖已满足且未派发过」的子任务一起并发派发：
    - 无依赖任务在第一轮就并行执行；
    - 有依赖的任务在依赖所在轮完成后，下一轮满足条件时再派发。
    子 agent 内部各自运行自己的 ReAct 子图。
    """

    def supervisor(state: TeamState) -> dict[str, Any]:
        """父 agent：把当前所有依赖已满足的子任务一次并发派发（动态 fan-out）。"""
        tasks = {task.task_id: task for task in state.get("tasks", []) or []}
        all_ids = set(tasks)
        results = dict(state.get("results", {}) or {})
        dispatched_ids = set(state.get("dispatched_ids", []) or [])
        # 尚未派发过的任务
        remaining = [task_id for task_id in all_ids if task_id not in dispatched_ids]

        runnable: list[str] = []
        valid_remaining: list[str] = []
        for task_id in remaining:
            task = tasks[task_id]
            deps = list(task.depends_on or [])
            # 依赖要么已完成，要么是尚未派发的任务（本轮之后仍可能完成）
            satisfiable = all(dep in results or dep in all_ids for dep in deps)
            if satisfiable:
                valid_remaining.append(task_id)
            if (
                satisfiable
                and get_agent(task.agent_id) is not None
                and all(dep in results for dep in deps)
            ):
                runnable.append(task_id)

        if not runnable:
            # 没有任何可执行任务：如果还有可能完成的剩余任务就说明依赖还没到齐，
            # 等下一轮；否则剩余任务全部不可执行（依赖缺失/agent 未知），标记失败收尾。
            if valid_remaining:
                logger.info("[TeamGraph] 剩余任务依赖未就绪，等待下一轮：%s", valid_remaining)
                return {
                    "dispatch": [],
                    "dispatch_texts": {},
                    "dispatched_ids": [],
                }
            failures: dict[str, str] = {}
            for task_id in remaining:
                task = tasks.get(task_id)
                if task is None:
                    failures[task_id] = f"子任务 {task_id} 失败：计划中缺少该任务定义。"
                elif get_agent(task.agent_id) is None:
                    failures[task_id] = f"子任务 {task_id} 失败：未知 agent {task.agent_id}。"
                else:
                    failures[task_id] = (
                        f"子任务 {task_id} 失败：依赖 {task.depends_on} 未完成。"
                    )
            if failures:
                logger.warning("[TeamGraph] 无法继续派发，标记失败：%s", list(failures))
            return {
                "results": {**results, **failures},
                "dispatch": [],
                "dispatch_texts": {},
                "dispatched_ids": [],
            }

        dispatched = int(state.get("dispatched", 0) or 0)
        return {
            "dispatch": runnable,
            "dispatch_texts": {
                task_id: _upstream_text(tasks[task_id], results)
                for task_id in runnable
            },
            "dispatched_ids": runnable,
            "dispatched": dispatched + 1,
        }

    def route_after_supervisor(state: TeamState) -> str | list[str]:
        """fan-out：返回本轮所有可派发任务对应的子节点名（可并行执行）。"""
        dispatch = state.get("dispatch") or []
        if not dispatch:
            return "aggregate"
        nodes: list[str] = []
        for task_id in dispatch:
            task = _find_task(state, task_id)
            if task is None:
                continue
            node = NODE_BY_AGENT.get(task.agent_id)
            if node is not None and node not in nodes:
                nodes.append(node)
        return nodes or "aggregate"

    def make_agent_node(agent_id: str):
        """子 agent 节点：驱动该 agent 自己的 ReAct 子图，并把事件透传给前端。"""

        async def run_agent(state: TeamState) -> dict[str, Any]:
            writer = _writer()
            dispatch = list(state.get("dispatch", []) or [])
            # fan-out 时每个子节点看到的是同一份 dispatch 列表，
            # 只能取属于自己 agent 的任务，否则所有节点都会抢跑同一个任务。
            task_id = ""
            for candidate_id in dispatch:
                candidate = _find_task(state, candidate_id)
                if candidate is not None and candidate.agent_id == agent_id:
                    task_id = candidate_id
                    break
            task = _find_task(state, task_id) if task_id else None
            agent = get_agent(agent_id)
            if task is None or agent is None:
                return {"results": {}}

            content = ""
            try:
                # 每个并发分支独立成段，避免并行 token 混进同一个气泡
                writer({"type": "new_response"})
                async for event in executor.astream(
                    agent=agent,
                    task=task,
                    upstream_text=str((state.get("dispatch_texts") or {}).get(task_id, "")),
                    conversation_context=state.get("conversation_context", ""),
                    bus=bus,
                ):
                    if event.get("type") == "agent_result":
                        content = str(event.get("content", ""))
                    writer(event)
            except Exception as exc:  # noqa: BLE001 - 单个子 agent 失败不能拖垮父图
                logger.exception(
                    "[TeamGraph] 子 agent %s 执行 task=%s 异常", agent_id, task_id
                )
                content = f"子任务 {task_id}（{agent.name}）执行异常：{exc}"
                writer(
                    {
                        "type": "error",
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "error": content,
                    }
                )
                if bus is not None:
                    # 子 agent 未能正常返回，补一条失败结果，保持 A2A 记录完整
                    bus.result(
                        from_agent=agent_id,
                        to_agent=PLANNER_ID,
                        task_id=task_id,
                        content=content,
                        payload={"agent_name": agent.name, "failed": True},
                    )

            return {
                "results": {task.task_id: content or "（该子任务未产出内容）"},
            }

        return run_agent

    async def aggregate(state: TeamState) -> dict[str, Any]:
        """父 agent 收尾：把子任务结果汇总成最终答复。"""
        writer = _writer()
        tasks = list(state.get("tasks", []) or [])
        results = dict(state.get("results", {}) or {})
        writer({"type": "new_response"})
        answer = await aggregate_results(
            model_builder,
            state.get("request", ""),
            tasks,
            results,
        )
        writer({"type": TEAM_FINAL_EVENT, "content": answer})
        return {"final_answer": answer}

    graph = StateGraph(TeamState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("aggregate", aggregate)
    for agent in DOWNSTREAM_AGENTS:
        graph.add_node(NODE_BY_AGENT[agent.agent_id], make_agent_node(agent.agent_id))

    graph.set_entry_point("supervisor")

    routing: dict[str, str] = {"aggregate": "aggregate"}
    for agent in DOWNSTREAM_AGENTS:
        node = NODE_BY_AGENT[agent.agent_id]
        routing[node] = node
        graph.add_edge(node, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, routing)
    graph.add_edge("aggregate", END)
    return graph.compile()
