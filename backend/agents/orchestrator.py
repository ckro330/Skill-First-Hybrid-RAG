from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from agents.a2a import A2ABus
from agents.executor import AgentExecutor
from agents.planner import PlanDecision
from agents.team_graph import (
    TEAM_FINAL_EVENT,
    TeamState,
    build_team_graph,
    order_tasks,
)


logger = logging.getLogger("coding_orchestrator")

MAX_TRANSCRIPT_CONTENT = 2000

__all__ = ["CodingOrchestrator", "order_tasks"]


class CodingOrchestrator:
    """用 LangGraph 编排 planner（父 agent）与四个下游 agent（子 agent）。

    每次请求：
      1. 按 depends_on 对 planner 产出的子任务做拓扑排序；
      2. 新建一条 A2A 总线（不跨会话复用，避免消息互相污染）；
      3. 运行父子团队图：supervisor 派发 -> 子 agent 执行 -> aggregate 汇总；
      4. 把图内 custom 事件原样透传给上层（graph/agent.py -> SSE）。
    """

    def __init__(
        self,
        base_dir: Path,
        model_builder: Callable[[], Any],
        executor: AgentExecutor,
        bus_factory: Callable[[], A2ABus] = A2ABus,
    ) -> None:
        self.base_dir = base_dir
        self._model_builder = model_builder
        self._executor = executor
        # 保留工厂注入点：后续换成真正的 HTTP/JSON-RPC A2A 传输层时无需改上层
        self._bus_factory = bus_factory

    async def astream(
        self,
        message: str,
        decision: PlanDecision,
        conversation_context: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        ordered = order_tasks(decision.tasks)
        bus = self._bus_factory()

        yield {
            "type": "plan",
            "domain": decision.domain,
            "reason": decision.reason,
            "multi_agent": decision.multi_agent,
            "tasks": [task.to_dict() for task in ordered],
        }

        graph = build_team_graph(
            executor=self._executor,
            bus=bus,
            model_builder=self._model_builder,
        )
        initial: TeamState = {
            "request": message,
            "tasks": ordered,
            "dispatch": [],
            "dispatch_texts": {},
            "dispatched_ids": [],
            "results": {},
            "conversation_context": conversation_context,
            "dispatched": 0,
            "final_answer": "",
        }

        final_answer = ""
        async for item in graph.astream(initial, stream_mode=["custom"]):
            # 单模式下 langgraph 可能返回裸 chunk，多模式返回 (mode, chunk)
            if isinstance(item, tuple):
                mode, payload = item
                if mode != "custom":
                    continue
            else:
                payload = item
            event = payload
            if not isinstance(event, dict):
                continue
            if event.get("type") == TEAM_FINAL_EVENT:
                final_answer = str(event.get("content", ""))
                continue
            yield event

        yield {"type": "a2a_transcript", "messages": self._transcript(bus)}
        yield {"type": "done", "content": final_answer or "（本次未产出内容）"}

    @staticmethod
    def _transcript(bus: A2ABus) -> list[dict[str, Any]]:
        """输出本次请求的 A2A 消息记录，便于前端展示 agent 间通信过程。"""

        messages: list[dict[str, Any]] = []
        for message in bus.history():
            data = message.to_dict()
            content = str(data.get("content", ""))
            if len(content) > MAX_TRANSCRIPT_CONTENT:
                data["content"] = content[:MAX_TRANSCRIPT_CONTENT] + "\n...[truncated]"
            messages.append(data)
        return messages
