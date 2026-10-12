from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


KIND_TASK = "task"
KIND_RESULT = "result"

PLANNER_ID = "planner"


def new_task_id(prefix: str = "t") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class A2AMessage:
    """A2A 消息：planner 与下游 agent 之间的任务派发 / 结果回传。"""

    from_agent: str
    to_agent: str
    kind: str
    content: str
    task_id: str = ""
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "task_id": self.task_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "kind": self.kind,
            "content": self.content,
            "payload": self.payload,
            "created_at": self.created_at,
        }


class A2ABus:
    """进程内 A2A 消息总线：记录 planner 与下游 agent 之间的任务派发 / 结果回传。

    调用方只依赖 task()/result()/history()，后续可替换为真正的
    HTTP/JSON-RPC A2A 传输层而不影响上层逻辑。总线按请求创建，不跨会话复用。
    """

    def __init__(self) -> None:
        self._messages: list[A2AMessage] = []

    def send(self, message: A2AMessage) -> A2AMessage:
        self._messages.append(message)
        return message

    def task(
        self,
        *,
        from_agent: str,
        to_agent: str,
        task_id: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> A2AMessage:
        return self.send(
            A2AMessage(
                from_agent=from_agent,
                to_agent=to_agent,
                kind=KIND_TASK,
                content=content,
                task_id=task_id,
                payload=payload or {},
            )
        )

    def result(
        self,
        *,
        from_agent: str,
        to_agent: str,
        task_id: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> A2AMessage:
        return self.send(
            A2AMessage(
                from_agent=from_agent,
                to_agent=to_agent,
                kind=KIND_RESULT,
                content=content,
                task_id=task_id,
                payload=payload or {},
            )
        )

    def history(self) -> list[A2AMessage]:
        return list(self._messages)
