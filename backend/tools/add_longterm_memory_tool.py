from __future__ import annotations

import asyncio
from typing import Type

from langchain_core.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class AddLongtermMemoryInput(BaseModel):
    content: str = Field(
        ...,
        description="要写入长期记忆的精炼内容，需自包含、可复用",
    )
    type: str = Field(
        "preference",
        description="记忆类型：profile(用户身份/画像) / preference(偏好) / project(项目信息) / episodic(经验教训)",
    )
    importance: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="重要度 0~1。明确要求记住/身份偏好 0.9~1.0；可复用经验方法论 0.75~0.9；任务级细节 0.5~0.75；一次性事实 0.2~0.5；闲聊 0~0.2",
    )
    user_id: str | None = Field(None, description="用户 ID，默认使用 default")


class AddLongtermMemoryTool(BaseTool):
    name: str = "add_longterm_memory"
    description: str = (
        "当用户明确要求记住某信息、表达持久偏好/身份/习惯，"
        "或对话中出现值得跨任务复用的经验、方法论时调用，"
        "把精炼后的内容写入长期记忆库。"
        "importance 标准：明确要求记住/身份偏好声明 0.9~1.0；"
        "可复用经验方法论 0.75~0.9；任务级细节 0.5~0.75；"
        "一次性事实 0.2~0.5；闲聊 0~0.2。只在 importance>0.7 时写入。"
    )
    args_schema: Type[BaseModel] = AddLongtermMemoryInput

    def _run(
        self,
        content: str,
        type: str = "preference",
        importance: float = 0.8,
        user_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        # 函数级导入，避免 graph.agent <-> tools 循环导入
        from graph.agent import agent_manager

        manager = agent_manager.memory_manager
        if manager is None or not manager.is_available:
            return "长期记忆未就绪（PostgreSQL 未连接或未初始化），未写入。"
        try:
            record = manager.add_memory_sync(
                user_id=user_id or manager.default_user_id,
                content=content,
                type=type,
                importance=importance,
            )
        except Exception as exc:  # noqa: BLE001 - 工具异常必须返回给模型可读信息
            return f"长期记忆写入失败: {exc}"
        if record is None:
            return "长期记忆写入失败（add_memory 返回空）。"
        return (
            f"已写入长期记忆：id={record.id}, type={record.type}, "
            f"importance={record.importance}。"
        )

    async def _arun(
        self,
        content: str,
        type: str = "preference",
        importance: float = 0.8,
        user_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run,
            content,
            type,
            importance,
            user_id,
            None,
        )
