from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from graph.agent import agent_manager
from memory.memory_manager import MemoryManager

router = APIRouter()

MemoryType = Literal["profile", "preference", "project", "episodic"]
MemoryStatus = Literal["active", "inactive"]


class CreateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1)
    type: MemoryType = "episodic"
    importance: float = Field(0.5, ge=0.0, le=1.0)
    user_id: str | None = None


class UpdateMemoryRequest(BaseModel):
    content: str | None = Field(None, min_length=1)
    type: MemoryType | None = None
    importance: float | None = Field(None, ge=0.0, le=1.0)
    status: MemoryStatus | None = None


class SearchMemoryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    user_id: str | None = None


def _require_memory_manager() -> MemoryManager:
    if agent_manager.memory_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    return agent_manager.memory_manager


def _ensure_available(manager: MemoryManager) -> None:
    if not manager.is_available:
        raise HTTPException(
            status_code=503,
            detail="长期记忆未就绪（DATABASE_URL 未配置或 PostgreSQL 不可用）",
        )


def _user_id(manager: MemoryManager, user_id: str | None) -> str:
    return user_id or manager.default_user_id


@router.post("/memories")
async def create_memory(payload: CreateMemoryRequest) -> dict:
    manager = _require_memory_manager()
    _ensure_available(manager)
    record = await manager.add_memory(
        user_id=_user_id(manager, payload.user_id),
        content=payload.content,
        type=payload.type,
        importance=payload.importance,
    )
    if record is None:
        raise HTTPException(status_code=500, detail="长期记忆写入失败")
    return record.to_dict()


@router.get("/memories")
async def list_memories(
    user_id: str | None = Query(default=None),
    status: MemoryStatus | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    manager = _require_memory_manager()
    _ensure_available(manager)
    records = await manager.list_memories(
        user_id=_user_id(manager, user_id),
        status=status,
        limit=limit,
    )
    return [record.to_dict() for record in records]


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str) -> dict:
    manager = _require_memory_manager()
    _ensure_available(manager)
    record = await manager.get_memory(memory_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return record.to_dict()


@router.put("/memories/{memory_id}")
async def update_memory(memory_id: str, payload: UpdateMemoryRequest) -> dict:
    manager = _require_memory_manager()
    _ensure_available(manager)
    record = await manager.update_memory(
        memory_id,
        content=payload.content,
        type=payload.type,
        importance=payload.importance,
        status=payload.status,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return record.to_dict()


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str) -> dict[str, bool]:
    manager = _require_memory_manager()
    _ensure_available(manager)
    deleted = await manager.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"ok": True}


@router.post("/memories/search")
async def search_memories(payload: SearchMemoryRequest) -> list[dict]:
    manager = _require_memory_manager()
    _ensure_available(manager)
    records = await manager.retrieve_memories(
        user_id=_user_id(manager, payload.user_id),
        query=payload.query,
        top_k=payload.top_k,
    )
    return [record.to_dict() for record in records]
