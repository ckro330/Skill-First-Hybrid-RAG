from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_settings
from memory.repository import MEMORY_STATUSES, MEMORY_TYPES, MemoryRepository

logger = logging.getLogger("memory.manager")


@dataclass
class MemoryRecord:
    id: str
    user_id: str
    type: str
    content: str
    importance: float
    status: str
    created_at: datetime
    updated_at: datetime
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "content": self.content,
            "importance": self.importance,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "score": self.score,
        }


class MemoryManager:
    """长期记忆统一访问入口（与 SessionManager 同级）。

    AgentManager → MemoryManager → MemoryRepository → PostgreSQL/pgvector。
    DATABASE_URL 未配置或 PostgreSQL 不可用时自动降级：
    retrieve_memories 返回空列表，写操作返回 None，不影响 Agent 主链路。
    """

    def __init__(self, base_dir: Path) -> None:
        settings = get_settings()
        self.base_dir = base_dir
        self.default_user_id = settings.memory_default_user_id
        self._embed_model: Any = self._build_embed_model()
        self._repository: MemoryRepository | None = None
        if settings.database_url:
            self._repository = MemoryRepository(
                database_url=settings.database_url,
                vector_dim=settings.memory_embedding_dimension,
                schema_path=Path(__file__).resolve().parent / "schema.sql",
            )

    def _build_embed_model(self) -> Any:
        settings = get_settings()
        try:
            if settings.embedding_provider == "local":
                from knowledge_retrieval.embeddings import LocalEmbedding

                return LocalEmbedding()
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(
                api_key=settings.embedding_api_key,
                api_base=settings.embedding_base_url,
                model=settings.embedding_model,
            )
        except Exception:
            logger.exception("embedding 模型初始化失败，长期记忆将不可用")
            return None

    def initialize(self) -> None:
        if self._repository is None:
            logger.warning(
                "DATABASE_URL 未配置，长期记忆处于降级模式（检索返回空）"
            )
            return
        self._repository.initialize()
        if not self.is_available:
            logger.warning("PostgreSQL 不可用，长期记忆降级为空")

    @property
    def is_available(self) -> bool:
        return self._repository is not None and self._repository.is_available

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embed_model is None:
            raise RuntimeError("embedding 模型不可用")
        batch = getattr(self._embed_model, "get_text_embedding_batch", None)
        if batch is not None:
            return batch(texts)
        return [self._embed_model.get_text_embedding(text) for text in texts]

    def add_memory_sync(
        self,
        user_id: str,
        content: str,
        type: str = "episodic",
        importance: float = 0.5,
        status: str = "active",
    ) -> MemoryRecord | None:
        """同步写入一条长期记忆（供工具等非 async 上下文调用）。"""
        content = (content or "").strip()
        if not content:
            raise ValueError("content 不能为空")
        if type not in MEMORY_TYPES:
            raise ValueError(f"type 必须是 {', '.join(MEMORY_TYPES)} 之一")
        if status not in MEMORY_STATUSES:
            raise ValueError(f"status 必须是 {', '.join(MEMORY_STATUSES)} 之一")
        if not self.is_available:
            logger.warning("长期记忆未就绪，忽略 add_memory 调用")
            return None
        try:
            embeddings = self._embed([content])
            row = self._repository.insert(
                user_id,
                type,
                content,
                float(importance),
                status,
                embeddings[0],
            )
        except Exception:
            logger.exception("add_memory 失败")
            return None
        return MemoryRecord(**row)

    async def add_memory(
        self,
        user_id: str,
        content: str,
        type: str = "episodic",
        importance: float = 0.5,
        status: str = "active",
    ) -> MemoryRecord | None:
        return await asyncio.to_thread(
            self.add_memory_sync,
            user_id,
            content,
            type,
            importance,
            status,
        )

    async def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        types: list[str] | None = None,
        min_score: float | None = None,
    ) -> list[MemoryRecord]:
        query = (query or "").strip()
        if not self.is_available or not query:
            return []
        threshold = (
            get_settings().memory_retrieval_min_score if min_score is None else min_score
        )
        try:
            embeddings = await asyncio.to_thread(self._embed, [query])
            rows = await asyncio.to_thread(
                self._repository.search,
                user_id,
                embeddings[0],
                max(1, top_k),
                types,
            )
        except Exception:
            logger.exception("retrieve_memories 失败")
            return []
        records = [MemoryRecord(**row) for row in rows]

        # 回传门限：search 已按相似度降序取 top_k，因此对结果做后置过滤
        # 与"先过滤再取 top_k"等价（top_k 之外的结果分数只会更低）。
        if threshold is None or threshold <= 0:
            return records
        kept = [
            record
            for record in records
            if record.score is None or float(record.score) >= threshold
        ]
        if len(kept) != len(records):
            logger.info(
                "[Memory] 相关度门限 %.2f：召回 %d 条，回传 %d 条",
                threshold,
                len(records),
                len(kept),
            )
        return kept

    async def get_memory(self, memory_id: str) -> MemoryRecord | None:
        if not self.is_available:
            return None
        try:
            row = await asyncio.to_thread(self._repository.get, memory_id)
        except Exception:
            logger.exception("get_memory 失败")
            return None
        return MemoryRecord(**row) if row else None

    async def list_memories(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        if not self.is_available:
            return []
        try:
            rows = await asyncio.to_thread(
                self._repository.list_memories, user_id, status, limit
            )
        except Exception:
            logger.exception("list_memories 失败")
            return []
        return [MemoryRecord(**row) for row in rows]

    async def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        type: str | None = None,
        importance: float | None = None,
        status: str | None = None,
    ) -> MemoryRecord | None:
        if not self.is_available:
            return None
        try:
            embedding = None
            if content is not None:
                content = content.strip()
                if not content:
                    raise ValueError("content 不能为空")
                embedding = (await asyncio.to_thread(self._embed, [content]))[0]
            row = await asyncio.to_thread(
                self._repository.update,
                memory_id,
                content=content,
                type_=type,
                importance=importance,
                status=status,
                embedding=embedding,
            )
        except Exception:
            logger.exception("update_memory 失败")
            return None
        return MemoryRecord(**row) if row else None

    async def delete_memory(self, memory_id: str) -> bool:
        if not self.is_available:
            return False
        try:
            return bool(await asyncio.to_thread(self._repository.delete, memory_id))
        except Exception:
            logger.exception("delete_memory 失败")
            return False
