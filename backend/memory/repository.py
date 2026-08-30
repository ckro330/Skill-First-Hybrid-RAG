from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger("memory.repository")

MEMORY_TYPES = ("profile", "preference", "project", "episodic")
MEMORY_STATUSES = ("active", "inactive")


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in embedding) + "]"


class MemoryRepository:
    """PostgreSQL + pgvector 数据访问层。

    embedding 以 `'[0.1,0.2,...]'::vector` 文本形式传给 pgvector，避免额外引入
    pgvector 的 Python adapter 依赖。所有方法均为同步实现，由 MemoryManager
    通过 asyncio.to_thread 包装后提供给 async 上层。
    """

    def __init__(self, database_url: str, vector_dim: int, schema_path: Path) -> None:
        self._database_url = database_url
        self._vector_dim = vector_dim
        self._schema_path = schema_path
        self._pool: ConnectionPool | None = None

    @property
    def is_available(self) -> bool:
        return self._pool is not None

    def initialize(self) -> None:
        if self._pool is not None:
            return
        try:
            pool = ConnectionPool(
                conninfo=self._database_url,
                min_size=1,
                max_size=4,
                open=False,
                kwargs={"row_factory": dict_row},
            )
            pool.open(wait=True, timeout=5)
            self._pool = pool
            self.ensure_schema()
        except Exception:
            logger.exception("PostgreSQL 连接失败，长期记忆进入降级模式")
            self.close()

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                logger.exception("MemoryRepository 关闭连接池失败")
            self._pool = None

    def ensure_schema(self) -> None:
        if self._pool is None:
            return
        ddl = self._schema_path.read_text(encoding="utf-8").replace("{{dim}}", str(self._vector_dim))
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def _check_available(self) -> None:
        if self._pool is None:
            raise RuntimeError("MemoryRepository is not available (PostgreSQL not connected)")

    def insert(
        self,
        user_id: str,
        type_: str,
        content: str,
        importance: float,
        status: str,
        embedding: list[float],
    ) -> dict[str, Any]:
        self._check_available()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories (user_id, type, content, importance, status, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    RETURNING id, user_id, type, content, importance, status, created_at, updated_at
                    """,
                    (user_id, type_, content, importance, status, _vector_literal(embedding)),
                )
                row = cur.fetchone()
            conn.commit()
        return row

    def get(self, memory_id: str) -> dict[str, Any] | None:
        self._check_available()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, type, content, importance, status, created_at, updated_at
                    FROM memories
                    WHERE id = %s
                    """,
                    (memory_id,),
                )
                return cur.fetchone()

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
        types: list[str] | None = None,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        self._check_available()
        vector_text = _vector_literal(query_embedding)
        sql = """
            SELECT id, user_id, type, content, importance, status, created_at, updated_at,
                   1 - (embedding <=> %s::vector) AS score
            FROM memories
            WHERE user_id = %s AND status = %s
        """
        params: list[Any] = [vector_text, user_id, status]
        if types:
            sql += " AND type = ANY(%s)"
            params.append(list(types))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([vector_text, top_k])

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def list_memories(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._check_available()
        sql = """
            SELECT id, user_id, type, content, importance, status, created_at, updated_at
            FROM memories
            WHERE user_id = %s
        """
        params: list[Any] = [user_id]
        if status is not None:
            sql += " AND status = %s"
            params.append(status)
        sql += " ORDER BY updated_at DESC LIMIT %s"
        params.append(limit)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        type_: str | None = None,
        importance: float | None = None,
        status: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any] | None:
        self._check_available()
        sets: list[str] = []
        params: list[Any] = []
        if content is not None:
            sets.append("content = %s")
            params.append(content)
        if type_ is not None:
            sets.append("type = %s")
            params.append(type_)
        if importance is not None:
            sets.append("importance = %s")
            params.append(importance)
        if status is not None:
            sets.append("status = %s")
            params.append(status)
        if embedding is not None:
            sets.append("embedding = %s::vector")
            params.append(_vector_literal(embedding))
        if not sets:
            return self.get(memory_id)

        sets.append("updated_at = now()")
        params.append(memory_id)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE memories
                    SET {", ".join(sets)}
                    WHERE id = %s
                    RETURNING id, user_id, type, content, importance, status, created_at, updated_at
                    """,
                    params,
                )
                row = cur.fetchone()
            conn.commit()
        return row

    def set_status(self, memory_id: str, status: str) -> dict[str, Any] | None:
        return self.update(memory_id, status=status)

    def delete(self, memory_id: str) -> bool:
        self._check_available()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
                deleted = cur.rowcount
            conn.commit()
        return bool(deleted)
