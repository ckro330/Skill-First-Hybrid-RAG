-- Long-term Memory 表结构（PostgreSQL + pgvector）
-- 向量维度 {{dim}} 由 MEMORY_EMBEDDING_DIMENSION 注入

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     TEXT NOT NULL,
    type        TEXT NOT NULL CHECK (type IN ('profile', 'preference', 'project', 'episodic')),
    content     TEXT NOT NULL,
    importance  REAL NOT NULL DEFAULT 0.5,
    status      TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector({{dim}})
);

CREATE INDEX IF NOT EXISTS memories_user_status_idx
    ON memories (user_id, status);

CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding vector_cosine_ops);
