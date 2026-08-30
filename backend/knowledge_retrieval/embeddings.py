from __future__ import annotations

import threading
from typing import Any

from llama_index.core.embeddings import BaseEmbedding

from config import get_settings

_embedding_model: Any = None
_model_lock = threading.Lock()


def get_local_embedding_model() -> Any:
    """加载并缓存 sentence-transformers 模型（进程内单例）。"""
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer

                _embedding_model = SentenceTransformer(get_settings().embedding_model)
    return _embedding_model


class LocalEmbedding(BaseEmbedding):
    """把本地 sentence-transformers 模型包装成 llama-index Embedding。"""

    def __init__(
        self,
        model_name: str | None = None,
        embed_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        settings = get_settings()
        super().__init__(
            model_name=model_name or settings.embedding_model,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = get_local_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in embeddings]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._encode([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def get_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        return self._encode(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)

    async def aget_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[list[float]]:
        return self._get_text_embeddings(texts)


class LocalLangchainEmbeddings:
    """供 ragas 等使用的 langchain 风格本地 embedding 适配器。"""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = get_local_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in embeddings]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
