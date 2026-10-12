from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.chat import router as chat_router
from api.compress import router as compress_router
from api.files import router as files_router
from api.knowledge_index import router as knowledge_index_router
from api.memories import router as memories_router
from api.sessions import router as sessions_router
from api.tokens import router as tokens_router
from config import get_settings
from graph.agent import agent_manager
from knowledge_retrieval import knowledge_indexer
from tools.skills_scanner import refresh_snapshot


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    refresh_snapshot(settings.backend_dir)
    agent_manager.initialize(settings.backend_dir)
    knowledge_indexer.configure(settings.backend_dir)
    knowledge_indexer.rebuild_index()
    yield


app = FastAPI(
    title="Mini-OpenClaw API",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,  
    allow_origins=["*"],  #允许任何域名端口访问api
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#注册功能入口
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(sessions_router, prefix="/api", tags=["sessions"])
app.include_router(files_router, prefix="/api", tags=["files"])
app.include_router(tokens_router, prefix="/api", tags=["tokens"])
app.include_router(compress_router, prefix="/api", tags=["compress"])
app.include_router(knowledge_index_router, prefix="/api", tags=["knowledge"])
app.include_router(memories_router, prefix="/api", tags=["memories"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
