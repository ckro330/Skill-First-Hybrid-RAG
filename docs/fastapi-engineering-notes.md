# FastAPI 工程实践笔记 —— 以 Skill-First-Hybrid-RAG 为例

> 面向"已经会写 `@app.get`，但不知道真实项目怎么组织"的读者。
> 全文所有结论都来自本仓库的真实代码，不是通用教程的复述。

## 0. 先看全貌

这个项目的后端 API 层只有 9 个文件，全部在 `backend/`：

```text
backend/
├─ app.py                      # 应用入口：装配 + 生命周期
├─ config.py                   # 配置：.env → Settings 单例
└─ api/
   ├─ chat.py                  # 聊天（含 SSE 流式）
   ├─ sessions.py              # 会话 CRUD
   ├─ compress.py              # 会话压缩
   ├─ files.py                 # 可编辑文件读写
   ├─ tokens.py                # token 统计
   ├─ knowledge_index.py       # 知识索引状态 / 重建
   └─ memories.py              # 长期记忆 CRUD
```

**一句话概括这个项目的 FastAPI 用法**：`app.py` 只负责"装配"，`api/*.py` 只负责"翻译 HTTP ↔ 领域对象"，真正的逻辑全在下游模块（`graph/`、`memory/`、`knowledge_retrieval/`、`tools/`）。

这就是工程应用和写 demo 最大的区别：**路由层要尽量薄**。

---

## 1. 应用入口与生命周期

看 [backend/app.py](../backend/app.py)：

```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    refresh_snapshot(settings.backend_dir)          # 1) 生成技能快照
    agent_manager.initialize(settings.backend_dir)  # 2) 初始化 Agent
    knowledge_indexer.configure(settings.backend_dir)
    knowledge_indexer.rebuild_index()               # 3) 重建知识索引
    yield

app = FastAPI(title="Mini-OpenClaw API", version="0.1.0", lifespan=lifespan)
```

### 1.1 为什么用 `lifespan` 而不是模块级初始化

三种初始化位置，差别很大：

| 做法 | 问题 |
| --- | --- |
| 模块顶层直接初始化 | **import 就会执行**，无法控制时机；测试导入即触发副作用；失败时报错位置极具误导性 |
| `@app.on_event("startup")` | 已被 FastAPI 官方标记弃用，且启动/关闭逻辑分散在两个回调里 |
| `lifespan`（本项目） | 启动与关闭成对出现在同一个函数里，能拿到 `yield` 前的资源、`yield` 后释放 |

### 1.2 这一条是本仓库真实踩过的坑

`backend/api/tokens.py` 第 14 行：

```python
ENCODER = tiktoken.get_encoding("cl100k_base")
```

这行写在**模块顶层**，而 `tiktoken` 第一次取词表时会联网下载 BPE 文件。于是每次 `import app` 都隐含一次网络请求——缓存缺失时进程会卡死在建立连接阶段，uvicorn 因为"先跑完 lifespan 才 bind 端口"，表现出的现象是**端口一直不监听、前端连不上**。

结论：**import 应该是廉价的、离线的、无副作用的**。凡是"加载模型、连数据库、下载文件、读大文件"的动作，一律放进 `lifespan` 或改成懒加载。

### 1.3 启动顺序是有依赖的

上面 1→2→3 的顺序不能乱：`refresh_snapshot` 先生成技能快照，`agent_manager.initialize` 之后才读得到；索引重建依赖 `configure` 先注入 `base_dir`。

**工程建议**：lifespan 里超过三个步骤时，把顺序依赖写成注释或拆成 `_bootstrap(...)` 函数，别让下一个人靠猜。

---

## 2. 路由分层：`APIRouter` 是模块的接口契约

每个 `api/*.py` 的模式完全一致：

```python
# backend/api/sessions.py
router = APIRouter()          # 只声明，不注册

@router.get("/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    ...
```

然后在 `app.py` 里统一装配（第 47-53 行）：

```python
app.include_router(chat_router,            prefix="/api", tags=["chat"])
app.include_router(sessions_router,        prefix="/api", tags=["sessions"])
app.include_router(files_router,           prefix="/api", tags=["files"])
app.include_router(tokens_router,          prefix="/api", tags=["tokens"])
app.include_router(compress_router,        prefix="/api", tags=["compress"])
app.include_router(knowledge_index_router, prefix="/api", tags=["knowledge"])
app.include_router(memories_router,        prefix="/api", tags=["memories"])
```

### 2.1 这样做换来三件事

1. **`app.py` 变成一张装配清单**——想知道系统有哪些能力，看这 7 行就够。
2. **`prefix` 只写一次**：各模块内部写 `/sessions`，不用重复写 `/api/sessions`。
3. **`tags` 直接决定 `/docs` 的分组**：打开 Swagger UI 会按 chat / sessions / memories 分块，对接前端时非常好用。

### 2.2 一个容易被忽略的细节

`backend/api/__init__.py` 里**只有一句文档字符串，不导入任何路由**。这是刻意的：

- 一旦 `api/__init__.py` 去 import 各个 router，`app.py` 导入任何一个模块都会连带触发全部模块的 import 副作用（比如上面说的 tiktoken）。
- 路由模块之间更容易绕成循环导入。

**约定**：`__init__.py` 保持空、或只放文档字符串/类型/常量，导入关系永远由 `app.py` 单向发起。

### 2.3 什么时候该拆文件

经验线：**一个路由文件对应一个"资源/领域"**，而不是对应技术层次。

- ✅ `sessions.py`（会话资源）、`memories.py`（记忆资源）
- ❌ `gets.py` / `posts.py`（按 HTTP 方法拆）

本项目的 `compress.py` 是个合理例外：它虽然操作的是 session，但语义上是一条独立的后台计算链路（调 LLM 做摘要），单独成文件反而更清楚。

---

## 3. 请求与响应契约：Pydantic

### 3.1 请求体：用 `BaseModel` 做**声明式校验**

```python
# backend/api/memories.py
class CreateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1)
    type: MemoryType = "episodic"                  # 见下方 Literal
    importance: float = Field(0.5, ge=0.0, le=1.0)
    user_id: str | None = None
```

你几乎不用写 `if not payload.content: raise ...`——Pydantic 会在进入函数体之前就拦下非法请求，并自动生成 422 响应和 OpenAPI 里的字段说明。

**枚举用 `Literal` 而不是 `str`**（`backend/api/memories.py` 第 12-13 行）：

```python
MemoryType = Literal["profile", "preference", "project", "episodic"]
MemoryStatus = Literal["active", "inactive"]
```

好处是 Swagger UI 里直接给你下拉框，同时数据库里也不会写进意料之外的值。

### 3.2 查询参数：用 `Query(...)` 校验

```python
# backend/api/files.py
@router.get("/files")
async def read_file(path: str = Query(..., min_length=1)) -> dict[str, str]:
    ...

# backend/api/memories.py
async def list_memories(
    user_id: str | None = Query(default=None),
    status: MemoryStatus | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
```

`Query(...)` 里的 `...` 表示必填，`ge/le/min_length` 会被校验并体现在文档里。注意 `limit` 的 `le=500`——**给列表接口加上限**是最容易被漏掉的防护，没有它就是一个被随便打爆的接口。

### 3.3 响应：本项目用返回类型注解，而**没有**用 `response_model`

```python
async def list_sessions() -> list[dict[str, Any]]:
    return session_manager.list_sessions()
```

`-> list[dict[str, Any]]` 会被 FastAPI 读进 OpenAPI 文档，但**它不做运行时过滤**。这与 `response_model=` 有本质区别：

| | 返回类型注解 | `response_model=SomeModel` |
| --- | --- | --- |
| 出现在 OpenAPI | ✅ | ✅ |
| 运行时过滤多余字段 | ❌ | ✅ |
| 强制字段类型 | ❌ | ✅ |

本项目选择前者，因为下游返回的就是普通 dict（会话 JSON、记忆 dict），再套一层模型收益有限。**但你要知道代价**：如果哪天 `MemoryRecord.to_dict()` 里带了不该外泄的字段（比如 embedding 向量），注解不会帮你挡住。

> 判断标准：**返回结构稳定且会外泄给第三方的接口，用 `response_model`；内部调试型接口，返回注解够用。**

---

## 4. 错误处理：`HTTPException` 的语义分层

本项目的状态码用法很规范，可以直接抄：

| 状态码 | 使用场景 | 例子 |
| --- | --- | --- |
| 400 | 请求本身合法，但业务上不允许 | `compress.py`: 消息不足 4 条 |
| 404 | 资源不存在 | `memories.py`: `Memory not found` |
| 500 | 依赖已就绪，但操作失败 | `memories.py`: 写入长期记忆失败 |
| 503 | **服务尚未就绪** | 全项目统一的 `Agent manager is not initialized` |

### 4.1 `503` 这个模式值得单独学

```python
session_manager = agent_manager.session_manager
if session_manager is None:
    raise HTTPException(status_code=503, detail="Agent manager is not initialized")
```

因为资源是在 `lifespan` 里初始化的，存在一个"应用起来了但还没初始化完"的时间窗。用 503（Service Unavailable）而不是 500，语义上是准确的：**不是你的请求错了，是我现在还不能服务**。

### 4.2 但这 6 行重复了 14 次

`rg "Agent manager is not initialized"` 在项目里能搜到 **14 处**。这是典型的**依赖注入该出场的地方**，见下一节。

---

## 5. 依赖注入：本项目用的是"模块级单例"，不是 `Depends`

这是最值得你注意的一点：**全项目没有一处 `Depends`**。

它的实际做法是（`backend/api/sessions.py`）：

```python
from graph.agent import agent_manager   # 模块级单例

@router.get("/sessions")
async def list_sessions():
    session_manager = agent_manager.session_manager   # 手动取
    if session_manager is None:
        raise HTTPException(status_code=503, ...)
```

`agent_manager` 是 `backend/graph/agent.py` 末尾创建的全局对象，生命周期由 lifespan 驱动。

### 5.1 为什么这样能跑，以及代价

能跑，是因为这是一个**单进程、单用户、本地运行**的工作台：没有多租户、没有请求级状态、没有横向扩展。此时全局单例是最省事的做法。

但代价是真实的：

- **难测试**：想测 `list_sessions`，你必须先让全局 `agent_manager` 处于初始化状态。
- **契约不显式**：看函数签名不知道它依赖了什么，得读函数体。
- **每个 handler 都要重复写那 6 行防御代码**。

### 5.2 用 `Depends` 改造的样子

如果要把这个项目往"正经服务"推一步，第一步就是引入依赖：

```python
# backend/api/deps.py
from fastapi import Depends, HTTPException

from graph.agent import agent_manager
from graph.session_manager import SessionManager
from memory.memory_manager import MemoryManager


def get_session_manager() -> SessionManager:
    manager = agent_manager.session_manager
    if manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    return manager


def get_memory_manager() -> MemoryManager:
    manager = agent_manager.memory_manager
    if manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")
    if not manager.is_available:
        raise HTTPException(status_code=503, detail="长期记忆未就绪")
    return manager
```

调用点立刻瘦身：

```python
@router.get("/sessions")
async def list_sessions(
    sessions: SessionManager = Depends(get_session_manager),
) -> list[dict[str, Any]]:
    return sessions.list_sessions()


@router.get("/memories")
async def list_memories(
    memories: MemoryManager = Depends(get_memory_manager),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict]:
    # 为聚焦依赖注入，这里省略了 user_id / status 参数
    records = await memories.list_memories(
        user_id=memories.default_user_id, limit=limit
    )
    return [record.to_dict() for record in records]
```

### 5.3 `Depends` 到底解决了什么

1. **校验集中**：`503` 判断只写一次，新增接口不可能漏掉。
2. **签名即契约**：看参数就知道依赖什么。
3. **测试可替换**：`app.dependency_overrides[get_session_manager] = lambda: FakeSessionManager()`，不需要动全局状态。
4. **可组合**：`get_current_user` 里再 `Depends(get_session_manager)`，形成依赖树。

> 顺便记住一条 FastAPI 细节：**`Depends` 的函数里不要写 `async def` 却做阻塞 I/O**。`def` 形式的依赖会被自动丢进线程池；`async def` 形式的依赖跑在事件循环里，阻塞就会卡住所有人。

### 5.4 什么时候可以不引入 `Depends`

像本项目这样：单机、单用户、无鉴权、无多租户、接口数量几十个以内——单例 + 显式判空是可以接受的工程取舍。

**但至少要意识到自己放弃的是什么**，别在面试里被问"你的依赖是怎么注入的"时才发现自己从没想过这个问题。

---

## 6. 流式响应：SSE 的工程细节（本项目最值得学的部分）

`backend/api/chat.py` 是全书最复杂的一个文件，也是最能体现工程功力的地方。

### 6.1 返回类型不固定

```python
@router.post("/chat")
async def chat(payload: ChatRequest):
    ...
    if payload.stream:
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    final_text = ""
    async for raw_event in event_generator():
        if raw_event.startswith("event: done"):
            final_text = raw_event
    return JSONResponse({"content": final_text})
```

同一个接口，`stream=true` 走流式，`false` 走一次性 JSON。**注意它复用同一个 `event_generator()`**，而不是写两份逻辑——这是很关键的设计：流式与非流式只差"怎么消费"，不差"怎么生产"。

代价：函数没有返回类型注解，OpenAPI 里也描述不清响应结构。对一个本地工具可以接受；对外 API 通常建议拆成 `/chat` 与 `/chat/stream` 两个端点。

### 6.2 SSE 的报文格式

```python
def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```

三个知识点：

1. **`ensure_ascii=False`**：不加的话中文会变成 `\u4e2d\u6587`，体积翻几倍。
2. **结尾必须是 `\n\n`**：SSE 用空行分隔消息块。前端正是在找这个边界（见 `frontend/src/lib/api.ts` 的 `buffer.indexOf("\n\n")`）。
3. **必须配 `media_type="text/event-stream"`**，否则浏览器不会按 event stream 处理。

### 6.3 最容易被忽略：流开出去之后，异常不能再变成状态码

```python
try:
    async for event in agent_manager.astream(...):
        ...
        yield _sse(event_type, data)
except Exception as exc:
    persist_segments(fallback_content=f"请求失败: {str(exc) or 'unknown error'}")
    yield _sse("error", {"error": str(exc)})
```

一旦 `StreamingResponse` 开始吐字节，**HTTP 状态码和响应头就已经发出去了**，这时候再抛异常，客户端只会看到"连接中断"，前端拿不到任何有用信息。

所以流式接口的错误处理必须自己兜：把异常翻译成一个 `error` 事件，塞进流里。这是 SSE 与普通接口最大的思维差异——**错误也是一种事件**。

（真实案例：本项目实测时上游模型返回 402、上游连接被掐断，前端都是靠这个 `error` 事件拿到信息的。）

### 6.4 生成器里的"分段落盘"

`event_generator` 里还有一个细节：它把 token 累积到 `current_segment`，遇到 `new_response` 事件就把当前段落存盘、另起一段；`done` 时统一落盘。

```python
def persist_segments(fallback_content: str | None = None) -> None:
    nonlocal current_segment, conversation_saved
    if conversation_saved:
        return              # ← 幂等：error 分支和 done 分支都可能调用它
    ...
```

`conversation_saved` 这个标志保证了**无论正常结束还是异常中断，都不会重复落盘**。这类"幂等保护"在流式链路里是必需的，因为触发落盘的路径往往不止一条。

### 6.5 客户端断开的现实

`event_generator` 的结构其实只有一个 `try / except`，**没有 `finally`**：

```python
try:
    async for event in agent_manager.astream(...):
        ...
except Exception as exc:
    persist_segments(fallback_content=f"请求失败: ...")
    yield _sse("error", {"error": str(exc)})
```

这意味着：**上游报错**（模型 402、连接被掐断）会走 `except`，落盘照常完成；但**客户端主动断开**时抛的是 `asyncio.CancelledError`，它继承自 `BaseException` 而不是 `Exception`，`except Exception` 捕不到——这一轮的分段就不会落盘。

这是本项目一个真实存在的边界缺口，正确写法是把收尾逻辑放进 `finally`：

```python
try:
    ...
except Exception as exc:
    yield _sse("error", {"error": str(exc)})
finally:
    persist_segments()        # finally 对 CancelledError 同样生效
```

（`persist_segments` 里已经有 `conversation_saved` 幂等保护，放进 `finally` 不会重复落盘。）

生产系统还要额外注意：

- 服务端要能被取消，否则后台会持续烧 LLM 额度；
- 已产生的部分结果要不要保留，是产品决策，本项目选择保留。

### 6.6 SSE vs WebSocket vs 轮询

| 方案 | 适用 |
| --- | --- |
| SSE | **单向、服务端推送**，本项目场景（LLM token 流）最优；原生支持断线重连；实现成本最低 |
| WebSocket | 需要双向实时通信（协同编辑、游戏） |
| 轮询 | 状态更新频率低、实现要最简单（本项目 `knowledge/index/status` 就是轮询） |

---

## 7. 同步 / 异步边界：本项目最讲"良心"的地方

FastAPI 允许 `def` 和 `async def` 两种处理函数：

- `def` handler：FastAPI 自动丢进线程池执行，**阻塞也不影响事件循环**。
- `async def` handler：跑在事件循环里，**里面任何阻塞调用都会卡住整个服务**。

本项目的下游全是同步阻塞库（`psycopg_pool` 连 PostgreSQL、`sentence-transformers` 编码），所以 `memory/memory_manager.py` 的做法是标准答案：

```python
async def retrieve_memories(self, user_id: str, query: str, top_k: int = 5):
    ...
    embeddings = await asyncio.to_thread(self._embed, [query])      # 模型编码 → 线程池
    rows = await asyncio.to_thread(self._repository.search, ...)    # 数据库查询 → 线程池
```

**模式**：*async 接口 + `asyncio.to_thread` 包住同步调用*。好处是调用方永远是 `await`，将来换异步驱动不用改上层。

反面教材（千万别这么写）：

```python
async def retrieve_memories(...):
    embeddings = self._embed([query])          # ❌ 同步编码，卡死事件循环
    rows = self._repository.search(...)        # ❌ 同步查库，卡死事件循环
```

### 7.1 后台任务

`backend/api/knowledge_index.py`：

```python
@router.post("/knowledge/index/rebuild")
async def rebuild_index() -> dict[str, bool]:
    if knowledge_indexer.is_building():
        return {"accepted": True}
    asyncio.create_task(asyncio.to_thread(knowledge_indexer.rebuild_index))
    return {"accepted": True}
```

这是"**立即返回、后台干活**"的经典写法，用轮询查状态（`GET /knowledge/index/status`）。里面有两个关键点：

1. `create_task` 前先查 `is_building()`，**避免并发触发第二次重建**（索引重建是重操作，重复触发会打爆 CPU 和内存）。
2. 返回体是 `{"accepted": true}` 而不是"已完成"——语义上准确表达了"我收下了，但不代表干完了"。

**它的不足**也值得知道：`create_task` 创建的任务没有句柄、没有重试、没有进度百分比，而且**如果进程在任务执行中退出，任务会丢**。生产环境一般用任务队列（Celery / arq）替代。

---

## 8. 配置管理

`backend/config.py` 用了一个非常轻的方案：

```python
@dataclass(frozen=True)
class Settings:
    backend_dir: Path
    llm_provider: str
    ...

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    ...
```

### 8.1 三个设计点

1. **`frozen=True`**：配置一旦构造就不可变，避免运行期被某个环节偷偷改掉。
2. **`lru_cache(maxsize=1)`**：进程内单例 + 只解析一次 `.env`，比全局变量更干净，也比每次读环境变量快。
3. **读取优先级显式**：`_first_config_value` 先看 `backend/.env`，再回退系统环境变量——**本地开发方便，容器部署也能覆盖**。

### 8.2 与 `pydantic-settings` 的取舍

| | 本项目（dataclass + lru_cache） | `pydantic-settings` |
| --- | --- | --- |
| 依赖 | 无（dataclass 是标准库） | 需要额外包 |
| 类型转换/校验 | 手写 | 自动（int/float/bool/嵌套模型） |
| 前缀、嵌套、`.env` 多环境 | 手写 | 一行声明 |

本项目配置项只有十几个且以字符串为主，手写完全够用。配置项一旦超过二三十个、出现嵌套结构，就该换 `pydantic-settings`。

### 8.3 一个必须记住的坑

`get_settings` 带 `@lru_cache`，**测试里改了环境变量它不会重新读**。要么在 fixture 里 `get_settings.cache_clear()`，要么把 settings 也做成依赖项注入。

---

## 9. 中间件：CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

因为前端 `localhost:3000`、后端 `localhost:8004`，属于跨源，必须开 CORS。

**但注意**：`allow_origins=["*"]` 与 `allow_credentials=True` 的组合，浏览器规范上是互斥的（带凭据的请求不允许通配源）。当前项目没有用 Cookie/Authorization 凭据，所以没暴露问题。一旦引入登录态，必须把 `allow_origins` 换成白名单，否则要么请求直接被浏览器拦截，要么就得放开到不安全的地步。

**工程建议**：白名单从配置里读，写在 `Settings` 里，别硬编码。

---

## 10. 本地运行的两个坑（都是这次真实踩到的）

### 10.1 工作目录决定能不能启动

```text
# 在 backend/ 下执行 —— 正常
python -m uvicorn app:app --port 8004

# 在仓库根目录执行 —— 报错
ERROR: Error loading ASGI app. Could not import module "app".
```

原因：uvicorn 收到 `app:app` 会做 `importlib.import_module("app")`，它只在 `sys.path` 里找。`backend/` 没有 `__init__.py`（不是包），所以从根目录既找不到 `app`，也不能写 `backend.app:app`。

两种解法：

```powershell
# 解法一：切目录（README 的方式）
cd backend
.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8004

# 解法二：用 --app-dir 指定模块搜索路径
.\backend\.venv\Scripts\python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 8004
```

`--app-dir` 本质上就是"启动前把某个目录加进 `sys.path`"，等价于先 `cd` 再启动。

### 10.2 `--reload` 会多起进程

`--reload` 下 uvicorn 有一个"监控父进程 + 服务子进程"的结构，某些环境下还会出现启动器进程。排查启动问题时建议先关掉 `--reload`，避免看到一堆进程分不清谁是谁。

---

## 11. 新增一个接口的检查清单

照着这个项目的约定，加一个"标签"资源的 CRUD 应该是：

1. 新建 `backend/api/tags.py`，`router = APIRouter()`。
2. 写请求模型 `class CreateTagRequest(BaseModel)`，字段用 `Field` 加约束，枚举用 `Literal`。
3. 写 handler，路径不带 `/api`（前缀由 `app.py` 统一加），**必须在函数开头校验依赖是否就绪**（或者用 `Depends`）。
4. 业务逻辑放到下游模块（如 `memory/repository.py`），不在路由里写 SQL / 文件操作。
5. 在 `app.py` 里 `from api.tags import router as tags_router` 并 `include_router(..., prefix="/api", tags=["tags"])`。
6. 打开 `/docs` 确认分组、字段说明、校验规则都正确。
7. 需要访问外网/加载模型的初始化，放进 `lifespan`，**不要写在模块顶层**。

---

## 12. 自检清单：这个项目可能被问到的问题

对着这份清单自问自答一遍，FastAPI 的工程面试基本就覆盖了：

1. 为什么要用 `lifespan` 而不是顶层初始化？举出本项目里最典型的反例。（→ 第 1.2 节）
2. `APIRouter` 的 `prefix` 和 `tags` 分别影响什么？
3. `response_model` 和返回类型注解有什么区别？本项目为什么没用前者？
4. 为什么"资源未初始化"用 503 而不是 500？
5. 本项目为什么没有用 `Depends`？改造它对你的项目有什么收益？
6. SSE 的报文格式是什么？为什么结尾必须有两个换行？
7. 流式响应里出异常了，还能返回 500 吗？为什么？你打算怎么把错误告诉前端？
8. `async def` handler 里调同步阻塞函数会发生什么？`asyncio.to_thread` 解决了什么？
9. `asyncio.create_task` 做后台任务有什么风险？
10. `@lru_cache` 的配置单例在测试里有什么坑？
11. `allow_origins=["*"]` 配 `allow_credentials=True` 会怎样？
12. uvicorn 报 `Could not import module "app"` 时你会怎么排查？

---

## 附：本项目的分层边界（记住这张图，路由就不会写胖）

```text
HTTP 层        api/*.py
               ├─ 校验请求（Pydantic）
               ├─ 调用下游
               └─ 把结果/异常翻译成 HTTP 语义
                    ↓
编排层         graph/*、agents/*、knowledge_retrieval/*
               └─ 业务决策、流程编排、状态机
                    ↓
能力层         memory/*、tools/*
               └─ 数据库、文件、命令行、模型调用
```

判断一个路由函数是不是"写胖了"，用一句话就行：

> **路由层允许出现的东西：参数校验、依赖判空、调用下游、组装响应、抛 HTTPException。**
> **除此之外的一切（SQL、文件读写、循环里的业务判断、模型调用）都不该出现在 `api/` 里。**
