# Ragclaw

本项目是一个面向本地运行、文件优先、可审计的 Agent 工作台。

- 对话、工具调用、检索过程都会落到本地文件
- 长期记忆由 PostgreSQL + pgvector 的 `MemoryManager` 统一管理
- 技能不是黑盒函数，而是可读可改的 `SKILL.md`
- 前端可以直接看到流式回复、工具链路和检索证据

如果你想做一个“能解释自己为什么这样回答”的 Agent，这个仓库就是为这种方向准备的。

## 项目特点

- 本地优先：后端和前端都可以直接本地启动，不依赖 MySQL / Redis
- 文件即事实源：`workspace/` 和本地 `knowledge/` 都是可见、可改、可版本管理的文件
- Prompt 可解释：系统提示词由多个 Markdown 文件实时组装
- 技能可审计：每个技能都是 `skills/*/SKILL.md`
- 检索可观测：前端会展示知识检索步骤、证据来源和工具调用
- 编排可观测：Planner 的领域判定 / 任务拆解、下游 Agent 的 A2A 消息链都会推到前端

## 当前能力

- FastAPI + SSE 流式聊天
- 会话持久化到 `backend/sessions/*.json`
- 长期记忆：PostgreSQL + pgvector（`MemoryManager` 统一入口）
- 记忆自主召回：每轮对话前自动检索长期记忆，对话结束后按 importance 自主决定是否入库，无需开关
- Planner 自动分流：一次 LLM 调用同时完成领域判定（coding / rag / general）与编码任务拆解
- 代码助手多 Agent 编排：LangGraph 父子图 + supervisor / aggregate + A2A 消息总线，子任务支持依赖排序与并行派发
- 代码助手工具链：读写文件、grep、日志检索 / 统计、运行测试、git 状态 / 差异 / 历史 / 暂存 / 提交 / 分支
- 知识库 RAG 由 Planner 自动分流：判定为知识库领域时自动进入检索链路，无需开关
- 本地知识库检索
- 前端三栏工作台
- 在线编辑 Skills / Workspace 文件
- 检索证据、工具调用、Raw Messages 可视化

当前内置技能包括：

- `rag-skill`：本地知识库检索
- `coding-skill`：代码开发（阅读/分析代码、定位 Bug、修改代码、运行测试与验证）
- `code-modification`：面向仓库的精确代码修改（先读懂上下文，再最小改动、自检、复查差异）
- `log-analysis`：日志过滤、错误分布统计、报错时间线与根因定位
- `test-execution`：运行既有测试、复现问题、验证改动与回归
- `test-authoring`：按项目现有框架与风格补写单元 / 回归测试
- `git-management`：仓库状态 / 差异 / 历史 / 分支，暂存与提交（只提交不推送）
- `web-search`：联网搜索
- `get_weather`：天气查询
- `retry-lesson-capture`：失败经验沉淀

## 代码助手多 Agent 编排

当 Planner 判定领域为 `coding` 时，请求会进入代码助手多 Agent 链路：

```mermaid
flowchart TB
    U["用户请求"] --> M["AgentManager.astream()（SSE）"]
    M --> MEM["长期记忆检索"] --> P{"Planner：领域判定 + 任务拆解"}
    P -- "rag" --> RAG["知识库链路：knowledge_orchestrator + rag-skill"]
    P -- "general" --> GEN["通用问答：直连 LLM"]
    P -- "coding" --> SUP["supervisor（父 agent）"]
    SUP --> A1["log-analysis-agent"]
    SUP --> A2["code-modify-agent"]
    SUP --> A3["test-agent"]
    SUP --> A4["git-agent"]
    A1 --> SUP
    A2 --> SUP
    A3 --> SUP
    A4 --> SUP
    SUP -- "待办清空" --> AGG["aggregate：汇总子任务结果"]
    AGG --> O["最终答复（含 a2a_transcript）"]
```

- Planner 用一次 LLM 调用同时完成领域判定与任务拆解，输出 `{domain, reason, tasks[], depends_on}`
- 父图（`supervisor`）按依赖做稳定拓扑排序，依赖满足即并行 fan-out 到下游 Agent
- 下游 Agent 各自绑定一个 Skill 与工具白名单，内部复用现有 ReAct 子图执行，完成后回到 `supervisor` 继续派发
- Planner 与下游 Agent 之间通过进程内 A2A 消息总线（`task` / `result`）通信，总线每次请求独立新建
- 所有子任务跑完后由 `aggregate` 汇总：单任务直返、多任务由 LLM 汇总，失败项如实保留

相关代码位置：

- Planner：[backend/agents/planner.py](backend/agents/planner.py)
- 下游 Agent 与工具白名单：[backend/agents/registry.py](backend/agents/registry.py)
- LangGraph 父子图：[backend/agents/team_graph.py](backend/agents/team_graph.py)
- 子任务执行器：[backend/agents/executor.py](backend/agents/executor.py)
- A2A 消息总线：[backend/agents/a2a.py](backend/agents/a2a.py)
- 架构图：[docs/architecture-multi-agent.png](docs/architecture-multi-agent.png) / [docs/langgraph-team-graph.png](docs/langgraph-team-graph.png)

## 知识库目录说明

仓库自带示例知识库，位于 `backend/knowledge/`。

其中包含：

- FAQ / Markdown / JSON 示例数据
- PDF / Excel 等异构知识文件
- 顶层和子目录的 `data_structure.md`

你可以直接使用仓库内已有知识文件做体验，也可以继续向该目录补充自己的资料。

## 知识库检索链路

当前仓库已经实现了一条“Skill 优先，混合检索兜底”的知识检索链路。

### 已实现流程

```mermaid
flowchart LR
    U["用户问题"] --> A["Skill Router 三层决策"]
    A -- "选择 rag-skill" --> S["Skill Retriever Agent"]
    A -- "NONE / 其他技能" --> G["主 Agent（按需读取对应 SKILL.md）"]
    S --> D{"Skill 证据是否足够?"}
    D -- "是" --> R["直接基于 Skill 证据回答"]
    D -- "否" --> V["向量检索"]
    D -- "否" --> B["BM25 检索"]
    S --> F["RRF 融合"]
    V --> F
    B --> F
    F --> R
    R --> O["返回答案与引用证据"]
```

### 当前实现范围

- 请求进入 Agent 前，先由三层 Skill Router 决定技能：L1 指令判断（`/` 开头，占位）→ L2 向量（sentence-transformers）+ BM25（rank_bm25）+ RRF 混合检索 → L3 仅当 L2 不确定时才调用轻量 LLM 在候选中决策
- L2 唯一命中时直接采用、不调用路由 LLM；无候选时 L3 从全部技能中选择（此时只传 name + description）
- 路由选中 `rag-skill` 时进入知识检索链路，选中其他技能时主 Agent 先读取对应 SKILL.md 再执行，无合适技能则走普通对话
- `skill` 检索始终优先执行
- 当 `skill` 结果为 `partial / not_found / uncertain` 时，才触发混合检索兜底
- 混合检索当前由：
  - 向量检索
  - BM25 检索
  - RRF 融合
  组成
- 当前知识索引主要覆盖 `knowledge/` 中的 `.md` 和 `.json`
- Excel / PDF 仍主要依赖 `rag-skill` 的专门处理流程

相关代码位置：

- 后端入口：[backend/app.py](backend/app.py)
- Agent 主入口：[backend/graph/agent.py](backend/graph/agent.py)
- Skill 路由决策：[backend/graph/skill_router.py](backend/graph/skill_router.py)
- 知识检索编排：[backend/knowledge_retrieval/orchestrator.py](backend/knowledge_retrieval/orchestrator.py)
- 向量 / BM25 索引：[backend/knowledge_retrieval/indexer.py](backend/knowledge_retrieval/indexer.py)
- 混合检索：[backend/knowledge_retrieval/hybrid_retriever.py](backend/knowledge_retrieval/hybrid_retriever.py)
- 技能检索代理：[backend/knowledge_retrieval/knowledge_retriever_agent.py](backend/knowledge_retrieval/knowledge_retriever_agent.py)

## 系统结构

```text
ragclaw/
├─ backend/
│  ├─ api/                    # Chat、session、file、token、knowledge index 接口
│  ├─ graph/                  # Agent、prompt、session、memory 相关逻辑
│  ├─ agents/                 # 代码助手多 Agent：planner / registry / executor / team_graph / A2A 消息总线
│  ├─ knowledge/              # 仓库内置示例知识库
│  ├─ knowledge_retrieval/    # Skill 优先 + hybrid fallback 检索链路
│  ├─ memory/                 # 长期记忆：MemoryManager + pgvector Repository
│  ├─ scripts/                # 评测与辅助脚本
│  ├─ sessions/               # 会话历史
│  ├─ skills/                 # 技能目录，每个技能核心是 SKILL.md
│  ├─ storage/                # 索引、评测产物等缓存
│  ├─ tools/                  # terminal / read_file / write_file / edit_file / grep / log_* / run_tests / git_* / python_repl / fetch_url
│  ├─ workspace/              # SOUL / USER / AGENTS 等系统上下文组件
│  └─ app.py                  # FastAPI 入口
├─ frontend/
│  ├─ src/app/                # 页面入口
│  ├─ src/components/         # 聊天面板、检索面板、编辑器等 UI
│  └─ src/lib/                # API 客户端与状态管理
├─ docs/                     # 架构图与工程实践笔记
└─ README.md
```

## 技术栈

### 后端

- Python 3.10+
- FastAPI
- Uvicorn
- LangChain 1.x
- LangGraph（多 Agent 父子图编排）
- LlamaIndex
- PostgreSQL + pgvector（长期记忆）
- OpenAI-compatible model API
- Ragas（用于离线评估）

### 前端

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Monaco Editor

## 默认模型配置

代码层默认配置见 [backend/config.py](backend/config.py)：

- LLM Provider: `deepseek`
- LLM Model: `deepseek-v4-flash`
- Embedding Provider: `local`（sentence-transformers 本地模型，无需 API key）
- Embedding Model: `BAAI/bge-small-zh-v1.5`

示例环境文件 [backend/.env.example](backend/.env.example) 默认使用：

- LLM Provider: `deepseek`
- Embedding Provider: `local`

也支持：

- `local`（仅 embedding）
- `zhipu`
- `bailian`
- `deepseek`
- `openai`

## 环境变量

示例文件见 [backend/.env.example](backend/.env.example)。

当前配置读取顺序：

1. 优先读取 `backend/.env`
2. 若未配置，再回退到系统环境变量

最少建议配置：

```env
DEEPSEEK_API_KEY=your_key

EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

TAVILY_API_KEY=your_key
```

可选调参（写入 `backend/.env` 或系统环境变量）：

- `MEMORY_IMPORTANCE_THRESHOLD`：对话结束后记忆入库的重要度门限，默认 `0.7`
- `MEMORY_RETRIEVAL_MIN_SCORE`：长期记忆召回的最低余弦相似度，默认 `0.55`
- `SKILL_ROUTE_THRESHOLD`：Skill 路由 L2 的判定阈值，默认 `0.30`

## 快速开始

### 1. 启动后端

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app:app --host 127.0.0.1 --port 8004 --reload
```

后端健康检查：

```text
http://127.0.0.1:8004/health
```

### 2. 启动前端

```powershell
cd frontend
npm install
npm run dev
```

前端默认地址：

```text
http://127.0.0.1:3000
```

前端默认请求后端：

```text
http://127.0.0.1:8004/api
```

如果后端端口不是 `8004`，可以设置：

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8010/api"
```

## 5 分钟体验路径

1. 启动前后端，发送一条普通聊天消息，确认流式输出正常
2. 打开右侧 Inspector，查看工具调用和 Raw Messages
3. 直接提问，观察 Memory 检索步骤是否自动注入 Agent 上下文
4. 提一个知识库问题，观察 Skill 检索和 fallback 检索是否出现
5. 查看 `backend/sessions/*.json`，确认消息、工具和检索记录已落盘

## 常用接口

- `GET /health`
- `POST /api/chat`
- `GET /api/sessions`
- `GET /api/files`
- `POST /api/files`
- `GET /api/knowledge/index/status`
- `POST /api/knowledge/index/rebuild`

## 评测脚本

仓库内已经包含检索与评测相关脚本，产物写入 `backend/storage/eval_outputs/`。

- `scripts/build_rag_eval_dataset.py`：构建 RAG 评测数据集
- `scripts/ragas_offline_eval.py`：Ragas 离线评测（上下文召回率、忠实度等）
- `scripts/evaluate_faq_agent_retrieval.py` / `scripts/evaluate_faq_system_accuracy.py`：FAQ 检索与系统准确率评测
- `scripts/measure_planner_cache.py`：测量 Planner 规划链路的缓存命中率与 token 成本

适合做：

- 路由命中率评测
- FAQ 检索准确率评测
- Ragas 指标评测
- Planner 缓存命中率评测

## 当前边界

- 更适合本地开发和研究，不是完整的生产 SaaS
- Excel / PDF 的高级处理仍主要依赖技能链路
- 混合检索当前主要覆盖 Markdown / JSON 类知识文件
- 技能和知识链路仍在快速迭代中
- 代码助手 Agent 默认只做本地读写、测试与提交，不执行 push 等推送操作

## 致谢

`skill` 设计思路参考了 [ConardLi/rag-skill](https://github.com/ConardLi/rag-skill)。
