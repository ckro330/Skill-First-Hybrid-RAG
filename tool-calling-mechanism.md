# 工具调用机制与工作流（小型速查）

> 主题：本项目里 Agent 是如何调用工具的，以及 `agent.py`、LangGraph、`tools/*.py` 三者各自扮演什么角色。

---

## 1. 核心结论

`agent.py` 里的 `astream` **不直接调用工具**。它只是：

- 用 `create_agent()` 构建一个 LangGraph 图（这个图才是真正的 Agent）；
- 用 `agent.astream(..., stream_mode=["messages", "updates"])` 订阅图执行过程；
- 把图吐出来的事件翻译成 `tool_start` / `tool_end` 转发给前端。

真正执行工具的是 **LangGraph 图内部的 `tools` 节点（ToolNode）**，真正干活的代码在 `backend/tools/*.py` 的 `_run` 方法里。

## 2. create_agent 的 LangGraph 图结构

```text
        ┌─────────────────────────────────────────────┐
        ▼                                             │
   ┌─────────┐   有 tool_calls    ┌─────────┐         │
   │  model  │ ─────────────────► │  tools  │─────────┘
   │ (LLM)   │                    │ ToolNode│
   └─────────┘                    └─────────┘
        │
        │ 无 tool_calls（最终回答）
        ▼
       END
```

| 部分 | 作用 |
| --- | --- |
| 状态 `messages` | 对话历史，带 `add_messages` reducer 自动累加 |
| `model` 节点 | 绑定工具调用 LLM，输出 `AIMessage` |
| `tools` 节点 | `ToolNode`：执行工具调用，返回 `ToolMessage` |
| 条件边 | 模型输出带 `tool_calls` → `tools`；否则 → `END`；工具执行完 → 回到 `model` |

## 3. 工具从注册到执行的三阶段

```text
阶段① 注册绑定    create_agent(model, tools=[...])
                 → 工具登记进图（name → 实例 查表）
                 → bind_tools，把工具 schema 发给 LLM
                 → 此时工具一次都没执行

阶段② 模型声明    LLM 输出 tool_calls（"我想调 terminal"）
                 → 仍未执行，只是模型说了想调

阶段③ 真正执行    LangGraph 路由到 tools 节点
                 → ToolNode 按 name 查表找到实例
                 → tool.invoke() → BaseTool 内部 → _run()
                 → 生成 ToolMessage 回到 model 节点
```

## 4. 三方职责划分

| 角色 | 代码位置 | 职责 |
| --- | --- | --- |
| 编排者 | `backend/graph/agent.py`（`AgentManager.astream`） | 构建图、订阅流、翻译事件，**不碰工具** |
| 真正的 Agent | `create_agent()` 返回的编译图 | 决定调什么工具，驱动 model ↔ tools 循环 |
| 工具分发 | LangGraph `ToolNode` | 按 `tool.name` 查表，调 `tool.invoke()` |
| 工具本体 | `backend/tools/*.py`（如 `TerminalTool`） | 继承 `BaseTool`，实现 `_run()` / `_arun()`，真正执行命令/读文件 |

### 关键约定：LangGraph 怎么找到 `_run`

LangGraph 不"发现" `_run`，它只依赖两件事：

1. **查表**：`ToolNode` 构造时建 `{tool.name: tool}`，运行时用模型输出的 `tool_calls[].name` 查表；
2. **接口约定**：查表后只调 `tool.invoke()`，由 `BaseTool` 基类内部转发到 `_run`（同步）或 `_arun`（异步）。

```python
# langgraph ToolNode 内部（示意）
tool = self.tools_by_name["terminal"]          # 按名字找实例
result = tool.invoke({"command": "ls"})        # 公共接口

# langchain_core BaseTool（示意）
class BaseTool:
    def invoke(self, input, ...):
        return self._run(**input)   # 转发到子类实现的 _run
```

## 5. 项目中 tools 的注册链

```python
# backend/tools/__init__.py
def get_all_tools(base_dir: Path) -> list[BaseTool]:
    return [
        TerminalTool(root_dir=base_dir),
        PythonReplTool(root_dir=base_dir),
        FetchURLTool(),
        ReadFileTool(root_dir=base_dir),
    ]

# backend/graph/agent.py
self.tools = get_all_tools(base_dir)
agent = create_agent(
    model=self._build_chat_model(),
    tools=self.tools if tools_override is None else tools_override,
    system_prompt=...,
)
```

```text
agent.py: get_all_tools() → 实例化工具类
   ↓
create_agent(tools=[...]) → ToolNode 建表 {name: 实例}，模型 bind_tools
   ↓
模型输出 tool_calls[{name: "terminal", args: {...}}]
   ↓
ToolNode: tools_by_name["terminal"] → TerminalTool 实例
   ↓
tool.invoke(...) → BaseTool 内部 → self._run(...)  ← tools/terminal_tool.py
```

## 6. astream 里如何观察到工具调用

`astream` 订阅 `stream_mode=["messages", "updates"]` 两个通道：

- **messages 通道**：`model` 节点的 token 级输出 → `token` 事件（按 `metadata["langgraph_node"] == "model"` 过滤）；
- **updates 通道**：每执行完一个节点的状态更新。看到带 `tool_calls` 的 AI 消息 → `tool_start`；看到 `ToolMessage`（`type == "tool"`）→ `tool_end` + `new_response`。

```python
# agent.py（观察逻辑示意）
if tool_calls:
    pending_tools[call_id] = {"tool": tool_name, "input": str(tool_args)}
    yield {"type": "tool_start", "tool": tool_name, "input": str(tool_args)}

if message_type == "tool":
    pending = pending_tools.pop(tool_call_id, ...)
    yield {"type": "tool_end", "tool": pending["tool"], "output": output}
    yield {"type": "new_response"}
```

`tool_start` 与 `tool_end` 之间的间隔，就是 `ToolNode` 执行工具的耗时；`pending_tools` 用 `tool_call_id` 配对，处理多工具调用的乱序返回。

## 7. 两条分支的差异

| 分支 | 是否走 LangGraph 图 | 是否有工具调用 |
| --- | --- | --- |
| 通用对话 | 走 `create_agent` 的图 | 有：model ↔ tools 循环，可多轮 |
| 知识库查询 | 不走主图，直接 `_astream_model_answer` 调 LLM | 无：检索由独立的 `skill_retriever_agent` 完成 |

注意：知识库分支的"无工具"是指**主回答**阶段；检索阶段的 `skill_retriever_agent` 是另一个独立的 LangGraph 图，它内部有自己的工具调用（terminal / read_file / python_repl）。

## 8. 常见疑问

**传 `tools=...` 给 create_agent 是一次调用吗？**
不是。那只是注册绑定。真正执行次数由模型决定：一条消息可能一次都不调，也可能调多轮。

**主程序不是 agent 吗？为什么工具不在 agent.py 里调？**
"Agent"指 `create_agent()` 返回的编译图；`agent.py` 的 `astream` 只是入口编排者。工具调用发生在图的 `tools` 节点内部。

**LangGraph 会自己去找 `_run` 吗？**
会，但靠的是注册表（`name → 实例`）加接口约定（`invoke → _run`），不是运行时反射发现。

---

关联文件：`backend/graph/agent.py`、`backend/tools/__init__.py`、`backend/tools/*.py`、`frontend/src/components/chat/ThoughtChain.tsx`
