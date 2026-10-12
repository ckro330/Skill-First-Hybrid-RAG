from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:  # pragma: no cover - optional dependency at runtime
    ChatDeepSeek = None

from agents.executor import AgentExecutor
from agents.orchestrator import CodingOrchestrator
from agents.planner import DOMAIN_CODING, DOMAIN_GENERAL, DOMAIN_RAG, Planner, planner
from config import get_settings
from graph.agent_state import MAX_STEPS, STATUS_RUNNING
from graph.context_builder import AgentContextBuilder
from graph.llm_usage import usage_collector
from graph.prompt_builder import build_system_prompt
from graph.react_agent import build_react_agent
from graph.session_manager import SessionManager
from graph.skill_loader import SkillLoader
from graph.skill_router import RouteDecision, skill_router
from graph.skill_tool_registry import SkillToolRegistry
from knowledge_retrieval import knowledge_orchestrator
from memory.memory_manager import MemoryManager, MemoryRecord
from memory.repository import MEMORY_TYPES
from tools import get_all_tools

KNOWLEDGE_SKILL_ID = "rag-skill"
MEMORY_IMPORTANCE_THRESHOLD = 0.7
logger = logging.getLogger("react_agent")


#返回文本内容
def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def _parse_memory_analysis(text: str) -> dict[str, Any]:
    """解析记忆分析模型的 JSON 输出，失败时返回全 0 默认值。"""
    default: dict[str, Any] = {"importance": 0.0, "type": "preference", "content": ""}
    try:
        cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().strip("`")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\"importance\"\s*:\s*([0-9.]+)", text)
        if not match:
            return default
        data = {"importance": float(match.group(1))}

    importance = max(0.0, min(1.0, float(data.get("importance") or 0)))
    type_ = str(data.get("type") or "preference")
    if type_ not in MEMORY_TYPES:
        type_ = "preference"
    content = str(data.get("content") or "").strip()
    return {"importance": importance, "type": type_, "content": content}


def _history_contains(history: list[dict[str, Any]], text: str) -> bool:
    """history 里是否已经包含这段文本。

    load_session_for_agent 会把压缩摘要并入首条 assistant 消息，
    而对话接口又会单独传一次 compressed_context；不去重的话同一段摘要
    会占两份 prompt（纯浪费 token）。
    """
    needle = (text or "").strip()
    if not needle:
        return False
    return any(needle in str(item.get("content", "") or "") for item in history)


class AgentManager:
    def __init__(self) -> None:
        self.base_dir: Path | None = None
        self.session_manager: SessionManager | None = None
        self.tools = []
        self.skill_loader: SkillLoader | None = None
        self.skill_tool_registry: SkillToolRegistry | None = None
        self.context_builder: AgentContextBuilder | None = None
        self.memory_manager: MemoryManager | None = None
        self.planner: Planner | None = None
        self.agent_executor: AgentExecutor | None = None
        self.coding_orchestrator: CodingOrchestrator | None = None

    def initialize(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.session_manager = SessionManager(base_dir)
        self.tools = get_all_tools(base_dir)
        self.skill_loader = SkillLoader(base_dir)
        self.skill_tool_registry = SkillToolRegistry(base_dir)
        self.context_builder = AgentContextBuilder(base_dir)
        self.memory_manager = MemoryManager(base_dir)
        self.memory_manager.initialize()
        knowledge_orchestrator.configure(base_dir, self._build_chat_model)
        skill_router.configure(base_dir, self._build_chat_model)
        planner.configure(base_dir, self._build_chat_model)
        self.planner = planner
        self.agent_executor = AgentExecutor(
            base_dir,
            self._build_chat_model,
            self.skill_loader,
            self.skill_tool_registry,
        )
        # A2A 总线由 orchestrator 每次请求内部新建，避免跨会话消息互相污染
        self.coding_orchestrator = CodingOrchestrator(
            base_dir,
            self._build_chat_model,
            self.agent_executor,
        )

#配置对话模型，没有配置好deepseek的时候替换
    def _build_chat_model(self, callbacks: list[Any] | None = None):
        settings = get_settings()
        # 所有 LLM 调用都经由这里构造，因此把用量采集统一挂在这一处
        handlers: list[Any] = [usage_collector, *(callbacks or [])]

        if settings.llm_provider == "deepseek":
            if ChatDeepSeek is None:
                raise RuntimeError("langchain-deepseek is not installed")
            if not settings.llm_api_key:
                raise RuntimeError("Missing API key for provider deepseek")
            return ChatDeepSeek(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=0,
                callbacks=handlers,
            )

        if not settings.llm_api_key:
            raise RuntimeError(f"Missing API key for provider {settings.llm_provider}")

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
            callbacks=handlers,
        )
    
#构建上下文
    def _build_messages(self, history: list[dict[str, Any]]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in history:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": str(item.get("content", ""))})
        return messages

#格式化检索到的上下文
    def _format_retrieval_context(self, results: list[MemoryRecord]) -> str:
        lines = ["[Memory retrieved context]"]
        for idx, item in enumerate(results, start=1):
            text = str(item.content).strip()
            source = f"memory:{item.id}"
            lines.append(f"{idx}. Source: {source} (type={item.type})\n{text}")
        return "\n\n".join(lines)

#整理检索结果，以便后续产出
    def _format_memory_retrieval_step(self, results: list[MemoryRecord]) -> dict[str, Any]:
        return {
            "kind": "memory",   
            "stage": "memory",
            "title": f"Memory 检索到 {len(results)} 条记忆",
            "message": "已将 Memory 召回结果注入当前请求上下文。",
            "results": [
                {
                    "source_path": f"memory:{item.id}",
                    "source_type": "memory",
                    "locator": item.type,
                    "snippet": str(item.content).strip(),
                    "channel": "memory",
                    "score": float(item.score) if item.score is not None else None,
                    "parent_id": None,
                }
                for item in results
            ],
        }

#整理 Conversation Context（方案第六节：检索 + 压缩，不注入全部历史）
    def _build_conversation_context(
        self,
        history: list[dict[str, Any]],
        retrievals: list[MemoryRecord],
        compressed_context: str,
        *,
        include_recent_turns: bool = True,
        limit: int = 4000,
    ) -> str:
        """组装当轮动态上下文（压缩摘要 + 记忆召回 + 可选的历史简略）。

        include_recent_turns 只在"没有把真实历史交给模型"的链路上开启
        （代码助手下游子 Agent）；主 Agent 已经把完整历史作为 messages 传入，
        再塞一份"最近对话（简略）"既是重复，也会让前缀每轮变化。
        """
        lines: list[str] = []
        if compressed_context:
            lines.append("[之前对话摘要]\n" + compressed_context.strip())
        if retrievals:
            lines.append(self._format_retrieval_context(retrievals))

        if include_recent_turns:
            recent = [
                item for item in history[-6:] if item.get("role") in {"user", "assistant"}
            ]
            turns: list[str] = []
            for item in recent:
                role = "用户" if item.get("role") == "user" else "助手"
                content = str(item.get("content", "") or "").strip().replace("\n", " ")[:200]
                if content:
                    turns.append(f"- {role}: {content}")
            if turns:
                lines.append("[最近对话（简略）]\n" + "\n".join(turns))

        context = "\n\n".join(lines)
        if len(context) > limit:
            return context[:limit] + "\n...[truncated]"
        return context

#把历史转成 LangChain 消息，并把当轮动态上下文并进最后一条 user 消息
    def _build_agent_messages(
        self,
        history: list[dict[str, Any]],
        message: str,
        turn_context: str,
    ) -> list[Any]:
        """历史只追加、不重写；动态内容只出现在末尾。

        这样上一轮的请求才是这一轮请求的**前缀**，上下文缓存才可能命中。
        反过来，只要把动态内容塞回 system，前缀从第一个 token 就断了。
        """
        messages: list[Any] = []
        for item in self._build_messages(history):
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            else:
                messages.append(AIMessage(content=item["content"]))

        if turn_context.strip():
            section = self.context_builder.build_conversation_context_section(turn_context)
            content = f"{section}\n\n---\n\n{message}"
        else:
            content = message
        messages.append(HumanMessage(content=content))
        return messages

#整理knowledge检索结果，判断是否检索成功或者找到检索状态
    def _format_knowledge_context(self, retrieval_result) -> str:
        lines = ["[Knowledge retrieval evidence]"]
        lines.append(f"Status: {retrieval_result.status}")
        if retrieval_result.reason:
            lines.append(f"Reason: {retrieval_result.reason}")
        if retrieval_result.fallback_used:
            lines.append("Fallback: skill evidence was insufficient, so vector/BM25 retrieval was used.")
        if not retrieval_result.evidences:
            lines.append("No direct evidence was found.")
            return "\n".join(lines)

        for index, evidence in enumerate(retrieval_result.evidences, start=1):
            lines.append(
                f"{index}. [{evidence.channel}] {evidence.source_path} ({evidence.locator})\n{evidence.snippet}"
            )
        return "\n\n".join(lines)

#构造进行knowledge检索的提示词
    @staticmethod
    def _knowledge_answer_instructions() -> list[str]:
        """这些指令是常量：一旦掺入每次检索都不同的内容，system 前缀就失效了。

        检索状态与原因已经在 evidence 区块里（_format_knowledge_context），
        不需要再重复一遍。
        """
        return [
            "This is a knowledge-base question.",
            "Use only the provided knowledge retrieval evidence to answer.",
            "Do not perform additional knowledge-base inspection with tools.",
            "If the evidence is incomplete, explicitly say the current knowledge base only supports a partial answer or no direct answer.",
            "Do not fabricate facts.",
            "When evidence is insufficient, suggest narrowing the scope by directory, file, keyword, field name, or time range.",
            "Cite the file paths you relied on.",
        ]

#异步流式的agent输出答案，和stream：true用法相似，这里调用大模型生成答案
#这里的异步async负责异步输出多轮yield结果
    async def _astream_model_answer(
        self,
        messages: list[dict[str, str]],
        extra_instructions: list[str] | None = None,
    ):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")

        system_prompt = build_system_prompt(self.base_dir)
        if extra_instructions:
            system_prompt = f"{system_prompt}\n\n" + "\n\n".join(extra_instructions)

        model_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        model_messages.extend(messages)

        final_content_parts: list[str] = []
#这里的异步是异步输出结果，当大模型等待网络I/O操作时，async for可以让其他任务继续执行，提升效率
        async for chunk in self._build_chat_model().astream(model_messages):
            text = _stringify_content(getattr(chunk, "content", ""))
            if text:
                final_content_parts.append(text)
                yield {"type": "token", "content": text}

        yield {"type": "done", "content": "".join(final_content_parts).strip()}


    async def astream(
        self,
        message: str,
        history: list[dict[str, Any]],
        *,
        compressed_context: str = "",
    ):
        if self.base_dir is None:
            raise RuntimeError("AgentManager is not initialized")

        # 压缩摘要已经并入 history（load_session_for_agent），这里避免再注入一遍
        if _history_contains(history, compressed_context):
            compressed_context = ""

#长期记忆检索：每轮自主召回，不再依赖开关；coding / rag / general 三条链路都会用到，
#因此必须放在 planner 分流之前。未配置或 PostgreSQL 不可用时自动降级为空。
        retrievals: list[MemoryRecord] = []
        augmented_history = list(history)
        if self.memory_manager is not None and self.memory_manager.is_available:
            retrievals = await self.memory_manager.retrieve_memories(
                user_id=self.memory_manager.default_user_id,
                query=message,
                top_k=3,
            )
            if retrievals:
                #这里是创建一个新字典，**负责把上面的检索结果字典展开成键值对，和前面的type一起组成一个新的字典
                yield {"type": "retrieval", **self._format_memory_retrieval_step(retrievals)}
                augmented_history.append(
                    {
                        "role": "assistant",
                        "content": self._format_retrieval_context(retrievals),
                    }
                )

#外层 Planner：判定领域（coding / rag / general），coding 时拆解并派发给下游 Agent
        planner_decision = None
        if self.planner is not None:
            try:
                planner_decision = await self.planner.decide(message, history)
            except Exception:
                logger.exception("[Planner] 决策异常，放行到默认链路")
                planner_decision = None
        if planner_decision is not None:
            yield {
                "type": "planner",
                "domain": planner_decision.domain,
                "reason": planner_decision.reason,
                "stage": planner_decision.stage,
                "multi_agent": planner_decision.multi_agent,
                "tasks": [task.to_dict() for task in planner_decision.tasks],
            }
            if (
                planner_decision.domain == DOMAIN_CODING
                and self.coding_orchestrator is not None
            ):
                coding_context = self._build_conversation_context(
                    history,
                    retrievals,
                    compressed_context,
                )
                async for event in self.coding_orchestrator.astream(
                    message,
                    planner_decision,
                    conversation_context=coding_context,
                ):
                    yield event
                return

#Skill 路由决策：三层路由（L1 指令 / L2 混合检索 / L3 轻量 LLM）
        route_decision: RouteDecision | None = None
        if planner_decision is not None and planner_decision.domain == DOMAIN_RAG:
            route_decision = RouteDecision(
                kind="skill",
                skill_id=KNOWLEDGE_SKILL_ID,
                stage="planner",
                reason=f"Planner 判定为知识库领域：{planner_decision.reason}",
            )
        elif planner_decision is not None and planner_decision.domain == DOMAIN_GENERAL:
            route_decision = RouteDecision(
                kind="none",
                stage="planner",
                reason=f"Planner 判定为通用问答：{planner_decision.reason}",
            )
        if route_decision is None:
            try:
                route_decision = await skill_router.route(message)
            except Exception:
                # 路由失败时放行到通用分支，主 Agent 仍可按快照自行读取技能
                route_decision = RouteDecision(
                    kind="none",
                    stage="fallback",
                    reason="路由调用失败，放行到通用分支。",
                )
        logger.info(
            "[SkillRouter] selected skill = %s (stage=%s, candidates=%s, reason=%s)",
            route_decision.skill_id or "NONE",
            route_decision.stage,
            route_decision.candidates,
            route_decision.reason,
        )
        yield {
            "type": "skill_route",
            "skill": route_decision.skill_id or "NONE",
            "stage": route_decision.stage,
            "candidates": route_decision.candidates,
            "command": route_decision.command,
            "reason": route_decision.reason,
        }

        if route_decision.kind == "command":
            command_content = skill_router.handle_command(route_decision.command or "")
            yield {"type": "token", "content": command_content}
            yield {"type": "done", "content": command_content}
            return

        selected_skill = route_decision.skill_id or "NONE"

#知识库技能（rag-skill）：走 Skill 优先 + 混合检索兜底链路
        if selected_skill == KNOWLEDGE_SKILL_ID:
            knowledge_result = None
            async for event in knowledge_orchestrator.astream(message):
                if event.get("type") == "orchestrated_result":
                    knowledge_result = event["result"]
                    continue
                yield event

            if knowledge_result is not None:
                for step in knowledge_result.steps:
                    yield {"type": "retrieval", **step.to_dict()}
                augmented_history.append(
                    {
                        "role": "assistant",
                        "content": self._format_knowledge_context(knowledge_result),
                    }
                )

            messages = self._build_messages(augmented_history)
            messages.append({"role": "user", "content": message})

            async for event in self._astream_model_answer(
                messages,
                extra_instructions=self._knowledge_answer_instructions() if knowledge_result else None,
            ):
                yield event
            return

#通用 ReAct 分支（方案第三、五、六节）
        loaded_skill = None
        if selected_skill != "NONE":
            if self.skill_loader is None:
                raise RuntimeError("AgentManager is not initialized")
            loaded_skill = self.skill_loader.load(selected_skill)
            if loaded_skill is None:
                logger.warning(
                    "[SkillRouter] skill %s 未找到，放行到通用分支",
                    selected_skill,
                )
                selected_skill = "NONE"

        if self.skill_tool_registry is None or self.context_builder is None:
            raise RuntimeError("AgentManager is not initialized")

        # Skill → Tool 绑定：第一版默认全量 4 个工具
        tools = self.skill_tool_registry.get_tools_for_skill(
            loaded_skill.skill_id if loaded_skill is not None else None
        )
        # 主 Agent 会把完整历史作为 messages 传给模型，这里不再重复"最近对话（简略）"
        conversation_context = self._build_conversation_context(
            history,
            retrievals,
            compressed_context,
            include_recent_turns=False,
        )
        initial_task_context: dict[str, Any] = {
            "goal": message,
            "completed": [],
            "findings": [],
            "missing": [],
        }
        # system 只放静态内容（角色 + 当前技能），保证同 skill 下逐字节恒定
        system_content = self.context_builder.build_system_content(loaded_skill)

        def system_message_builder(_state: dict[str, Any]) -> str:
            return system_content

        agent = build_react_agent(
            self._build_chat_model(),
            tools,
            system_message_builder=system_message_builder,
            max_steps=MAX_STEPS,
        )
        messages = self._build_agent_messages(history, message, conversation_context)
        input_state = {
            "messages": messages,
            "current_skill": loaded_skill.skill_id if loaded_skill is not None else None,
            "task_context": initial_task_context,
            "conversation_context": conversation_context,
            "plan": [],
            "status": STATUS_RUNNING,
            "steps": 0,
            "decision_reason": "",
        }

        final_content_parts: list[str] = []
        last_ai_message = ""
        pending_tools: dict[str, dict[str, str]] = {}

#开始处理整个工作流的的节点
        async for mode, payload in agent.astream(
            input_state,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, metadata = payload
                #只取大模型节点的输出，最后归到text
                if metadata.get("langgraph_node") != "model":
                    continue
                text = _stringify_content(getattr(chunk, "content", ""))
                if text:
                    final_content_parts.append(text)
                    yield {"type": "token", "content": text}
                continue

#updates是langgraph的数据载体，节点之间的过程，此处没有到updates说明还没到下一个节点
            if mode != "updates":
                continue

            for update in payload.values():
                for agent_message in update.get("messages", []):
                    message_type = getattr(agent_message, "type", "")
                    tool_calls = getattr(agent_message, "tool_calls", []) or []

                    if message_type == "ai" and not tool_calls:
                        candidate = _stringify_content(getattr(agent_message, "content", ""))
                        if candidate:
                            last_ai_message = candidate
                            logger.info("[Agent] next_action = final_answer")

#如果解析到了toolcall，下面是对大模型返回的toolcall进行预处理，后面执行
                    if tool_calls:
                        for tool_call in tool_calls:
                            call_id = str(tool_call.get("id") or tool_call.get("name"))
                            tool_name = str(tool_call.get("name", "tool"))
                            tool_args = tool_call.get("args", "")
                            #保证tool_args是字符串，如果不是字符串就转成json字符串，方便后面input都是字符串统一管理
                            if not isinstance(tool_args, str):
                                tool_args = json.dumps(tool_args, ensure_ascii=False)
                            pending_tools[call_id] = {
                                "tool": tool_name,
                                "input": str(tool_args),
                            }
                            logger.info(
                                "[Agent] tool_call = %s, input = %s",
                                tool_name,
                                str(tool_args)[:500],
                            )
                            logger.info("[Agent] next_action = tool_call")
                            yield {
                                "type": "tool_start",#收到了toolcalls，决定调用工具
                                "tool": tool_name,
                                "input": str(tool_args),
                            }

                    if message_type == "tool":
                        tool_call_id = str(getattr(agent_message, "tool_call_id", ""))
                        pending = pending_tools.pop(
                            tool_call_id,
                            {"tool": getattr(agent_message, "name", "tool"), "input": ""},
                        )
                        output = _stringify_content(getattr(agent_message, "content", ""))
#agent.py 只负责把工具类实例化并注册进图，_run 是工具类自己实现的方法，两者之间没有直接调用，中间隔着 create_agent（注册）和 BaseTool.invoke（分发），有langgraph自己查找，它找的是 name → 实例 的注册表，找到实例后调 invoke()，再由 BaseTool 统一转到该实例的 _run/_arun
                        logger.info("[Tool] observation = %s", output[:500])
                        yield {
                            "type": "tool_end",
                            "tool": pending["tool"],
                            "output": output,
                        }
                        yield {"type": "new_response"}

        final_content = "".join(final_content_parts).strip() or last_ai_message.strip()
        logger.info("[Agent] final_answer = %s", final_content[:300])
        yield {"type": "done", "content": final_content}

#调用大模型生成一个标题
    async def generate_title(self, first_user_message: str) -> str:
        prompt = (
            "请根据用户的第一条消息生成一个中文会话标题。"
            "要求不超过 10 个汉字，不要带引号，不要解释。"
        )
        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": first_user_message},
                ]
            )
            title = _stringify_content(getattr(response, "content", "")).strip()
            return title[:10] or "新会话"
        except Exception:
            return (first_user_message.strip() or "新会话")[:10]

#摘要
    async def summarize_history(self, messages: list[dict[str, Any]]) -> str:
        prompt = (
            "请将以下对话压缩成中文摘要，控制在 500 字以内。"
            "重点保留用户目标、已完成步骤、重要结论和未解决事项。"
        )
        lines: list[str] = []
        for item in messages:
            role = item.get("role", "assistant")
            content = str(item.get("content", "") or "")
            if content:
                lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)

        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ]
            )
            summary = _stringify_content(getattr(response, "content", "")).strip()
            return summary[:500]
        except Exception:
            return transcript[:500]

#记忆分析：判断本轮对话是否值得写入长期记忆
    async def analyze_memory_candidate(
        self,
        user_message: str,
        assistant_reply: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = (
            "你是长期记忆分析师，根据 importance 标准判断下面这轮对话是否值得写入长期记忆。\n\n"
            "importance 标准（0~1）：\n"
            "- 0.9~1.0：用户明确要求记住的信息、身份/偏好声明、长期承诺\n"
            "- 0.75~0.9：可复用的方法论、经验教训、跨任务有价值的偏好或事实\n"
            "- 0.5~0.75：仅对当前任务有参考价值的细节\n"
            "- 0.2~0.5：一次性事实、临时请求\n"
            "- 0~0.2：寒暄、无关闲聊\n\n"
            "只输出一个 JSON 对象，不要输出任何其他内容：\n"
            '{"importance": <0~1的小数>, "type": "<profile|preference|project|episodic>", '
            '"content": "<精炼后可复用的记忆文本；importance<=0.7 时 content 必须为空字符串>"}\n\n'
            "对话内容：\n{transcript}"
        )

        lines: list[str] = []
        for item in history[-10:]:
            role = item.get("role", "assistant")
            content = str(item.get("content", "") or "").strip()
            if content:
                lines.append(f"{role}: {content[:300]}")
        lines.append(f"user: {str(user_message).strip()[:500]}")
        lines.append(f"assistant: {str(assistant_reply).strip()[:800]}")
        transcript = "\n".join(lines) or "（空对话）"

        try:
            response = await self._build_chat_model().ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ]
            )
            text = _stringify_content(getattr(response, "content", ""))
        except Exception:
            logger.exception("[Memory] 记忆分析调用失败")
            return {"importance": 0.0, "type": "preference", "content": ""}
        return _parse_memory_analysis(text)

    async def _store_memory_candidate(
        self,
        analysis: dict[str, Any],
    ) -> MemoryRecord | None:
        manager = self.memory_manager
        if manager is None or not manager.is_available:
            return None
        # 轻量去重：与已有记忆正文完全一致，或存在包含关系时跳过
        try:
            existing = await manager.list_memories(
                user_id=manager.default_user_id,
                status="active",
                limit=500,
            )
            normalized = analysis["content"].strip()
            for record in existing:
                existing_content = record.content.strip()
                if (
                    existing_content == normalized
                    or (normalized and existing_content and normalized in existing_content)
                    or (normalized and existing_content and existing_content in normalized)
                ):
                    return record
        except Exception:
            logger.exception("[Memory] 记忆去重查询失败")
        return await manager.add_memory(
            user_id=manager.default_user_id,
            content=analysis["content"],
            type=analysis["type"],
            importance=analysis["importance"],
        )

    async def process_memory_after_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """对话结束后：分析 importance，标注 session JSON，超阈值写入长期记忆。"""
        settings = get_settings()
        threshold = settings.memory_importance_threshold
        analysis = await self.analyze_memory_candidate(
            user_message,
            assistant_reply,
            history,
        )
        importance = float(analysis.get("importance") or 0)

        if self.session_manager is not None:
            self.session_manager.set_last_turn_importance(session_id, importance)

        if importance > threshold and analysis.get("content"):
            record = await self._store_memory_candidate(analysis)
            if record is not None:
                logger.info(
                    "[Memory] importance=%.2f > %.2f，已写入记忆 id=%s type=%s",
                    importance,
                    threshold,
                    record.id,
                    record.type,
                )
        else:
            logger.info(
                "[Memory] importance=%.2f（阈值 %.2f），本轮不写入长期记忆",
                importance,
                threshold,
            )
        return analysis


agent_manager = AgentManager()
