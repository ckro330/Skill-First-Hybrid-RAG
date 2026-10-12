from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from graph.agent import agent_manager
from graph.llm_usage import begin_turn, end_turn

router = APIRouter()
logger = logging.getLogger("chat")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str
    stream: bool = True


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _new_segment() -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [],
        "retrieval_steps": [],
        "plan": None,
        "agents": [],
        "a2a": [],
        "a2a_transcript": [],
    }


def _segment_has_content(segment: dict[str, Any]) -> bool:
    return bool(
        segment["content"].strip()
        or segment["tool_calls"]
        or segment["retrieval_steps"]
        or segment["plan"]
        or segment["agents"]
        or segment["a2a"]
        or segment["a2a_transcript"]
    )


@router.post("/chat")
async def chat(payload: ChatRequest):
#寻找用户的对话id
    session_manager = agent_manager.session_manager
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager is not initialized")

    history_record = session_manager.load_session_record(payload.session_id)
    history = session_manager.load_session_for_agent(payload.session_id)
    is_first_user_message = not any(
        message.get("role") == "user"
        for message in history_record.get("messages", [])
    )

    async def event_generator():
        segments: list[dict[str, Any]] = []
        current_segment = _new_segment()
        conversation_saved = False
        # 从本轮第一个 LLM 调用开始采集 token / 缓存命中
        begin_turn()

        def persist_segments(fallback_content: str | None = None) -> None:
            #塞入current_segment和conversation_saved，在调用的时候改变
            nonlocal current_segment, conversation_saved
            if conversation_saved:
                return

            if fallback_content:
                if current_segment["content"].strip():
                    current_segment["content"] = (
                        f"{current_segment['content'].rstrip()}\n\n{fallback_content}"
                    )
                else:
                    current_segment["content"] = fallback_content

            if _segment_has_content(current_segment):
                segments.append(current_segment)
                current_segment = _new_segment()

            session_manager.save_message(payload.session_id, "user", payload.message)
            for segment in segments:
                session_manager.save_message(
                    payload.session_id,
                    "assistant",
                    segment["content"],
                    tool_calls=segment["tool_calls"] or None,
                    retrieval_steps=segment["retrieval_steps"] or None,
                    plan=segment["plan"],
                    agents=segment["agents"] or None,
                    a2a=segment["a2a"] or None,
                    a2a_transcript=segment["a2a_transcript"] or None,
                )

            conversation_saved = True

        try:
            async for event in agent_manager.astream(
                payload.message,
                history,
                compressed_context=history_record.get("compressed_context", ""),
            ):
                event_type = event["type"]

                if event_type == "token":
                    current_segment["content"] += event.get("content", "")
                elif event_type == "tool_start":
                    current_segment["tool_calls"].append(
                        {
                            "tool": event.get("tool", "tool"),
                            "input": event.get("input", ""),
                            "output": "",
                        }
                    )
                elif event_type == "tool_end":
                    if current_segment["tool_calls"]:
                        current_segment["tool_calls"][-1]["output"] = event.get("output", "")
                elif event_type == "retrieval":
                    current_segment["retrieval_steps"].append(
                        {
                            "kind": event.get("kind", "knowledge"),
                            "stage": event.get("stage", "unknown"),
                            "title": event.get("title", "检索结果"),
                            "message": event.get("message", ""),
                            "results": event.get("results", []),
                        }
                    )
                elif event_type == "planner":
                    current_segment["plan"] = {
                        "domain": event.get("domain", ""),
                        "reason": event.get("reason", ""),
                        "stage": event.get("stage", ""),
                        "multi_agent": bool(event.get("multi_agent")),
                        "tasks": event.get("tasks", []),
                    }
                elif event_type == "plan":
                    existing_plan = current_segment["plan"] or {}
                    current_segment["plan"] = {
                        "domain": event.get("domain", existing_plan.get("domain", "")),
                        "reason": event.get("reason", existing_plan.get("reason", "")),
                        "stage": existing_plan.get("stage", "orchestrator"),
                        "multi_agent": bool(
                            event.get("multi_agent", existing_plan.get("multi_agent", False))
                        ),
                        "tasks": event.get("tasks", existing_plan.get("tasks", [])),
                    }
                elif event_type == "agent_start":
                    current_segment["agents"].append(
                        {
                            "task_id": event.get("task_id", ""),
                            "agent_id": event.get("agent_id", ""),
                            "agent_name": event.get("agent_name", ""),
                            "title": event.get("title", ""),
                            "status": "running",
                        }
                    )
                elif event_type == "a2a":
                    message = event.get("message")
                    if isinstance(message, dict):
                        current_segment["a2a"].append(message)
                elif event_type == "agent_result":
                    for run in current_segment["agents"]:
                        if run.get("task_id") == event.get("task_id"):
                            run["status"] = "done"
                    if not current_segment["content"].strip() and event.get("content"):
                        current_segment["content"] = str(event["content"])
                elif event_type == "a2a_transcript":
                    messages = event.get("messages")
                    if isinstance(messages, list):
                        current_segment["a2a_transcript"] = messages
                elif event_type == "new_response":
                    if _segment_has_content(current_segment):
                        segments.append(current_segment)
                    current_segment = _new_segment()
                elif event_type == "done":
                    if not current_segment["content"].strip() and event.get("content"):
                        current_segment["content"] = event["content"]
                    persist_segments()

                data = {key: value for key, value in event.items() if key != "type"}
                yield _sse(event_type, data)

                if event_type == "done":
                    # 本轮所有 LLM 调用的用量（含上下文缓存命中）随对话一起落盘
                    usage = end_turn()
                    session_manager.set_last_turn_usage(payload.session_id, usage)
                    yield _sse("usage", usage)

                if event_type == "done" and is_first_user_message:
                    title = await agent_manager.generate_title(payload.message)
                    session_manager.set_title(payload.session_id, title)
                    yield _sse(
                        "title",
                        {"session_id": payload.session_id, "title": title},
                    )

                if event_type == "done":
                    # 对话结束后的长期记忆分析：标注 importance，超阈值写入记忆库
                    try:
                        reply = "\n\n".join(
                            segment["content"]
                            for segment in segments
                            if segment["content"].strip()
                        )
                        await agent_manager.process_memory_after_turn(
                            payload.session_id,
                            payload.message,
                            reply,
                            history,
                        )
                    except Exception as exc:
                        logger.warning("[Memory] 对话后记忆分析失败: %s", exc)
        except Exception as exc:
            persist_segments(fallback_content=f"请求失败: {str(exc) or 'unknown error'}")
            yield _sse("error", {"error": str(exc)})

    if payload.stream:
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    final_text = ""
    async for raw_event in event_generator():
        if raw_event.startswith("event: done"):
            final_text = raw_event
    return JSONResponse({"content": final_text})
